# API Documentation - Polymerized-LQG Replicator/Recycler

## Overview
This document provides complete API reference for the Polymerized-LQG Replicator/Recycler system. All APIs are UQ-validated and include comprehensive safety monitoring.

---

## Core API Classes

### UQCorrectedReplicatorMath

Main mathematical framework providing UQ-validated calculations for all replication operations.

```python
class UQCorrectedReplicatorMath:
    """
    UQ-validated mathematical framework for replicator physics
    
    Provides conservative, physics-based calculations with comprehensive
    validation and safety monitoring. All enhancement factors are validated
    against theoretical bounds and experimental constraints.
    """
```

#### Constructor
```python
def __init__(self)
```
Initializes the mathematical framework with UQ-validated parameters.

**Parameters**: None

**Returns**: Configured UQCorrectedReplicatorMath instance

**Example**:
```python
math_framework = UQCorrectedReplicatorMath()
```

#### calculate_total_energy_target()
```python
def calculate_total_energy_target(self, mass_kg: float) -> dict
```
Calculate realistic energy requirements for object replication.

**Parameters**:
- `mass_kg` (float): Mass of object to replicate in kilograms (range: 0.001-1000)

**Returns**:
```python
{
    'base_energy_j': float,           # Base mc² energy in Joules
    'enhanced_energy_j': float,       # UQ-corrected energy requirement
    'dematerialization_energy_j': float,  # Energy for dematerialization (50%)
    'rematerialization_energy_j': float,  # Energy for rematerialization (50%)
    'enhancement_factor': float,      # Total enhancement achieved (typically 484×)
    'reduction_factor': float,        # Energy reduction from conventional (0.336)
    'peak_power_mw': float,          # Peak power requirement in MW
    'cycle_time_minutes': float,     # Estimated cycle completion time
    'uq_validation_score': float    # UQ compliance score (0-1)
}
```

**Raises**:
- `ValueError`: If mass_kg is outside valid range
- `PhysicsValidationError`: If calculations fail UQ validation

**Example**:
```python
result = math_framework.calculate_total_energy_target(1.0)
print(f"Energy required: {result['enhanced_energy_j']/1e9:.2f} GJ")
print(f"Enhancement factor: {result['enhancement_factor']:.1f}×")
```

#### validate_physics_consistency()
```python
def validate_physics_consistency(self) -> dict
```
Validate all physics parameters for consistency and realism.

**Parameters**: None

**Returns**:
```python
{
    'overall_valid': bool,           # True if all checks pass
    'compliance_score': float,       # Overall compliance (0-1)
    'individual_checks': {
        'energy_conservation': bool,
        'thermodynamic_limits': bool,
        'quantum_mechanical_bounds': bool,
        'general_relativity_compatibility': bool,
        'causality_preservation': bool,
        'enhancement_factor_realistic': bool,
        'parameter_consistency': bool
    },
    'failed_checks': list,          # List of failed validation checks
    'recommendations': list         # Recommended corrections if any failures
}
```

**Example**:
```python
validation = math_framework.validate_physics_consistency()
if validation['overall_valid']:
    print(f"✅ Physics validation passed ({validation['compliance_score']:.1%})")
else:
    print(f"❌ Validation failures: {validation['failed_checks']}")
```

#### generate_report()
```python
def generate_report(self, mass_kg: float) -> str
```
Generate comprehensive replication analysis report.

**Parameters**:
- `mass_kg` (float): Mass for analysis in kilograms

**Returns**:
- `str`: Formatted markdown report with complete analysis

**Example**:
```python
report = math_framework.generate_report(1.0)
print(report)
```

---

### ReplicatorPhysics

Core physics engine implementing matter dematerialization and rematerialization.

```python
class ReplicatorPhysics:
    """
    Core physics engine for matter replication
    
    Implements LQG-based dematerialization fields, polymer-enhanced fusion
    reactors, and comprehensive physics simulation for replication cycles.
    """
```

#### Constructor
```python
def __init__(self, shell_geometry: LQGShellGeometry, fusion_reactor: PolymerFusionReactor)
```

**Parameters**:
- `shell_geometry` (LQGShellGeometry): Configured field geometry
- `fusion_reactor` (PolymerFusionReactor): Power generation system

**Example**:
```python
shell = LQGShellGeometry(radius_inner=0.5, radius_outer=0.6)
reactor = PolymerFusionReactor(base_power_output=10e6)
physics = ReplicatorPhysics(shell, reactor)
```

#### dematerialize_object()
```python
def dematerialize_object(self, mass_kg: float, location: tuple, safety_checks: bool = True) -> dict
```
Convert physical object into quantum pattern data.

**Parameters**:
- `mass_kg` (float): Object mass in kilograms
- `location` (tuple): 3D position (x, y, z) in meters
- `safety_checks` (bool): Enable real-time safety monitoring (default: True)

**Returns**:
```python
{
    'success': bool,                 # True if dematerialization successful
    'pattern_data': {
        'quantum_state': np.ndarray, # Quantum state information
        'atomic_structure': dict,    # Atomic composition and bonds
        'molecular_data': dict,      # Molecular structure information
        'metadata': {
            'mass_kg': float,
            'volume_m3': float,
            'density_kg_m3': float,
            'composition': dict
        }
    },
    'energy_used_j': float,         # Actual energy consumed
    'time_seconds': float,          # Dematerialization duration
    'field_strength_max': float,    # Peak field strength achieved
    'pattern_fidelity': float,      # Pattern capture fidelity (0-1)
    'safety_status': dict,          # Safety monitoring results
    'uq_validation': dict          # UQ compliance during operation
}
```

**Raises**:
- `SafetyLimitError`: If safety parameters exceed limits
- `FieldGenerationError`: If LQG fields cannot be established
- `PatternCaptureError`: If pattern fidelity is insufficient

**Example**:
```python
result = physics.dematerialize_object(
    mass_kg=1.0,
    location=(0, 0, 0),
    safety_checks=True
)

if result['success']:
    print(f"✅ Dematerialization complete in {result['time_seconds']:.1f}s")
    print(f"   Pattern fidelity: {result['pattern_fidelity']:.3f}")
else:
    print(f"❌ Dematerialization failed: {result.get('error_message')}")
```

#### rematerialize_object()
```python
def rematerialize_object(self, pattern_data: dict, mass_kg: float, location: tuple, 
                        safety_checks: bool = True) -> dict
```
Reconstruct physical object from quantum pattern data.

**Parameters**:
- `pattern_data` (dict): Quantum pattern information from dematerialization
- `mass_kg` (float): Expected object mass in kilograms
- `location` (tuple): Target 3D position (x, y, z) in meters
- `safety_checks` (bool): Enable real-time safety monitoring (default: True)

**Returns**:
```python
{
    'success': bool,                # True if rematerialization successful
    'reconstructed_object': {
        'mass_kg': float,           # Actual reconstructed mass
        'volume_m3': float,         # Reconstructed volume
        'position': tuple,          # Final position (x, y, z)
        'integrity_score': float    # Structural integrity (0-1)
    },
    'energy_used_j': float,        # Actual energy consumed
    'time_seconds': float,         # Rematerialization duration
    'reconstruction_fidelity': float,  # Overall reconstruction quality
    'atomic_accuracy': float,      # Atomic-level reconstruction accuracy
    'molecular_accuracy': float,   # Molecular-level reconstruction accuracy
    'safety_status': dict,         # Safety monitoring results
    'uq_validation': dict         # UQ compliance during operation
}
```

**Example**:
```python
result = physics.rematerialize_object(
    pattern_data=demat_result['pattern_data'],
    mass_kg=1.0,
    location=(1, 0, 0),
    safety_checks=True
)

if result['success']:
    print(f"✅ Rematerialization complete in {result['time_seconds']:.1f}s")
    print(f"   Reconstruction fidelity: {result['reconstruction_fidelity']:.3f}")
```

---

### ReplicatorController

High-level control system coordinating all replication operations.

```python
class ReplicatorController:
    """
    High-level control system for replication operations
    
    Coordinates physics engine, safety systems, and user interface for
    complete replication cycle management with adaptive control and
    comprehensive error handling.
    """
```

#### Constructor
```python
def __init__(self, physics_engine: ReplicatorPhysics, safety_monitor: SafetyMonitor,
             control_frequency: float = 10.0)
```

**Parameters**:
- `physics_engine` (ReplicatorPhysics): Core physics simulation engine
- `safety_monitor` (SafetyMonitor): Real-time safety monitoring system
- `control_frequency` (float): Control loop frequency in Hz (default: 10.0)

#### execute_replication_cycle()
```python
def execute_replication_cycle(self, mass_kg: float, source_location: tuple,
                            target_location: tuple, target_fidelity: float = 0.999,
                            enable_advanced_monitoring: bool = True) -> dict
```
Execute complete replication cycle with safety monitoring.

**Parameters**:
- `mass_kg` (float): Object mass to replicate (kg)
- `source_location` (tuple): Source position (x, y, z) in meters
- `target_location` (tuple): Target position (x, y, z) in meters  
- `target_fidelity` (float): Minimum acceptable pattern fidelity (0-1)
- `enable_advanced_monitoring` (bool): Enable detailed performance monitoring

**Returns**:
```python
{
    'overall_success': bool,        # True if complete cycle successful
    'cycle_phases': {
        'initialization': dict,     # Initialization results
        'dematerialization': dict,  # Dematerialization phase results
        'pattern_storage': dict,    # Pattern storage and verification
        'rematerialization': dict   # Rematerialization phase results
    },
    'summary': {
        'total_time_minutes': float,    # Complete cycle duration
        'total_energy_gj': float,       # Total energy consumed (GJ)
        'enhancement_factor': float,    # Achieved enhancement factor
        'overall_fidelity': float,      # End-to-end fidelity
        'mass_conservation_error': float,  # Mass conservation accuracy
        'position_accuracy_mm': float   # Positioning accuracy
    },
    'performance_metrics': {
        'energy_efficiency': float,    # Overall energy efficiency
        'time_efficiency': float,      # Time vs theoretical optimum
        'safety_margin_average': float, # Average safety margins maintained
        'control_stability': float     # Control system stability metric
    },
    'safety_report': dict,          # Comprehensive safety analysis
    'uq_validation_report': dict,   # UQ compliance throughout cycle
    'abort_reason': str            # Reason if cycle was aborted
}
```

**Example**:
```python
controller = ReplicatorController(physics, safety_monitor)

result = controller.execute_replication_cycle(
    mass_kg=1.0,
    source_location=(0, 0, 0),
    target_location=(1, 0, 0),
    target_fidelity=0.999
)

if result['overall_success']:
    print(f"✅ Replication successful!")
    print(f"   Duration: {result['summary']['total_time_minutes']:.1f} minutes")
    print(f"   Energy: {result['summary']['total_energy_gj']:.2f} GJ")
    print(f"   Fidelity: {result['summary']['overall_fidelity']:.3f}")
else:
    print(f"❌ Replication failed: {result['abort_reason']}")
```

---

## Supporting Classes

### LQGShellGeometry

Defines the geometric configuration of dematerialization fields.

```python
class LQGShellGeometry:
    def __init__(self, radius_inner: float = 0.5, radius_outer: float = 0.6,
                 transition_width: float = 0.1, field_strength_max: float = 1e12)
```

**Key Methods**:
- `field_strength(r: float) -> float`: Calculate field strength at radius r
- `volume_integral() -> float`: Calculate total field volume
- `energy_density(r: float) -> float`: Calculate energy density at radius r

### PolymerFusionReactor

Polymer-enhanced fusion reactor for power generation.

```python
class PolymerFusionReactor:
    def __init__(self, base_power_output: float = 10e6, 
                 polymer_enhancement_factor: float = 1.15)
```

**Key Methods**:
- `available_power(plasma_temp_kev: float, magnetic_field_tesla: float) -> float`
- `fuel_consumption_rate() -> dict`
- `thermal_efficiency() -> float`

### SafetyMonitor

Real-time safety monitoring and emergency response system.

```python
class SafetyMonitor:
    def __init__(self, check_frequency_hz: float = 100)
```

**Key Methods**:
- `monitor_cycle(system_state: dict) -> dict`
- `validate_safety_parameters(parameters: dict) -> dict`
- `trigger_emergency_shutdown(reason: str) -> dict`

---

## Utility Functions

### Performance Analysis Functions

```python
def analyze_replication_performance(mass_kg: float, enhancement_factor: float = 484) -> dict:
    """Analyze expected performance metrics for given parameters"""

def compare_to_conventional_methods(mass_kg: float) -> dict:
    """Compare replicator performance to conventional matter manipulation"""

def calculate_scaling_performance(mass_range: list) -> list:
    """Calculate performance scaling across mass range"""
```

### Safety Utility Functions

```python
def validate_safety_parameters(system_state: dict) -> dict:
    """Validate all system parameters against safety limits"""

def emergency_shutdown_analysis(trigger_type: str) -> dict:
    """Analyze emergency shutdown requirements"""

def radiation_safety_check(detector_readings: dict) -> dict:
    """Monitor radiation levels from fusion and exotic fields"""
```

### UQ Validation Functions

```python
def validate_enhancement_claims(enhancement_factors: dict) -> dict:
    """Validate enhancement factor claims against physics bounds"""

def cross_system_consistency_check(systems: list) -> dict:
    """Check parameter consistency across integrated systems"""

def physics_compliance_analysis(calculations: dict) -> dict:
    """Comprehensive physics compliance validation"""
```

---

## Integration APIs

### Cross-Repository Integration

```python
class CrossRepositoryIntegration:
    """Integration with other physics frameworks"""
    
    def integrate_with_transporter(self, transporter_system) -> dict:
        """Integrate with matter transporter for transport+replication"""
    
    def integrate_with_fusion_framework(self, fusion_system) -> dict:
        """Integrate with polymer fusion framework for power"""
    
    def integrate_with_exotic_energy(self, exotic_system) -> dict:
        """Integrate with exotic energy generation for LQG fields"""
```

### Data Exchange Formats

```python
# Pattern Data Format
pattern_data_format = {
    'quantum_state': {
        'wavefunctions': np.ndarray,    # Quantum state wavefunctions
        'entanglement_map': dict,       # Quantum entanglement structure
        'coherence_times': np.ndarray   # Decoherence timescales
    },
    'atomic_structure': {
        'atomic_positions': np.ndarray,  # 3D atomic coordinates
        'atomic_numbers': np.ndarray,    # Atomic number for each atom
        'bond_structure': dict,          # Chemical bond information
        'electronic_structure': dict    # Electronic configuration
    },
    'metadata': {
        'timestamp': str,               # ISO format timestamp
        'replicator_id': str,          # Source replicator identifier
        'uq_validation_hash': str,     # UQ validation checksum
        'safety_certification': dict   # Safety parameter certification
    }
}

# Integration Message Format
integration_message_format = {
    'message_type': str,               # 'request', 'response', 'alert'
    'source_system': str,              # Source system identifier
    'target_system': str,              # Target system identifier
    'payload': dict,                   # System-specific data
    'uq_validation': dict,            # UQ compliance certification
    'safety_status': dict,            # Safety parameter status
    'timestamp': str                   # ISO format timestamp
}
```

---

## Error Handling

### Exception Classes

```python
class PhysicsValidationError(Exception):
    """Raised when physics calculations fail UQ validation"""

class SafetyLimitError(Exception):
    """Raised when safety parameters exceed acceptable limits"""

class FieldGenerationError(Exception):
    """Raised when LQG fields cannot be properly established"""

class PatternCaptureError(Exception):
    """Raised when pattern capture fidelity is insufficient"""

class EnergyBalanceError(Exception):
    """Raised when energy balance becomes unstable"""

class IntegrationError(Exception):
    """Raised when cross-system integration fails"""
```

### Error Response Format

```python
error_response_format = {
    'error_type': str,                # Exception class name
    'error_message': str,             # Human-readable error description
    'error_code': int,                # Numeric error code
    'system_state': dict,             # System state when error occurred
    'recommended_actions': list,       # Suggested remediation steps
    'safety_implications': dict,       # Safety analysis of error condition
    'recovery_possible': bool,         # Whether system can recover automatically
    'manual_intervention_required': bool,  # Whether manual intervention needed
    'timestamp': str                   # ISO format error timestamp
}
```

---

## Configuration

### System Configuration Format

```python
system_config = {
    'physics_parameters': {
        'mu_polymer': 0.15,                    # Polymer parameter
        'enhancement_factor_target': 484,      # Target enhancement factor
        'safety_factor_minimum': 1.1,          # Minimum safety margin
        'energy_balance_range': (0.8, 1.5)     # Stable energy balance range
    },
    'hardware_limits': {
        'max_field_strength': 1e12,            # Maximum field strength (V/m)
        'max_plasma_temperature': 100,         # Maximum plasma temp (keV)
        'max_power_output': 50e6,              # Maximum power output (W)
        'max_object_mass': 1000                # Maximum object mass (kg)
    },
    'safety_settings': {
        'monitoring_frequency': 100,           # Safety check frequency (Hz)
        'emergency_shutdown_time': 0.050,      # Emergency shutdown time (s)
        'radiation_limits': {
            'neutron_flux_max': 1e10,          # neutrons/cm²/s
            'gamma_dose_rate_max': 0.1,        # Sv/h
            'containment_pressure_max': 1e6    # Pa
        }
    },
    'performance_targets': {
        'cycle_time_target': 120,              # Target cycle time (s)
        'energy_efficiency_target': 0.85,      # Target energy efficiency
        'pattern_fidelity_minimum': 0.999,     # Minimum pattern fidelity
        'positioning_accuracy': 0.001          # Positioning accuracy (m)
    }
}
```

---

This API documentation provides complete reference for all public interfaces in the Polymerized-LQG Replicator/Recycler system. All APIs include comprehensive UQ validation and safety monitoring to ensure reliable, physics-based operation.

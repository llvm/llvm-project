#ifndef QUANTUM_TRANSFORMS_QUANTUMREWRITES_TD
#define QUANTUM_TRANSFORMS_QUANTUMREWRITES_TD

include "mlir/IR/OpBase.td"
include "mlir/Pass/PassBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "Dialect/Quantum/QuantumOps.td"

// def : Pat<(CNOTGateOp
//             (CNOTGateOp $q0, $q1):$res0__0)
//             (CNOTGateOp $q0, $q1):$res1__1)),
//           [(replaceWithValue $q0), (replaceWithValue $q1)]>;

// S.S = Z
def : Pat<(PhaseGateOp (PhaseGateOp $q)), (PauliZGateOp $q)>;
// S^-1 . S^-1 = Z
def : Pat<(PhaseDaggerGateOp (PhaseDaggerGateOp $q)), (PauliZGateOp $q)>;
// T.T = S
def : Pat<(TGateOp (TGateOp $q)), (PhaseGateOp $q)>;
// T^-1 . T^-1 = S^-1
def : Pat<(TDaggerGateOp (TDaggerGateOp $q)), (PhaseDaggerGateOp $q)>;

def : Pat<(RotateXOp $beta, (RotateXOp $alpha, $q)), (RotateXOp (AddFOp $alpha, $beta), $q)>;
def : Pat<(RotateYOp $beta, (RotateYOp $alpha, $q)), (RotateYOp (AddFOp $alpha, $beta), $q)>;
def : Pat<(RotateZOp $beta, (RotateZOp $alpha, $q)), (RotateZOp (AddFOp $alpha, $beta), $q)>;

// Merge adjacent U ops, if all angles are constants
// class mergeZYZ<component> : NativeCodeCall<"convertYZYtoZYZ($0, $1, $2)." # component # "()">;

// U(...).U(...) = U(...)
// U(theta, phi, lambda)
//           = U(theta2, phi2, lambda2).U(theta1, phi1, lambda1)
//           = Rz(phi2).Ry(theta2).Rz(lambda2+phi1).Ry(theta1).Rz(lambda1)
//           = Rz(phi2).Rz(phi').Ry(theta').Rz(lambda').Rz(lambda1)
//           = U(theta', phi2 + phi', lambda1 + lambda')
// def : Pat<(UniversalRotationGateOp
//             (ConstantFloatOp $theta2),
//             $phi2,
//             (ConstantFloatOp $lambda2),
//             (UniversalRotationGateOp
//                (ConstantFloatOp $theta1),
//                (ConstantFloatOp $phi1),
//                $lambda1,
//                $q)),
//           (UniversalRotationGateOp
//              (ConstantFloatOp (mergeZYZ<theta> theta1, theta2, lambda1)),
//              (AddFOp phi2, (ConstantFloatOp (mergeZYZ<phi>))),
//              (AddFOp lambda1, (ConstantFloatOp (mergeZYZ<lambda>))),
//              $q)>;

// https://qiskit.org/documentation/stubs/qiskit.transpiler.passes.RemoveDiagonalGatesBeforeMeasure.html
def : Pat<(Quantum_MeasureOp (RotateZOp $param, $q)), (Quantum_MeasureOp $q)>;
def : Pat<(Quantum_MeasureOp (PhaseGateOp $q)), (Quantum_MeasureOp $q)>;
def : Pat<(Quantum_MeasureOp (PhaseDaggerGateOp $q)), (Quantum_MeasureOp $q)>;
def : Pat<(Quantum_MeasureOp (TGateOp $q)), (Quantum_MeasureOp $q)>;
def : Pat<(Quantum_MeasureOp (TDaggerGateOp $q)), (Quantum_MeasureOp $q)>;
def : Pat<(Quantum_MeasureOp (PauliZGateOp $q)), (Quantum_MeasureOp $q)>;

def : Pat<(Quantum_MeasureQubitOp (RotateZOp $param, $q)), (Quantum_MeasureQubitOp $q)>;
def : Pat<(Quantum_MeasureQubitOp (PhaseGateOp $q)), (Quantum_MeasureQubitOp $q)>;
def : Pat<(Quantum_MeasureQubitOp (PhaseDaggerGateOp $q)), (Quantum_MeasureQubitOp $q)>;
def : Pat<(Quantum_MeasureQubitOp (TGateOp $q)), (Quantum_MeasureQubitOp $q)>;
def : Pat<(Quantum_MeasureQubitOp (TDaggerGateOp $q)), (Quantum_MeasureQubitOp $q)>;
def : Pat<(Quantum_MeasureQubitOp (PauliZGateOp $q)), (Quantum_MeasureQubitOp $q)>;

#endif // QUANTUM_TRANSFORMS_QUANTUMREWRITES_TD

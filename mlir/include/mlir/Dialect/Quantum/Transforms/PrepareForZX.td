#ifndef QUANTUM_TRANSFORMS_PREPAREFORZX_TD
#define QUANTUM_TRANSFORMS_PREPAREFORZX_TD

include "mlir/IR/OpBase.td"
include "mlir/Pass/PassBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "Dialect/Quantum/QuantumOps.td"

def Zero : NativeCodeCall<"$_builder.getF64FloatAttr(0.0)">;
def Pi : NativeCodeCall<"$_builder.getF64FloatAttr(M_PI)">;
def NegPi : NativeCodeCall<"$_builder.getF64FloatAttr(-M_PI)">;
def HalfPi : NativeCodeCall<"$_builder.getF64FloatAttr(M_PI/2)">;
def NegHalfPi : NativeCodeCall<"$_builder.getF64FloatAttr(-M_PI/2)">;
def QuarterPi : NativeCodeCall<"$_builder.getF64FloatAttr(M_PI/4)">;
def NegQuarterPi : NativeCodeCall<"$_builder.getF64FloatAttr(-M_PI/4)">;

def : Pat<(PauliXGateOp $q), (RotateXOp (ConstantOp (Pi)), $q)>;
def : Pat<(PauliYGateOp $q), (RotateYOp (ConstantOp (HalfPi)), $q)>;
def : Pat<(RotateYOp $alpha, $q),
            (RotateZOp (ConstantOp (HalfPi)),
              (RotateXOp $alpha,
                (RotateZOp (ConstantOp (NegHalfPi)), $q)))>;
def : Pat<(PauliZGateOp $q), (RotateZOp (ConstantOp (Pi)), $q)>;
def : Pat<(UniversalRotationGateOp $theta, $phi, $lamb, $q),
          (RotateZOp $lamb,
            (RotateYOp $theta,
              (RotateZOp $phi, $q)))>;

def : Pat<(PhaseGateOp $q), (RotateZOp (ConstantOp (HalfPi)), $q)>;
def : Pat<(PhaseDaggerGateOp $q), (RotateZOp (ConstantOp (NegHalfPi)), $q)>;
def : Pat<(TGateOp $q), (RotateZOp (ConstantOp (QuarterPi)), $q)>;
def : Pat<(TDaggerGateOp $q), (RotateZOp (ConstantOp (NegQuarterPi)), $q)>;

def : Pat<(HadamardGateOp $q),
          (RotateZOp (ConstantOp (HalfPi)),
            (RotateXOp (ConstantOp (HalfPi)),
              (RotateZOp (ConstantOp (HalfPi)), $q)))>;

// Remove unsupported ops
def : Pat<(Quantum_BarrierOp $q), (replaceWithValue $q)>;
#endif // QUANTUM_TRANSFORMS_PREPAREFORZX_TD

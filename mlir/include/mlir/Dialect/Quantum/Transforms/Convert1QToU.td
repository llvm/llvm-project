#ifndef QUANTUM_TRANSFORMS_CONVERT1QTOU_TD
#define QUANTUM_TRANSFORMS_CONVERT1QTOU_TD

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

def : Pat<(PauliXGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (Pi)),
             (ConstantOp (Zero)),
             (ConstantOp (Pi)),
             $q)>;

def : Pat<(PauliYGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (Pi)),
             (ConstantOp (HalfPi)),
             (ConstantOp (HalfPi)),
             $q)>;

def : Pat<(PauliZGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (Zero)),
             (ConstantOp (Zero)),
             (ConstantOp (Pi)),
             $q)>;

def : Pat<(HadamardGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (HalfPi)),
             (ConstantOp (Zero)),
             (ConstantOp (Pi)),
             $q)>;

def : Pat<(PhaseGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (Zero)),
             (ConstantOp (Zero)),
             (ConstantOp (HalfPi)),
             $q)>;

def : Pat<(PhaseDaggerGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (Zero)),
             (ConstantOp (Zero)),
             (ConstantOp (NegHalfPi)),
             $q)>;

def : Pat<(TGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (Zero)),
             (ConstantOp (Zero)),
             (ConstantOp (QuarterPi)),
             $q)>;

def : Pat<(TDaggerGateOp $q),
          (UniversalRotationGateOp
             (ConstantOp (Zero)),
             (ConstantOp (Zero)),
             (ConstantOp (NegQuarterPi)),
             $q)>;

def : Pat<(RotateXOp $alpha, $q),
          (UniversalRotationGateOp
             $alpha,
             (ConstantOp (NegHalfPi)),
             (ConstantOp (HalfPi)),
             $q)>;

def : Pat<(RotateYOp $alpha, $q),
          (UniversalRotationGateOp
             $alpha,
             (ConstantOp (Zero)),
             (ConstantOp (Zero)),
             $q)>;

def : Pat<(RotateZOp $alpha, $q),
          (UniversalRotationGateOp
             (ConstantOp (Zero)),
             (ConstantOp (Zero)),
             $alpha,
             $q)>;

#endif // QUANTUM_TRANSFORMS_CONVERT1QTOU_TD

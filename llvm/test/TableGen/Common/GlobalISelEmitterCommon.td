// Common target boilerplate for GlobalISelEmitter tests.
def MyTargetISA : InstrInfo;
def MyTarget : Target { let InstructionSet = MyTargetISA; }

class MyTargetGenericInstruction : GenericInstruction {
  let Namespace = "MyTarget";
}

def R0 : Register<"r0"> { let Namespace = "MyTarget"; }
def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add R0)>;
def GPR32Op : RegisterOperand<GPR32>;
def F0 : Register<"f0"> { let Namespace = "MyTarget"; }
def FPR32 : RegisterClass<"MyTarget", [f32], 32, (add F0)>;
def FPR32Op : RegisterOperand<FPR32>;
def B0 : Register<"b0"> { let Namespace = "MyTarget"; }
def GPR8 : RegisterClass<"MyTarget", [i8], 8, (add B0)>;
def GPR8Op : RegisterOperand<GPR8>;


def V0 : Register<"v0"> { let Namespace = "MyTarget"; }
def VecReg128 : RegisterClass<"MyTarget", [v4i32], 128, (add V0)>;

def p0 : PtrValueType <i32, 0>;

class I<dag OOps, dag IOps, list<dag> Pat>
  : Instruction {
  let Namespace = "MyTarget";
  let OutOperandList = OOps;
  let InOperandList = IOps;
  let Pattern = Pat;
}

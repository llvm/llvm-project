// RUN: llvm-tblgen -gen-instr-info -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def TestInstrInfo : InstrInfo {}

def TestTarget : Target {
  let InstructionSet = TestInstrInfo;
}

def R0 : Register<"r0">;
def R1 : Register<"r1">;
def R2 : Register<"r2">;
def R3 : Register<"r3">;
def R4 : Register<"r4">;
def R5 : Register<"r5">;
def R6 : Register<"r6">;
def R7 : Register<"r7">;

def RC0 : RegisterClass<"Test", [i32], 0, (add R0)>;
def RC1 : RegisterClass<"Test", [i32], 0, (add R1)>;
def RC2 : RegisterClass<"Test", [i32], 0, (add R2)>;
def RC3 : RegisterClass<"Test", [i32], 0, (add R3)>;
def RC4 : RegisterClass<"Test", [i32], 0, (add R4)>;
def RC5 : RegisterClass<"Test", [i32], 0, (add R5)>;
def RC6 : RegisterClass<"Test", [i32], 0, (add R6)>;
def RC7 : RegisterClass<"Test", [i32], 0, (add R7)>;

class TestInstruction<dag Operands> : Instruction {
  let Namespace = "Test";
  let OutOperandList = (outs);
  let InOperandList = Operands;
  let hasSideEffects = false;
}

def Long : TestInstruction<
  (ins RC0:$r0, RC1:$r1, RC2:$r2, RC3:$r3,
       RC4:$r4, RC5:$r5, RC6:$r6, RC7:$r7)>;
def Suffix : TestInstruction<
  (ins RC3:$r3, RC4:$r4, RC5:$r5, RC6:$r6, RC7:$r7)>;

defm : RemapAllTargetPseudoPointerOperands<RC0>;

// CHECK: { {{[0-9]+}}, 5, 0, {{.*}}TestTargetOpInfoBase {{[+]}} [[#SUFFIX_OFFSET:]], {{.*}} },  // Suffix
// CHECK: { {{[0-9]+}}, 8, 0, {{.*}}TestTargetOpInfoBase {{[+]}} [[#SUFFIX_OFFSET-3]], {{.*}} },  // Long

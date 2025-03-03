// RUN: llvm-tblgen -gen-instr-info -I %p/../../include %s | FileCheck %s

// Check that OpName enum and getNamedOperandIdx are as expected.

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { }

def arch : Target {
  let InstructionSet = archInstrInfo;
}

class InstBase : Instruction {
  let Namespace = "MyNamespace";
  let UseNamedOperandTable = 1;
  let Size = 1;
  field bits<8> Inst;
}

def Reg : Register<"reg">;
def RegClass : RegisterClass<"foo", [i32], 0, (add Reg)>;

def OpA : Operand<i32>;
def OpB : Operand<i32>;

def RegOp : RegisterOperand<RegClass>;

def InstA : InstBase {
  let OutOperandList = (outs OpA:$a);
  let InOperandList = (ins OpB:$b, i32imm:$c);
}

def InstB : InstBase {
  let OutOperandList = (outs i32imm:$d);
  let InOperandList = (ins unknown:$x);
}

def InstC : InstBase {
  let OutOperandList = (outs RegClass:$d);
  let InOperandList = (ins RegOp:$x);
}

// InstD has UseNamedOperandTable = 0, so it won't be handled in
// getNamedOperandIdx().
def InstD : InstBase {
  let OutOperandList = (outs RegClass:$e);
  let InOperandList = (ins RegOp:$f);
  let UseNamedOperandTable = 0;
}

// CHECK: #ifdef GET_INSTRINFO_OPERAND_ENUM
// CHECK: #undef GET_INSTRINFO_OPERAND_ENUM
// CHECK: namespace llvm::MyNamespace {
// CHECK: enum class OpName {
// CHECK:   a = 0,
// CHECK:   b = 1,
// CHECK:   c = 2,
// CHECK:   d = 3,
// CHECK:   x = 4,
// CHECK:   NUM_OPERAND_NAMES = 5,
// CHECK: }; // enum class OpName
// CHECK: } // end namespace llvm::MyNamespace
// CHECK: #endif //GET_INSTRINFO_OPERAND_ENUM

// CHECK: #ifdef GET_INSTRINFO_NAMED_OPS
// CHECK: #undef GET_INSTRINFO_NAMED_OPS
// CHECK: namespace llvm::MyNamespace {
// CHECK: LLVM_READONLY
// CHECK: int16_t getNamedOperandIdx(uint16_t Opcode, OpName Name) {
// CHECK:  assert(Name != OpName::NUM_OPERAND_NAMES);
// CHECK:  static constexpr int8_t OperandMap[][5] = {
// CHECK:      {0, 1, 2, -1, -1, },
// CHECK:      {-1, -1, -1, 0, 1, },
// CHECK:    };
// CHECK:    switch(Opcode) {
// CHECK:    case MyNamespace::InstA:
// CHECK:      return OperandMap[0][static_cast<unsigned>(Name)];
// CHECK:    case MyNamespace::InstB:
// CHECK:    case MyNamespace::InstC:
// CHECK:      return OperandMap[1][static_cast<unsigned>(Name)];
// CHECK:    default: return -1;
// CHECK:    }
// CHECK:  }
// CHECK:  } // end namespace llvm::MyNamespace
// CHECK:  #endif //GET_INSTRINFO_NAMED_OPS

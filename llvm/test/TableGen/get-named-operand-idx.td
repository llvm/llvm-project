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

// CHECK-LABEL: #ifdef GET_INSTRINFO_OPERAND_ENUM
// CHECK-NEXT:  #undef GET_INSTRINFO_OPERAND_ENUM
// CHECK-NEXT:  namespace llvm::MyNamespace {
// CHECK-NEXT:  enum class OpName : uint8_t {
// CHECK-NEXT:    a = 0,
// CHECK-NEXT:    b = 1,
// CHECK-NEXT:    c = 2,
// CHECK-NEXT:    d = 3,
// CHECK-NEXT:    x = 4,
// CHECK-NEXT:    NUM_OPERAND_NAMES = 5,
// CHECK-NEXT:  }; // enum class OpName
// CHECK-EMPTY:
// CHECK-NEXT:  LLVM_READONLY int16_t getNamedOperandIdx(uint16_t Opcode, OpName Name);
// CHECK-NEXT:  LLVM_READONLY OpName getOperandIdxName(uint16_t Opcode, int16_t Idx);
// CHECK-NEXT:  } // end namespace llvm::MyNamespace
// CHECK-NEXT:  #endif //GET_INSTRINFO_OPERAND_ENUM

// CHECK-LABEL: #ifdef GET_INSTRINFO_NAMED_OPS
// CHECK-NEXT:  #undef GET_INSTRINFO_NAMED_OPS
// CHECK-NEXT:  namespace llvm::MyNamespace {
// CHECK-NEXT:  LLVM_READONLY static uint8_t getInstructionIndexForOpLookup(uint16_t Opcode) {
// CHECK-NEXT:    static constexpr uint8_t InstructionIndex[] = {
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// CHECK-NEXT:      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 0,
// CHECK-NEXT:    };
// CHECK-NEXT:    return InstructionIndex[Opcode];
// CHECK-NEXT:  }
// CHECK-NEXT:  LLVM_READONLY int16_t getNamedOperandIdx(uint16_t Opcode, OpName Name) {
// CHECK-NEXT:    assert(Name != OpName::NUM_OPERAND_NAMES);
// CHECK-NEXT:    static constexpr int8_t OperandMap[][5] = {
// CHECK-NEXT:      {-1, -1, -1, -1, -1, },
// CHECK-NEXT:      {0, 1, 2, -1, -1, },
// CHECK-NEXT:      {-1, -1, -1, 0, 1, },
// CHECK-NEXT:    };
// CHECK-NEXT:    unsigned InstrIdx = getInstructionIndexForOpLookup(Opcode);
// CHECK-NEXT:    return OperandMap[InstrIdx][(unsigned)Name];
// CHECK-NEXT:  }
// CHECK-NEXT:  LLVM_READONLY OpName getOperandIdxName(uint16_t Opcode, int16_t Idx) {
// CHECK-NEXT:    assert(Idx >= 0 && Idx < 3);
// CHECK-NEXT:    static constexpr OpName OperandMap[][3] = {
// CHECK-NEXT:      {OpName::NUM_OPERAND_NAMES, OpName::NUM_OPERAND_NAMES, OpName::NUM_OPERAND_NAMES, },
// CHECK-NEXT:      {OpName::a, OpName::b, OpName::c, },
// CHECK-NEXT:      {OpName::d, OpName::x, OpName::NUM_OPERAND_NAMES, },
// CHECK-NEXT:    };
// CHECK-NEXT:    unsigned InstrIdx = getInstructionIndexForOpLookup(Opcode);
// CHECK-NEXT:    return OperandMap[InstrIdx][(unsigned)Idx];
// CHECK-NEXT:  }
// CHECK-NEXT:  } // end namespace llvm::MyNamespace
// CHECK-NEXT:  #endif //GET_INSTRINFO_NAMED_OPS

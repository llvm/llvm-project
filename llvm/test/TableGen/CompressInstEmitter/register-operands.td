// RUN: llvm-tblgen -gen-compress-inst-emitter -I %p/../../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def TestInstrInfo : InstrInfo;
def TestAsmWriter : AsmWriter {
  let PassSubtarget = 1;
}
def Test : Target {
  let InstructionSet = TestInstrInfo;
  let AssemblyWriters = [TestAsmWriter];
}
def R0 : Register<"r0">;
def R1 : Register<"r1">;
def GPR : RegisterClass<"Test", [i32], 32, (add R0, R1)>;
def GPROp : RegisterOperand<GPR>;

class TestInst<dag outs, dag ins, int size> : Instruction {
  let Namespace = "Test";
  let OutOperandList = outs;
  let InOperandList = ins;
  let Size = size;
}
def Wide : TestInst<(outs GPROp:$dst), (ins GPROp:$src), 4>;
def NarrowZero : TestInst<(outs GPROp:$dst), (ins), 2>;
def NarrowTied : TestInst<(outs GPROp:$dst), (ins GPROp:$src), 2> {
  let Constraints = "$src = $dst";
}

def WideTied : TestInst<(outs GPROp:$dst), (ins GPROp:$src), 4> {
  let Constraints = "$src = $dst";
}
def : CompressPat<(WideTied GPROp:$dst), (NarrowTied GPROp:$dst)>;

// Fixed and repeated registers must work with RegisterOperand wrappers as
// well as bare register classes.
def : CompressPat<(Wide GPROp:$dst, R0), (NarrowZero GPROp:$dst)>;
def : CompressPat<(Wide GPROp:$dst, GPROp:$dst), (NarrowTied GPROp:$dst)>;

def SmallImm : Operand<i32>, ImmLeaf<i32, [{ return isUInt<4>(Imm); }]> {
  let MCOperandPredicate = [{ return MCOp.isImm() && isUInt<4>(MCOp.getImm()); }];
}
def WideImm : TestInst<(outs GPROp:$dst), (ins SmallImm:$imm), 4>;
def NarrowImm : TestInst<(outs GPROp:$dst), (ins), 2>;
def : CompressPat<(WideImm GPROp:$dst, 7), (NarrowImm GPROp:$dst)>;

def WideOnly : TestInst<(outs GPROp:$dst), (ins), 4>;
def NarrowOnly : TestInst<(outs GPROp:$dst), (ins), 2>;
let isCompressOnly = true in
  def : CompressPat<(WideOnly GPROp:$dst), (NarrowOnly GPROp:$dst)>;

def WideVariadic : TestInst<(outs GPROp:$dst),
                            (ins GPROp:$src, variable_ops), 4>;
def NarrowVariadic : TestInst<(outs GPROp:$dst),
                              (ins GPROp:$src, variable_ops), 2>;
def : CompressPat<(WideVariadic GPROp:$dst, GPROp:$src),
                  (NarrowVariadic GPROp:$dst, GPROp:$src)>;

// CHECK-LABEL: #ifdef GEN_COMPRESS_MACHINE_INSTR
// CHECK: #ifndef Test_MACHINE_INSTR_REWRITE
// CHECK: static inline void rewriteMachineInstr(
// CHECK: Operands[I].setImplicit(false);
// CHECK: unsigned NumFixedOperands = MI.getDesc().getNumOperands();
// CHECK: Operands.push_back(MI.getOperand(I));
// CHECK: MO.setImplicit();
// CHECK: MI.getMF()->makeDebugValueSubstitution(
// CHECK-LABEL: static bool compressInst(MachineInstr &MI,
// CHECK-NOT: isVariadic()
// CHECK-NOT: MI.operands()
// CHECK: case Test::Wide:
// CHECK: Operands.push_back(MI.getOperand(0));
// CHECK: Operands.push_back(MI.getOperand(1));
// CHECK-NEXT: rewriteMachineInstr(MI, STI.getInstrInfo()->get(Test::NarrowTied), std::move(Operands), {0, 1});
// CHECK: rewriteMachineInstr(MI, STI.getInstrInfo()->get(Test::NarrowZero), std::move(Operands), {0});
// CHECK: case Test::WideTied:
// CHECK: Operands.push_back(MI.getOperand(0));
// CHECK: Operands.push_back(MI.getOperand(1));
// CHECK-NEXT: rewriteMachineInstr(MI, STI.getInstrInfo()->get(Test::NarrowTied), std::move(Operands), {0, 1});

// CHECK: case Test::WideVariadic:
// CHECK: rewriteMachineInstr(MI, STI.getInstrInfo()->get(Test::NarrowVariadic), std::move(Operands), {0, 1});

// CHECK-LABEL: #ifdef GEN_UNCOMPRESS_MACHINE_INSTR
// CHECK: #ifndef Test_MACHINE_INSTR_REWRITE
// CHECK: static bool TestValidateMachineOperandForUncompress(
// CHECK: return isUInt<4>(Imm);
// CHECK-LABEL: static bool uncompressInst(MachineInstr &MI,
// CHECK: case Test::NarrowImm:
// CHECK: Operands.push_back(MachineOperand::CreateImm(7));
// CHECK: rewriteMachineInstr(MI, STI.getInstrInfo()->get(Test::WideImm), std::move(Operands), {0, std::nullopt});
// CHECK-NOT: case Test::NarrowOnly:
// CHECK: case Test::NarrowTied:
// CHECK: Operands.push_back(MI.getOperand(0));
// CHECK: Operands.push_back(MI.getOperand(1));
// CHECK: rewriteMachineInstr(MI, STI.getInstrInfo()->get(Test::Wide), std::move(Operands), {0, 1});
// CHECK: case Test::NarrowVariadic:
// CHECK: rewriteMachineInstr(MI, STI.getInstrInfo()->get(Test::WideVariadic), std::move(Operands), {0, 1});
// CHECK: case Test::NarrowZero:
// CHECK: Operands.push_back(MI.getOperand(0));
// CHECK: Operands.push_back(MachineOperand::CreateReg(Test::R0, false));
// CHECK: rewriteMachineInstr(MI, STI.getInstrInfo()->get(Test::Wide), std::move(Operands), {0, std::nullopt});

// CHECK-LABEL: static bool isCompressibleInst(
// CHECK: case Test::Wide:
// CHECK: MI.getOperand(1).getReg() == MI.getOperand(0).getReg()
// CHECK: return true;
// CHECK: MI.getOperand(1).getReg() == Test::R0
// CHECK: return true;

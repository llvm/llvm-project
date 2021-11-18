// RUN: llvm-tblgen -gen-compress-inst-emitter -I %p/../../include %s | \
// RUN:     FileCheck --check-prefix=COMPRESS %s

// Check that combining conditions in AssemblerPredicate generates the correct
// output when using both the (all_of) AND operator, and the (any_of) OR
// operator in the RISC-V specific instruction compressor.

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo { }
def archAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

def arch : Target {
  let InstructionSet = archInstrInfo;
  let AssemblyWriters = [archAsmWriter];
}

let Namespace = "arch" in {
  def R0 : Register<"r0">;
}
def Regs : RegisterClass<"Regs", [i32], 32, (add R0)>;

class RVInst<int Opc, list<Predicate> Preds> : Instruction {
  let Size = 4;
  let OutOperandList = (outs);
  let InOperandList = (ins Regs:$r);
  field bits<32> Inst;
  let Inst = Opc;
  let AsmString = NAME # " $r";
  field bits<32> SoftFail = 0;
  let Predicates = Preds;
}
class RVInst16<int Opc, list<Predicate> Preds> : Instruction {
  let Size = 2;
  let OutOperandList = (outs);
  let InOperandList = (ins Regs:$r);
  field bits<16> Inst;
  let Inst = Opc;
  let AsmString = NAME # " $r";
  field bits<16> SoftFail = 0;
  let Predicates = Preds;
}

def AsmCond1 : SubtargetFeature<"cond1", "cond1", "true", "">;
def AsmCond2a: SubtargetFeature<"cond2a", "cond2a", "true", "">;
def AsmCond2b: SubtargetFeature<"cond2b", "cond2b", "true", "">;
def AsmCond3a: SubtargetFeature<"cond3a", "cond3a", "true", "">;
def AsmCond3b: SubtargetFeature<"cond3b", "cond3b", "true", "">;

def AsmPred1 : Predicate<"Pred1">, AssemblerPredicate<(all_of AsmCond1)>;
def AsmPred2 : Predicate<"Pred2">, AssemblerPredicate<(all_of AsmCond2a, AsmCond2b)>;
def AsmPred3 : Predicate<"Pred3">, AssemblerPredicate<(any_of AsmCond3a, AsmCond3b)>;

def BigInst : RVInst<1, [AsmPred1]>;

// COMPRESS-LABEL: static bool compressInst
// COMPRESS: case arch::BigInst
def SmallInst1 : RVInst16<1, []>;
def : CompressPat<(BigInst Regs:$r), (SmallInst1 Regs:$r), [AsmPred1]>;
// COMPRESS:      if (STI.getFeatureBits()[arch::AsmCond1] &&
// COMPRESS-NEXT: (MRI.getRegClass(arch::RegsRegClassID).contains(MI.getOperand(0).getReg()))) {
// COMPRESS-NEXT: // SmallInst1 $r

def SmallInst2 : RVInst16<2, []>;
def : CompressPat<(BigInst Regs:$r), (SmallInst2 Regs:$r), [AsmPred2]>;
// COMPRESS:      if (STI.getFeatureBits()[arch::AsmCond2a] &&
// COMPRESS-NEXT: STI.getFeatureBits()[arch::AsmCond2b] &&
// COMPRESS-NEXT: (MRI.getRegClass(arch::RegsRegClassID).contains(MI.getOperand(0).getReg()))) {
// COMPRESS-NEXT: // SmallInst2 $r

def SmallInst3 : RVInst16<2, []>;
def : CompressPat<(BigInst Regs:$r), (SmallInst3 Regs:$r), [AsmPred3]>;
// COMPRESS:      if ((STI.getFeatureBits()[arch::AsmCond3a] || STI.getFeatureBits()[arch::AsmCond3b]) &&
// COMPRESS-NEXT: (MRI.getRegClass(arch::RegsRegClassID).contains(MI.getOperand(0).getReg()))) {
// COMPRESS-NEXT: // SmallInst3 $r

def SmallInst4 : RVInst16<2, []>;
def : CompressPat<(BigInst Regs:$r), (SmallInst4 Regs:$r), [AsmPred1, AsmPred2]>;
// COMPRESS:      if (STI.getFeatureBits()[arch::AsmCond1] &&
// COMPRESS-NEXT: STI.getFeatureBits()[arch::AsmCond2a] &&
// COMPRESS-NEXT: STI.getFeatureBits()[arch::AsmCond2b] &&
// COMPRESS-NEXT: (MRI.getRegClass(arch::RegsRegClassID).contains(MI.getOperand(0).getReg()))) {
// COMPRESS-NEXT: // SmallInst4 $r

def SmallInst5 : RVInst16<2, []>;
def : CompressPat<(BigInst Regs:$r), (SmallInst5 Regs:$r), [AsmPred1, AsmPred3]>;
// COMPRESS:      if (STI.getFeatureBits()[arch::AsmCond1] &&
// COMPRESS-NEXT: (STI.getFeatureBits()[arch::AsmCond3a] || STI.getFeatureBits()[arch::AsmCond3b]) &&
// COMPRESS-NEXT: (MRI.getRegClass(arch::RegsRegClassID).contains(MI.getOperand(0).getReg()))) {
// COMPRESS-NEXT: // SmallInst5 $r

// COMPRESS-LABEL: static bool uncompressInst

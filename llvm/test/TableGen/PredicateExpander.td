// RUN: llvm-tblgen -gen-instr-info -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def TestInstrInfo : InstrInfo;
def Test : Target {
  let InstructionSet = TestInstrInfo;
}

let Namespace = "Test" in {
  def R0 : Register<"r0">;
  def GPR : RegisterClass<"GPR", [i32], 32, (add R0)>;

  let OutOperandList = (outs), InOperandList = (ins) in
  def Inst : Instruction;
}

defm : RemapAllTargetPseudoPointerOperands<GPR>;

def CheckImmEq
  : TIIPredicate<"checkImmEq", MCReturnStatement<CheckImmOperand<0, 7>>>;
def CheckImmEqString
  : TIIPredicate<"checkImmEqString",
      MCReturnStatement<CheckImmOperand_s<1, "Test::Value">>>;
def CheckImmSimple
  : TIIPredicate<"checkImmSimple", MCReturnStatement<CheckImmOperandSimple<1>>>;
def CheckImmLT
  : TIIPredicate<"checkImmLT", MCReturnStatement<CheckImmOperandLT<2, 17>>>;
def CheckImmLE
  : TIIPredicate<"checkImmLE", MCReturnStatement<CheckImmOperandLE<2, 18>>>;
def CheckImmGT
  : TIIPredicate<"checkImmGT", MCReturnStatement<CheckImmOperandGT<2, 19>>>;
def CheckImmGE
  : TIIPredicate<"checkImmGE", MCReturnStatement<CheckImmOperandGE<2, 20>>>;
def CheckImmRange
  : TIIPredicate<"checkImmRange", MCReturnStatement<CheckImmOperandRange<2, 21, 23>>>;

def CheckNotImmEq
  : TIIPredicate<"checkNotImmEq", MCReturnStatement<CheckNot<CheckImmOperand<1, 15>>>>;
def CheckNotImmSimple
  : TIIPredicate<"checkNotImmSimple", MCReturnStatement<CheckNot<CheckImmOperandSimple<1>>>>;

def CheckRegEq
  : TIIPredicate<"checkRegEq",
      MCReturnStatement<CheckRegOperand<1, R0>>>;
def CheckRegSimple
  : TIIPredicate<"checkRegSimple",
      MCReturnStatement<CheckRegOperandSimple<1>>>;

def CheckNotRegEq
  : TIIPredicate<"checkNotRegEq", MCReturnStatement<CheckNot<CheckRegOperand<2, R0>>>>;
def CheckNotRegSimple
  : TIIPredicate<"checkNotRegSimple", MCReturnStatement<CheckNot<CheckRegOperandSimple<2>>>>;

let FunctionMapper = "mapImm" in {
  def MappedImmEq : CheckImmOperand<2, 9>;
  def MappedImmEqString : CheckImmOperand_s<2, "Test::Value">;
  def MappedImmGE : CheckImmOperandGE<2, 25>;
  def MappedImmSimple : CheckImmOperandSimple<1>;
  def MappedImmLE : CheckImmOperandLE<2, 10>;
  def MappedImmLT : CheckImmOperandLT<2, 11>;
  def MappedImmGT : CheckImmOperandGT<2, 13>;
  def MappedImmRange : CheckImmOperandRange<2, 9, 31>;
}

let FunctionMapper = "mapReg" in {
  def MappedRegEq : CheckRegOperand<1, R0>;
  def MappedRegSimple : CheckRegOperandSimple<1>;
}

def CheckMappedImmEq
  : TIIPredicate<"checkMappedImmEq", MCReturnStatement<MappedImmEq>>;
def CheckMappedImmEqString
  : TIIPredicate<"checkMappedImmEqString", MCReturnStatement<MappedImmEqString>>;
def CheckMappedImmSimple
  : TIIPredicate<"checkMappedImmSimple", MCReturnStatement<MappedImmSimple>>;
def CheckMappedImmLT
  : TIIPredicate<"checkMappedImmLT", MCReturnStatement<MappedImmLT>>;
def CheckMappedImmLE
  : TIIPredicate<"checkMappedImmLE", MCReturnStatement<MappedImmLE>>;
def CheckMappedImmGT
  : TIIPredicate<"checkMappedImmGT", MCReturnStatement<MappedImmGT>>;
def CheckMappedImmGE
  : TIIPredicate<"checkMappedImmGE", MCReturnStatement<MappedImmGE>>;
def CheckMappedImmRange
  : TIIPredicate<"checkMappedImmRange", MCReturnStatement<MappedImmRange>>;

def CheckMappedRegEq
  : TIIPredicate<"checkMappedRegEq", MCReturnStatement<MappedRegEq>>;
def CheckMappedRegSimple
  : TIIPredicate<"checkMappedRegSimple", MCReturnStatement<MappedRegSimple>>;

// CHECK: bool TestInstrInfo::checkImmEq(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(0).isImm() && MI.getOperand(0).getImm() == 7);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkImmEqString(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(1).isImm() && MI.getOperand(1).getImm() == Test::Value);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkImmGE(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && MI.getOperand(2).getImm() >= 20);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkImmGT(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && MI.getOperand(2).getImm() > 19);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkImmLE(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && MI.getOperand(2).getImm() <= 18);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkImmLT(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && MI.getOperand(2).getImm() < 17);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkImmRange(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && MI.getOperand(2).getImm() >= 21 && MI.getOperand(2).getImm() <= 23);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkImmSimple(const MachineInstr &MI) {
// CHECK-NEXT:   return true;
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkMappedImmEq(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && mapImm(MI.getOperand(2).getImm()) == 9);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkMappedImmEqString(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && mapImm(MI.getOperand(2).getImm()) == Test::Value);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkMappedImmGE(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && mapImm(MI.getOperand(2).getImm()) >= 25);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkMappedImmGT(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && mapImm(MI.getOperand(2).getImm()) > 13);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkMappedImmLE(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && mapImm(MI.getOperand(2).getImm()) <= 10);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkMappedImmLT(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && mapImm(MI.getOperand(2).getImm()) < 11);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkMappedImmRange(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && mapImm(MI.getOperand(2).getImm()) >= 9 && mapImm(MI.getOperand(2).getImm()) <= 31);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkMappedImmSimple(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(1).isImm() && mapImm(MI.getOperand(1).getImm()));
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkMappedRegEq(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(1).isReg() && mapReg(MI.getOperand(1).getReg()) == Test::R0);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkMappedRegSimple(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(1).isReg() && mapReg(MI.getOperand(1).getReg()));
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkNotImmEq(const MachineInstr &MI) {
// CHECK-NEXT:   return !(MI.getOperand(1).isImm() && MI.getOperand(1).getImm() == 15);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkNotImmSimple(const MachineInstr &MI) {
// CHECK-NEXT:   return false;
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkNotRegEq(const MachineInstr &MI) {
// CHECK-NEXT:   return !(MI.getOperand(2).isReg() && MI.getOperand(2).getReg() == Test::R0);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkNotRegSimple(const MachineInstr &MI) {
// CHECK-NEXT:   return !(MI.getOperand(2).isReg() && MI.getOperand(2).getReg().isValid());
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkRegEq(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == Test::R0);
// CHECK-NEXT: }

// CHECK: bool TestInstrInfo::checkRegSimple(const MachineInstr &MI) {
// CHECK-NEXT:   return (MI.getOperand(1).isReg() && MI.getOperand(1).getReg().isValid());
// CHECK-NEXT: }

// CHECK: namespace llvm::Test_MC {

// CHECK: bool checkImmEq(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(0).isImm() && MI.getOperand(0).getImm() == 7);
// CHECK-NEXT: }

// CHECK: bool checkImmEqString(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(1).isImm() && MI.getOperand(1).getImm() == Test::Value);
// CHECK-NEXT: }

// CHECK: bool checkImmGE(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && MI.getOperand(2).getImm() >= 20);
// CHECK-NEXT: }

// CHECK: bool checkImmGT(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && MI.getOperand(2).getImm() > 19);
// CHECK-NEXT: }

// CHECK: bool checkImmLE(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && MI.getOperand(2).getImm() <= 18);
// CHECK-NEXT: }

// CHECK: bool checkImmLT(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && MI.getOperand(2).getImm() < 17);
// CHECK-NEXT: }

// CHECK: bool checkImmRange(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && MI.getOperand(2).getImm() >= 21 && MI.getOperand(2).getImm() <= 23);
// CHECK-NEXT: }

// CHECK: bool checkImmSimple(const MCInst &MI) {
// CHECK-NEXT:   return true;
// CHECK-NEXT: }

// CHECK: bool checkMappedImmEq(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && mapImm(MI.getOperand(2).getImm()) == 9);
// CHECK-NEXT: }

// CHECK: bool checkMappedImmEqString(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && mapImm(MI.getOperand(2).getImm()) == Test::Value);
// CHECK-NEXT: }

// CHECK: bool checkMappedImmGE(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && mapImm(MI.getOperand(2).getImm()) >= 25);
// CHECK-NEXT: }

// CHECK: bool checkMappedImmGT(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && mapImm(MI.getOperand(2).getImm()) > 13);
// CHECK-NEXT: }

// CHECK: bool checkMappedImmLE(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && mapImm(MI.getOperand(2).getImm()) <= 10);
// CHECK-NEXT: }

// CHECK: bool checkMappedImmLT(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && mapImm(MI.getOperand(2).getImm()) < 11);
// CHECK-NEXT: }

// CHECK: bool checkMappedImmRange(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(2).isImm() && mapImm(MI.getOperand(2).getImm()) >= 9 && mapImm(MI.getOperand(2).getImm()) <= 31);
// CHECK-NEXT: }

// CHECK: bool checkMappedImmSimple(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(1).isImm() && mapImm(MI.getOperand(1).getImm()));
// CHECK-NEXT: }

// CHECK: bool checkMappedRegEq(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(1).isReg() && mapReg(MI.getOperand(1).getReg()) == Test::R0);
// CHECK-NEXT: }

// CHECK: bool checkMappedRegSimple(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(1).isReg() && mapReg(MI.getOperand(1).getReg()));
// CHECK-NEXT: }

// CHECK: bool checkNotImmEq(const MCInst &MI) {
// CHECK-NEXT:   return !(MI.getOperand(1).isImm() && MI.getOperand(1).getImm() == 15);
// CHECK-NEXT: }

// CHECK: bool checkNotImmSimple(const MCInst &MI) {
// CHECK-NEXT:   return false;
// CHECK-NEXT: }

// CHECK: bool checkNotRegEq(const MCInst &MI) {
// CHECK-NEXT:   return !(MI.getOperand(2).isReg() && MI.getOperand(2).getReg() == Test::R0);
// CHECK-NEXT: }

// CHECK: bool checkNotRegSimple(const MCInst &MI) {
// CHECK-NEXT:   return !(MI.getOperand(2).isReg() && MI.getOperand(2).getReg().isValid());
// CHECK-NEXT: }

// CHECK: bool checkRegEq(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == Test::R0);
// CHECK-NEXT: }

// CHECK: bool checkRegSimple(const MCInst &MI) {
// CHECK-NEXT:   return (MI.getOperand(1).isReg() && MI.getOperand(1).getReg().isValid());
// CHECK-NEXT: }

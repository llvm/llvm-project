//===- MIRPrinter.cpp - MIR serialization format printer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the class that prints out the LLVM IR and machine
// functions using the MIR serialization format.
//
//===----------------------------------------------------------------------===//

#include "MIRPrinter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/GlobalISel/RegisterBank.h"
#include "llvm/CodeGen/MIRYamlMapping.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

namespace {

/// This structure describes how to print out stack object references.
struct FrameIndexOperand {
  std::string Name;
  unsigned ID;
  bool IsFixed;

  FrameIndexOperand(StringRef Name, unsigned ID, bool IsFixed)
      : Name(Name.str()), ID(ID), IsFixed(IsFixed) {}

  /// Return an ordinary stack object reference.
  static FrameIndexOperand create(StringRef Name, unsigned ID) {
    return FrameIndexOperand(Name, ID, /*IsFixed=*/false);
  }

  /// Return a fixed stack object reference.
  static FrameIndexOperand createFixed(unsigned ID) {
    return FrameIndexOperand("", ID, /*IsFixed=*/true);
  }
};

} // end anonymous namespace

namespace llvm {

/// This class prints out the machine functions using the MIR serialization
/// format.
class MIRPrinter {
  raw_ostream &OS;
  DenseMap<const uint32_t *, unsigned> RegisterMaskIds;
  /// Maps from stack object indices to operand indices which will be used when
  /// printing frame index machine operands.
  DenseMap<int, FrameIndexOperand> StackObjectOperandMapping;

public:
  MIRPrinter(raw_ostream &OS) : OS(OS) {}

  void print(const MachineFunction &MF);

  void convert(yaml::MachineFunction &MF, const MachineRegisterInfo &RegInfo,
               const TargetRegisterInfo *TRI);
  void convert(ModuleSlotTracker &MST, yaml::MachineFrameInfo &YamlMFI,
               const MachineFrameInfo &MFI);
  void convert(yaml::MachineFunction &MF,
               const MachineConstantPool &ConstantPool);
  void convert(ModuleSlotTracker &MST, yaml::MachineJumpTable &YamlJTI,
               const MachineJumpTableInfo &JTI);
  void convertStackObjects(yaml::MachineFunction &MF,
                           const MachineFrameInfo &MFI, MachineModuleInfo &MMI,
                           ModuleSlotTracker &MST,
                           const TargetRegisterInfo *TRI);

private:
  void initRegisterMaskIds(const MachineFunction &MF);
};

/// This class prints out the machine instructions using the MIR serialization
/// format.
class MIPrinter {
  raw_ostream &OS;
  ModuleSlotTracker &MST;
  const DenseMap<const uint32_t *, unsigned> &RegisterMaskIds;
  const DenseMap<int, FrameIndexOperand> &StackObjectOperandMapping;

public:
  MIPrinter(raw_ostream &OS, ModuleSlotTracker &MST,
            const DenseMap<const uint32_t *, unsigned> &RegisterMaskIds,
            const DenseMap<int, FrameIndexOperand> &StackObjectOperandMapping)
      : OS(OS), MST(MST), RegisterMaskIds(RegisterMaskIds),
        StackObjectOperandMapping(StackObjectOperandMapping) {}

  void print(const MachineBasicBlock &MBB);

  void print(const MachineInstr &MI);
  void printMBBReference(const MachineBasicBlock &MBB);
  void printIRBlockReference(const BasicBlock &BB);
  void printIRValueReference(const Value &V);
  void printStackObjectReference(int FrameIndex);
  void printOffset(int64_t Offset);
  void printTargetFlags(const MachineOperand &Op);
  void print(const MachineOperand &Op, const TargetRegisterInfo *TRI,
             unsigned I, bool ShouldPrintRegisterTies,
             const MachineRegisterInfo *MRI = nullptr, bool IsDef = false);
  void print(const MachineMemOperand &Op);

  void print(const MCCFIInstruction &CFI, const TargetRegisterInfo *TRI);
};

} // end namespace llvm

namespace llvm {
namespace yaml {

/// This struct serializes the LLVM IR module.
template <> struct BlockScalarTraits<Module> {
  static void output(const Module &Mod, void *Ctxt, raw_ostream &OS) {
    Mod.print(OS, nullptr);
  }
  static StringRef input(StringRef Str, void *Ctxt, Module &Mod) {
    llvm_unreachable("LLVM Module is supposed to be parsed separately");
    return "";
  }
};

} // end namespace yaml
} // end namespace llvm

static void printReg(unsigned Reg, raw_ostream &OS,
                     const TargetRegisterInfo *TRI) {
  // TODO: Print Stack Slots.
  if (!Reg)
    OS << '_';
  else if (TargetRegisterInfo::isVirtualRegister(Reg))
    OS << '%' << TargetRegisterInfo::virtReg2Index(Reg);
  else if (Reg < TRI->getNumRegs())
    OS << '%' << StringRef(TRI->getName(Reg)).lower();
  else
    llvm_unreachable("Can't print this kind of register yet");
}

static void printReg(unsigned Reg, yaml::StringValue &Dest,
                     const TargetRegisterInfo *TRI) {
  raw_string_ostream OS(Dest.Value);
  printReg(Reg, OS, TRI);
}

void MIRPrinter::print(const MachineFunction &MF) {
  initRegisterMaskIds(MF);

  yaml::MachineFunction YamlMF;
  YamlMF.Name = MF.getName();
  YamlMF.Alignment = MF.getAlignment();
  YamlMF.ExposesReturnsTwice = MF.exposesReturnsTwice();

  YamlMF.Legalized = MF.getProperties().hasProperty(
      MachineFunctionProperties::Property::Legalized);
  YamlMF.RegBankSelected = MF.getProperties().hasProperty(
      MachineFunctionProperties::Property::RegBankSelected);
  YamlMF.Selected = MF.getProperties().hasProperty(
      MachineFunctionProperties::Property::Selected);

  convert(YamlMF, MF.getRegInfo(), MF.getSubtarget().getRegisterInfo());
  ModuleSlotTracker MST(MF.getFunction()->getParent());
  MST.incorporateFunction(*MF.getFunction());
  convert(MST, YamlMF.FrameInfo, MF.getFrameInfo());
  convertStackObjects(YamlMF, MF.getFrameInfo(), MF.getMMI(), MST,
                      MF.getSubtarget().getRegisterInfo());
  if (const auto *ConstantPool = MF.getConstantPool())
    convert(YamlMF, *ConstantPool);
  if (const auto *JumpTableInfo = MF.getJumpTableInfo())
    convert(MST, YamlMF.JumpTableInfo, *JumpTableInfo);
  raw_string_ostream StrOS(YamlMF.Body.Value.Value);
  bool IsNewlineNeeded = false;
  for (const auto &MBB : MF) {
    if (IsNewlineNeeded)
      StrOS << "\n";
    MIPrinter(StrOS, MST, RegisterMaskIds, StackObjectOperandMapping)
        .print(MBB);
    IsNewlineNeeded = true;
  }
  StrOS.flush();
  yaml::Output Out(OS);
  Out << YamlMF;
}

void MIRPrinter::convert(yaml::MachineFunction &MF,
                         const MachineRegisterInfo &RegInfo,
                         const TargetRegisterInfo *TRI) {
  MF.TracksRegLiveness = RegInfo.tracksLiveness();

  // Print the virtual register definitions.
  for (unsigned I = 0, E = RegInfo.getNumVirtRegs(); I < E; ++I) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(I);
    yaml::VirtualRegisterDefinition VReg;
    VReg.ID = I;
    if (RegInfo.getRegClassOrNull(Reg))
      VReg.Class =
          StringRef(TRI->getRegClassName(RegInfo.getRegClass(Reg))).lower();
    else if (RegInfo.getRegBankOrNull(Reg))
      VReg.Class = StringRef(RegInfo.getRegBankOrNull(Reg)->getName()).lower();
    else {
      VReg.Class = std::string("_");
      assert(RegInfo.getSize(Reg) && "Generic registers must have a size");
    }
    unsigned PreferredReg = RegInfo.getSimpleHint(Reg);
    if (PreferredReg)
      printReg(PreferredReg, VReg.PreferredRegister, TRI);
    MF.VirtualRegisters.push_back(VReg);
  }

  // Print the live ins.
  for (auto I = RegInfo.livein_begin(), E = RegInfo.livein_end(); I != E; ++I) {
    yaml::MachineFunctionLiveIn LiveIn;
    printReg(I->first, LiveIn.Register, TRI);
    if (I->second)
      printReg(I->second, LiveIn.VirtualRegister, TRI);
    MF.LiveIns.push_back(LiveIn);
  }
  // The used physical register mask is printed as an inverted callee saved
  // register mask.
  const BitVector &UsedPhysRegMask = RegInfo.getUsedPhysRegsMask();
  if (UsedPhysRegMask.none())
    return;
  std::vector<yaml::FlowStringValue> CalleeSavedRegisters;
  for (unsigned I = 0, E = UsedPhysRegMask.size(); I != E; ++I) {
    if (!UsedPhysRegMask[I]) {
      yaml::FlowStringValue Reg;
      printReg(I, Reg, TRI);
      CalleeSavedRegisters.push_back(Reg);
    }
  }
  MF.CalleeSavedRegisters = CalleeSavedRegisters;
}

void MIRPrinter::convert(ModuleSlotTracker &MST,
                         yaml::MachineFrameInfo &YamlMFI,
                         const MachineFrameInfo &MFI) {
  YamlMFI.IsFrameAddressTaken = MFI.isFrameAddressTaken();
  YamlMFI.IsReturnAddressTaken = MFI.isReturnAddressTaken();
  YamlMFI.HasStackMap = MFI.hasStackMap();
  YamlMFI.HasPatchPoint = MFI.hasPatchPoint();
  YamlMFI.StackSize = MFI.getStackSize();
  YamlMFI.OffsetAdjustment = MFI.getOffsetAdjustment();
  YamlMFI.MaxAlignment = MFI.getMaxAlignment();
  YamlMFI.AdjustsStack = MFI.adjustsStack();
  YamlMFI.HasCalls = MFI.hasCalls();
  YamlMFI.MaxCallFrameSize = MFI.getMaxCallFrameSize();
  YamlMFI.HasOpaqueSPAdjustment = MFI.hasOpaqueSPAdjustment();
  YamlMFI.HasVAStart = MFI.hasVAStart();
  YamlMFI.HasMustTailInVarArgFunc = MFI.hasMustTailInVarArgFunc();
  if (MFI.getSavePoint()) {
    raw_string_ostream StrOS(YamlMFI.SavePoint.Value);
    MIPrinter(StrOS, MST, RegisterMaskIds, StackObjectOperandMapping)
        .printMBBReference(*MFI.getSavePoint());
  }
  if (MFI.getRestorePoint()) {
    raw_string_ostream StrOS(YamlMFI.RestorePoint.Value);
    MIPrinter(StrOS, MST, RegisterMaskIds, StackObjectOperandMapping)
        .printMBBReference(*MFI.getRestorePoint());
  }
}

void MIRPrinter::convertStackObjects(yaml::MachineFunction &MF,
                                     const MachineFrameInfo &MFI,
                                     MachineModuleInfo &MMI,
                                     ModuleSlotTracker &MST,
                                     const TargetRegisterInfo *TRI) {
  // Process fixed stack objects.
  unsigned ID = 0;
  for (int I = MFI.getObjectIndexBegin(); I < 0; ++I) {
    if (MFI.isDeadObjectIndex(I))
      continue;

    yaml::FixedMachineStackObject YamlObject;
    YamlObject.ID = ID;
    YamlObject.Type = MFI.isSpillSlotObjectIndex(I)
                          ? yaml::FixedMachineStackObject::SpillSlot
                          : yaml::FixedMachineStackObject::DefaultType;
    YamlObject.Offset = MFI.getObjectOffset(I);
    YamlObject.Size = MFI.getObjectSize(I);
    YamlObject.Alignment = MFI.getObjectAlignment(I);
    YamlObject.IsImmutable = MFI.isImmutableObjectIndex(I);
    YamlObject.IsAliased = MFI.isAliasedObjectIndex(I);
    MF.FixedStackObjects.push_back(YamlObject);
    StackObjectOperandMapping.insert(
        std::make_pair(I, FrameIndexOperand::createFixed(ID++)));
  }

  // Process ordinary stack objects.
  ID = 0;
  for (int I = 0, E = MFI.getObjectIndexEnd(); I < E; ++I) {
    if (MFI.isDeadObjectIndex(I))
      continue;

    yaml::MachineStackObject YamlObject;
    YamlObject.ID = ID;
    if (const auto *Alloca = MFI.getObjectAllocation(I))
      YamlObject.Name.Value =
          Alloca->hasName() ? Alloca->getName() : "<unnamed alloca>";
    YamlObject.Type = MFI.isSpillSlotObjectIndex(I)
                          ? yaml::MachineStackObject::SpillSlot
                          : MFI.isVariableSizedObjectIndex(I)
                                ? yaml::MachineStackObject::VariableSized
                                : yaml::MachineStackObject::DefaultType;
    YamlObject.Offset = MFI.getObjectOffset(I);
    YamlObject.Size = MFI.getObjectSize(I);
    YamlObject.Alignment = MFI.getObjectAlignment(I);

    MF.StackObjects.push_back(YamlObject);
    StackObjectOperandMapping.insert(std::make_pair(
        I, FrameIndexOperand::create(YamlObject.Name.Value, ID++)));
  }

  for (const auto &CSInfo : MFI.getCalleeSavedInfo()) {
    yaml::StringValue Reg;
    printReg(CSInfo.getReg(), Reg, TRI);
    auto StackObjectInfo = StackObjectOperandMapping.find(CSInfo.getFrameIdx());
    assert(StackObjectInfo != StackObjectOperandMapping.end() &&
           "Invalid stack object index");
    const FrameIndexOperand &StackObject = StackObjectInfo->second;
    if (StackObject.IsFixed)
      MF.FixedStackObjects[StackObject.ID].CalleeSavedRegister = Reg;
    else
      MF.StackObjects[StackObject.ID].CalleeSavedRegister = Reg;
  }
  for (unsigned I = 0, E = MFI.getLocalFrameObjectCount(); I < E; ++I) {
    auto LocalObject = MFI.getLocalFrameObjectMap(I);
    auto StackObjectInfo = StackObjectOperandMapping.find(LocalObject.first);
    assert(StackObjectInfo != StackObjectOperandMapping.end() &&
           "Invalid stack object index");
    const FrameIndexOperand &StackObject = StackObjectInfo->second;
    assert(!StackObject.IsFixed && "Expected a locally mapped stack object");
    MF.StackObjects[StackObject.ID].LocalOffset = LocalObject.second;
  }

  // Print the stack object references in the frame information class after
  // converting the stack objects.
  if (MFI.hasStackProtectorIndex()) {
    raw_string_ostream StrOS(MF.FrameInfo.StackProtector.Value);
    MIPrinter(StrOS, MST, RegisterMaskIds, StackObjectOperandMapping)
        .printStackObjectReference(MFI.getStackProtectorIndex());
  }

  // Print the debug variable information.
  for (MachineModuleInfo::VariableDbgInfo &DebugVar :
       MMI.getVariableDbgInfo()) {
    auto StackObjectInfo = StackObjectOperandMapping.find(DebugVar.Slot);
    assert(StackObjectInfo != StackObjectOperandMapping.end() &&
           "Invalid stack object index");
    const FrameIndexOperand &StackObject = StackObjectInfo->second;
    assert(!StackObject.IsFixed && "Expected a non-fixed stack object");
    auto &Object = MF.StackObjects[StackObject.ID];
    {
      raw_string_ostream StrOS(Object.DebugVar.Value);
      DebugVar.Var->printAsOperand(StrOS, MST);
    }
    {
      raw_string_ostream StrOS(Object.DebugExpr.Value);
      DebugVar.Expr->printAsOperand(StrOS, MST);
    }
    {
      raw_string_ostream StrOS(Object.DebugLoc.Value);
      DebugVar.Loc->printAsOperand(StrOS, MST);
    }
  }
}

void MIRPrinter::convert(yaml::MachineFunction &MF,
                         const MachineConstantPool &ConstantPool) {
  unsigned ID = 0;
  for (const MachineConstantPoolEntry &Constant : ConstantPool.getConstants()) {
    // TODO: Serialize target specific constant pool entries.
    if (Constant.isMachineConstantPoolEntry())
      llvm_unreachable("Can't print target specific constant pool entries yet");

    yaml::MachineConstantPoolValue YamlConstant;
    std::string Str;
    raw_string_ostream StrOS(Str);
    Constant.Val.ConstVal->printAsOperand(StrOS);
    YamlConstant.ID = ID++;
    YamlConstant.Value = StrOS.str();
    YamlConstant.Alignment = Constant.getAlignment();
    MF.Constants.push_back(YamlConstant);
  }
}

void MIRPrinter::convert(ModuleSlotTracker &MST,
                         yaml::MachineJumpTable &YamlJTI,
                         const MachineJumpTableInfo &JTI) {
  YamlJTI.Kind = JTI.getEntryKind();
  unsigned ID = 0;
  for (const auto &Table : JTI.getJumpTables()) {
    std::string Str;
    yaml::MachineJumpTable::Entry Entry;
    Entry.ID = ID++;
    for (const auto *MBB : Table.MBBs) {
      raw_string_ostream StrOS(Str);
      MIPrinter(StrOS, MST, RegisterMaskIds, StackObjectOperandMapping)
          .printMBBReference(*MBB);
      Entry.Blocks.push_back(StrOS.str());
      Str.clear();
    }
    YamlJTI.Entries.push_back(Entry);
  }
}

void MIRPrinter::initRegisterMaskIds(const MachineFunction &MF) {
  const auto *TRI = MF.getSubtarget().getRegisterInfo();
  unsigned I = 0;
  for (const uint32_t *Mask : TRI->getRegMasks())
    RegisterMaskIds.insert(std::make_pair(Mask, I++));
}

void MIPrinter::print(const MachineBasicBlock &MBB) {
  assert(MBB.getNumber() >= 0 && "Invalid MBB number");
  OS << "bb." << MBB.getNumber();
  bool HasAttributes = false;
  if (const auto *BB = MBB.getBasicBlock()) {
    if (BB->hasName()) {
      OS << "." << BB->getName();
    } else {
      HasAttributes = true;
      OS << " (";
      int Slot = MST.getLocalSlot(BB);
      if (Slot == -1)
        OS << "<ir-block badref>";
      else
        OS << (Twine("%ir-block.") + Twine(Slot)).str();
    }
  }
  if (MBB.hasAddressTaken()) {
    OS << (HasAttributes ? ", " : " (");
    OS << "address-taken";
    HasAttributes = true;
  }
  if (MBB.isEHPad()) {
    OS << (HasAttributes ? ", " : " (");
    OS << "landing-pad";
    HasAttributes = true;
  }
  if (MBB.getAlignment()) {
    OS << (HasAttributes ? ", " : " (");
    OS << "align " << MBB.getAlignment();
    HasAttributes = true;
  }
  if (HasAttributes)
    OS << ")";
  OS << ":\n";

  bool HasLineAttributes = false;
  // Print the successors
  if (!MBB.succ_empty()) {
    OS.indent(2) << "successors: ";
    for (auto I = MBB.succ_begin(), E = MBB.succ_end(); I != E; ++I) {
      if (I != MBB.succ_begin())
        OS << ", ";
      printMBBReference(**I);
      if (MBB.hasSuccessorProbabilities())
        OS << '(' << MBB.getSuccProbability(I) << ')';
    }
    OS << "\n";
    HasLineAttributes = true;
  }

  // Print the live in registers.
  const auto *TRI = MBB.getParent()->getSubtarget().getRegisterInfo();
  assert(TRI && "Expected target register info");
  if (!MBB.livein_empty()) {
    OS.indent(2) << "liveins: ";
    bool First = true;
    for (const auto &LI : MBB.liveins()) {
      if (!First)
        OS << ", ";
      First = false;
      printReg(LI.PhysReg, OS, TRI);
      if (LI.LaneMask != ~0u)
        OS << ':' << PrintLaneMask(LI.LaneMask);
    }
    OS << "\n";
    HasLineAttributes = true;
  }

  if (HasLineAttributes)
    OS << "\n";
  bool IsInBundle = false;
  for (auto I = MBB.instr_begin(), E = MBB.instr_end(); I != E; ++I) {
    const MachineInstr &MI = *I;
    if (IsInBundle && !MI.isInsideBundle()) {
      OS.indent(2) << "}\n";
      IsInBundle = false;
    }
    OS.indent(IsInBundle ? 4 : 2);
    print(MI);
    if (!IsInBundle && MI.getFlag(MachineInstr::BundledSucc)) {
      OS << " {";
      IsInBundle = true;
    }
    OS << "\n";
  }
  if (IsInBundle)
    OS.indent(2) << "}\n";
}

/// Return true when an instruction has tied register that can't be determined
/// by the instruction's descriptor.
static bool hasComplexRegisterTies(const MachineInstr &MI) {
  const MCInstrDesc &MCID = MI.getDesc();
  for (unsigned I = 0, E = MI.getNumOperands(); I < E; ++I) {
    const auto &Operand = MI.getOperand(I);
    if (!Operand.isReg() || Operand.isDef())
      // Ignore the defined registers as MCID marks only the uses as tied.
      continue;
    int ExpectedTiedIdx = MCID.getOperandConstraint(I, MCOI::TIED_TO);
    int TiedIdx = Operand.isTied() ? int(MI.findTiedOperandIdx(I)) : -1;
    if (ExpectedTiedIdx != TiedIdx)
      return true;
  }
  return false;
}

void MIPrinter::print(const MachineInstr &MI) {
  const auto *MF = MI.getParent()->getParent();
  const auto &MRI = MF->getRegInfo();
  const auto &SubTarget = MF->getSubtarget();
  const auto *TRI = SubTarget.getRegisterInfo();
  assert(TRI && "Expected target register info");
  const auto *TII = SubTarget.getInstrInfo();
  assert(TII && "Expected target instruction info");
  if (MI.isCFIInstruction())
    assert(MI.getNumOperands() == 1 && "Expected 1 operand in CFI instruction");

  bool ShouldPrintRegisterTies = hasComplexRegisterTies(MI);
  unsigned I = 0, E = MI.getNumOperands();
  for (; I < E && MI.getOperand(I).isReg() && MI.getOperand(I).isDef() &&
         !MI.getOperand(I).isImplicit();
       ++I) {
    if (I)
      OS << ", ";
    print(MI.getOperand(I), TRI, I, ShouldPrintRegisterTies, &MRI,
          /*IsDef=*/true);
  }

  if (I)
    OS << " = ";
  if (MI.getFlag(MachineInstr::FrameSetup))
    OS << "frame-setup ";
  OS << TII->getName(MI.getOpcode());
  if (isPreISelGenericOpcode(MI.getOpcode())) {
    assert(MI.getType().isValid() && "Generic instructions must have a type");
    unsigned NumTypes = MI.getNumTypes();
    OS << (NumTypes > 1 ? " {" : "") << ' ';
    for (unsigned i = 0; i < NumTypes; ++i) {
      MI.getType(i).print(OS);
      if (i + 1 != NumTypes)
        OS <<  ", ";
    }
    OS << (NumTypes > 1 ? " }" : "") << ' ';
  }
  if (I < E)
    OS << ' ';

  bool NeedComma = false;
  for (; I < E; ++I) {
    if (NeedComma)
      OS << ", ";
    print(MI.getOperand(I), TRI, I, ShouldPrintRegisterTies);
    NeedComma = true;
  }

  if (MI.getDebugLoc()) {
    if (NeedComma)
      OS << ',';
    OS << " debug-location ";
    MI.getDebugLoc()->printAsOperand(OS, MST);
  }

  if (!MI.memoperands_empty()) {
    OS << " :: ";
    bool NeedComma = false;
    for (const auto *Op : MI.memoperands()) {
      if (NeedComma)
        OS << ", ";
      print(*Op);
      NeedComma = true;
    }
  }
}

void MIPrinter::printMBBReference(const MachineBasicBlock &MBB) {
  OS << "%bb." << MBB.getNumber();
  if (const auto *BB = MBB.getBasicBlock()) {
    if (BB->hasName())
      OS << '.' << BB->getName();
  }
}

static void printIRSlotNumber(raw_ostream &OS, int Slot) {
  if (Slot == -1)
    OS << "<badref>";
  else
    OS << Slot;
}

void MIPrinter::printIRBlockReference(const BasicBlock &BB) {
  OS << "%ir-block.";
  if (BB.hasName()) {
    printLLVMNameWithoutPrefix(OS, BB.getName());
    return;
  }
  const Function *F = BB.getParent();
  int Slot;
  if (F == MST.getCurrentFunction()) {
    Slot = MST.getLocalSlot(&BB);
  } else {
    ModuleSlotTracker CustomMST(F->getParent(),
                                /*ShouldInitializeAllMetadata=*/false);
    CustomMST.incorporateFunction(*F);
    Slot = CustomMST.getLocalSlot(&BB);
  }
  printIRSlotNumber(OS, Slot);
}

void MIPrinter::printIRValueReference(const Value &V) {
  if (isa<GlobalValue>(V)) {
    V.printAsOperand(OS, /*PrintType=*/false, MST);
    return;
  }
  if (isa<Constant>(V)) {
    // Machine memory operands can load/store to/from constant value pointers.
    OS << '`';
    V.printAsOperand(OS, /*PrintType=*/true, MST);
    OS << '`';
    return;
  }
  OS << "%ir.";
  if (V.hasName()) {
    printLLVMNameWithoutPrefix(OS, V.getName());
    return;
  }
  printIRSlotNumber(OS, MST.getLocalSlot(&V));
}

void MIPrinter::printStackObjectReference(int FrameIndex) {
  auto ObjectInfo = StackObjectOperandMapping.find(FrameIndex);
  assert(ObjectInfo != StackObjectOperandMapping.end() &&
         "Invalid frame index");
  const FrameIndexOperand &Operand = ObjectInfo->second;
  if (Operand.IsFixed) {
    OS << "%fixed-stack." << Operand.ID;
    return;
  }
  OS << "%stack." << Operand.ID;
  if (!Operand.Name.empty())
    OS << '.' << Operand.Name;
}

void MIPrinter::printOffset(int64_t Offset) {
  if (Offset == 0)
    return;
  if (Offset < 0) {
    OS << " - " << -Offset;
    return;
  }
  OS << " + " << Offset;
}

static const char *getTargetFlagName(const TargetInstrInfo *TII, unsigned TF) {
  auto Flags = TII->getSerializableDirectMachineOperandTargetFlags();
  for (const auto &I : Flags) {
    if (I.first == TF) {
      return I.second;
    }
  }
  return nullptr;
}

void MIPrinter::printTargetFlags(const MachineOperand &Op) {
  if (!Op.getTargetFlags())
    return;
  const auto *TII =
      Op.getParent()->getParent()->getParent()->getSubtarget().getInstrInfo();
  assert(TII && "expected instruction info");
  auto Flags = TII->decomposeMachineOperandsTargetFlags(Op.getTargetFlags());
  OS << "target-flags(";
  const bool HasDirectFlags = Flags.first;
  const bool HasBitmaskFlags = Flags.second;
  if (!HasDirectFlags && !HasBitmaskFlags) {
    OS << "<unknown>) ";
    return;
  }
  if (HasDirectFlags) {
    if (const auto *Name = getTargetFlagName(TII, Flags.first))
      OS << Name;
    else
      OS << "<unknown target flag>";
  }
  if (!HasBitmaskFlags) {
    OS << ") ";
    return;
  }
  bool IsCommaNeeded = HasDirectFlags;
  unsigned BitMask = Flags.second;
  auto BitMasks = TII->getSerializableBitmaskMachineOperandTargetFlags();
  for (const auto &Mask : BitMasks) {
    // Check if the flag's bitmask has the bits of the current mask set.
    if ((BitMask & Mask.first) == Mask.first) {
      if (IsCommaNeeded)
        OS << ", ";
      IsCommaNeeded = true;
      OS << Mask.second;
      // Clear the bits which were serialized from the flag's bitmask.
      BitMask &= ~(Mask.first);
    }
  }
  if (BitMask) {
    // When the resulting flag's bitmask isn't zero, we know that we didn't
    // serialize all of the bit flags.
    if (IsCommaNeeded)
      OS << ", ";
    OS << "<unknown bitmask target flag>";
  }
  OS << ") ";
}

static const char *getTargetIndexName(const MachineFunction &MF, int Index) {
  const auto *TII = MF.getSubtarget().getInstrInfo();
  assert(TII && "expected instruction info");
  auto Indices = TII->getSerializableTargetIndices();
  for (const auto &I : Indices) {
    if (I.first == Index) {
      return I.second;
    }
  }
  return nullptr;
}

void MIPrinter::print(const MachineOperand &Op, const TargetRegisterInfo *TRI,
                      unsigned I, bool ShouldPrintRegisterTies,
                      const MachineRegisterInfo *MRI, bool IsDef) {
  printTargetFlags(Op);
  switch (Op.getType()) {
  case MachineOperand::MO_Register:
    if (Op.isImplicit())
      OS << (Op.isDef() ? "implicit-def " : "implicit ");
    else if (!IsDef && Op.isDef())
      // Print the 'def' flag only when the operand is defined after '='.
      OS << "def ";
    if (Op.isInternalRead())
      OS << "internal ";
    if (Op.isDead())
      OS << "dead ";
    if (Op.isKill())
      OS << "killed ";
    if (Op.isUndef())
      OS << "undef ";
    if (Op.isEarlyClobber())
      OS << "early-clobber ";
    if (Op.isDebug())
      OS << "debug-use ";
    printReg(Op.getReg(), OS, TRI);
    // Print the sub register.
    if (Op.getSubReg() != 0)
      OS << '.' << TRI->getSubRegIndexName(Op.getSubReg());
    if (ShouldPrintRegisterTies && Op.isTied() && !Op.isDef())
      OS << "(tied-def " << Op.getParent()->findTiedOperandIdx(I) << ")";
    assert((!IsDef || MRI) && "for IsDef, MRI must be provided");
    if (IsDef && MRI->getSize(Op.getReg()))
      OS << '(' << MRI->getSize(Op.getReg()) << ')';
    break;
  case MachineOperand::MO_Immediate:
    OS << Op.getImm();
    break;
  case MachineOperand::MO_CImmediate:
    Op.getCImm()->printAsOperand(OS, /*PrintType=*/true, MST);
    break;
  case MachineOperand::MO_FPImmediate:
    Op.getFPImm()->printAsOperand(OS, /*PrintType=*/true, MST);
    break;
  case MachineOperand::MO_MachineBasicBlock:
    printMBBReference(*Op.getMBB());
    break;
  case MachineOperand::MO_FrameIndex:
    printStackObjectReference(Op.getIndex());
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    OS << "%const." << Op.getIndex();
    printOffset(Op.getOffset());
    break;
  case MachineOperand::MO_TargetIndex: {
    OS << "target-index(";
    if (const auto *Name = getTargetIndexName(
            *Op.getParent()->getParent()->getParent(), Op.getIndex()))
      OS << Name;
    else
      OS << "<unknown>";
    OS << ')';
    printOffset(Op.getOffset());
    break;
  }
  case MachineOperand::MO_JumpTableIndex:
    OS << "%jump-table." << Op.getIndex();
    break;
  case MachineOperand::MO_ExternalSymbol:
    OS << '$';
    printLLVMNameWithoutPrefix(OS, Op.getSymbolName());
    printOffset(Op.getOffset());
    break;
  case MachineOperand::MO_GlobalAddress:
    Op.getGlobal()->printAsOperand(OS, /*PrintType=*/false, MST);
    printOffset(Op.getOffset());
    break;
  case MachineOperand::MO_BlockAddress:
    OS << "blockaddress(";
    Op.getBlockAddress()->getFunction()->printAsOperand(OS, /*PrintType=*/false,
                                                        MST);
    OS << ", ";
    printIRBlockReference(*Op.getBlockAddress()->getBasicBlock());
    OS << ')';
    printOffset(Op.getOffset());
    break;
  case MachineOperand::MO_RegisterMask: {
    auto RegMaskInfo = RegisterMaskIds.find(Op.getRegMask());
    if (RegMaskInfo != RegisterMaskIds.end())
      OS << StringRef(TRI->getRegMaskNames()[RegMaskInfo->second]).lower();
    else
      llvm_unreachable("Can't print this machine register mask yet.");
    break;
  }
  case MachineOperand::MO_RegisterLiveOut: {
    const uint32_t *RegMask = Op.getRegLiveOut();
    OS << "liveout(";
    bool IsCommaNeeded = false;
    for (unsigned Reg = 0, E = TRI->getNumRegs(); Reg < E; ++Reg) {
      if (RegMask[Reg / 32] & (1U << (Reg % 32))) {
        if (IsCommaNeeded)
          OS << ", ";
        printReg(Reg, OS, TRI);
        IsCommaNeeded = true;
      }
    }
    OS << ")";
    break;
  }
  case MachineOperand::MO_Metadata:
    Op.getMetadata()->printAsOperand(OS, MST);
    break;
  case MachineOperand::MO_MCSymbol:
    OS << "<mcsymbol " << *Op.getMCSymbol() << ">";
    break;
  case MachineOperand::MO_CFIIndex: {
    const auto &MMI = Op.getParent()->getParent()->getParent()->getMMI();
    print(MMI.getFrameInstructions()[Op.getCFIIndex()], TRI);
    break;
  }
  case MachineOperand::MO_IntrinsicID: {
    Intrinsic::ID ID = Op.getIntrinsicID();
    if (ID < Intrinsic::num_intrinsics)
      OS << "intrinsic(@" << Intrinsic::getName(ID, None) << ')';
    else {
      const MachineFunction &MF = *Op.getParent()->getParent()->getParent();
      const TargetIntrinsicInfo *TII = MF.getTarget().getIntrinsicInfo();
      OS << "intrinsic(@" << TII->getName(ID) << ')';
    }
    break;
  }
  case MachineOperand::MO_Predicate: {
    auto Pred = static_cast<CmpInst::Predicate>(Op.getPredicate());
    OS << (CmpInst::isIntPredicate(Pred) ? "int" : "float") << "pred("
       << CmpInst::getPredicateName(Pred) << ')';
    break;
  }
  }
}

void MIPrinter::print(const MachineMemOperand &Op) {
  OS << '(';
  // TODO: Print operand's target specific flags.
  if (Op.isVolatile())
    OS << "volatile ";
  if (Op.isNonTemporal())
    OS << "non-temporal ";
  if (Op.isInvariant())
    OS << "invariant ";
  if (Op.isLoad())
    OS << "load ";
  else {
    assert(Op.isStore() && "Non load machine operand must be a store");
    OS << "store ";
  }
  OS << Op.getSize();
  if (const Value *Val = Op.getValue()) {
    OS << (Op.isLoad() ? " from " : " into ");
    printIRValueReference(*Val);
  } else if (const PseudoSourceValue *PVal = Op.getPseudoValue()) {
    OS << (Op.isLoad() ? " from " : " into ");
    assert(PVal && "Expected a pseudo source value");
    switch (PVal->kind()) {
    case PseudoSourceValue::Stack:
      OS << "stack";
      break;
    case PseudoSourceValue::GOT:
      OS << "got";
      break;
    case PseudoSourceValue::JumpTable:
      OS << "jump-table";
      break;
    case PseudoSourceValue::ConstantPool:
      OS << "constant-pool";
      break;
    case PseudoSourceValue::FixedStack:
      printStackObjectReference(
          cast<FixedStackPseudoSourceValue>(PVal)->getFrameIndex());
      break;
    case PseudoSourceValue::GlobalValueCallEntry:
      OS << "call-entry ";
      cast<GlobalValuePseudoSourceValue>(PVal)->getValue()->printAsOperand(
          OS, /*PrintType=*/false, MST);
      break;
    case PseudoSourceValue::ExternalSymbolCallEntry:
      OS << "call-entry $";
      printLLVMNameWithoutPrefix(
          OS, cast<ExternalSymbolPseudoSourceValue>(PVal)->getSymbol());
      break;
    }
  }
  printOffset(Op.getOffset());
  if (Op.getBaseAlignment() != Op.getSize())
    OS << ", align " << Op.getBaseAlignment();
  auto AAInfo = Op.getAAInfo();
  if (AAInfo.TBAA) {
    OS << ", !tbaa ";
    AAInfo.TBAA->printAsOperand(OS, MST);
  }
  if (AAInfo.Scope) {
    OS << ", !alias.scope ";
    AAInfo.Scope->printAsOperand(OS, MST);
  }
  if (AAInfo.NoAlias) {
    OS << ", !noalias ";
    AAInfo.NoAlias->printAsOperand(OS, MST);
  }
  if (Op.getRanges()) {
    OS << ", !range ";
    Op.getRanges()->printAsOperand(OS, MST);
  }
  OS << ')';
}

static void printCFIRegister(unsigned DwarfReg, raw_ostream &OS,
                             const TargetRegisterInfo *TRI) {
  int Reg = TRI->getLLVMRegNum(DwarfReg, true);
  if (Reg == -1) {
    OS << "<badreg>";
    return;
  }
  printReg(Reg, OS, TRI);
}

void MIPrinter::print(const MCCFIInstruction &CFI,
                      const TargetRegisterInfo *TRI) {
  switch (CFI.getOperation()) {
  case MCCFIInstruction::OpSameValue:
    OS << "same_value ";
    if (CFI.getLabel())
      OS << "<mcsymbol> ";
    printCFIRegister(CFI.getRegister(), OS, TRI);
    break;
  case MCCFIInstruction::OpOffset:
    OS << "offset ";
    if (CFI.getLabel())
      OS << "<mcsymbol> ";
    printCFIRegister(CFI.getRegister(), OS, TRI);
    OS << ", " << CFI.getOffset();
    break;
  case MCCFIInstruction::OpDefCfaRegister:
    OS << "def_cfa_register ";
    if (CFI.getLabel())
      OS << "<mcsymbol> ";
    printCFIRegister(CFI.getRegister(), OS, TRI);
    break;
  case MCCFIInstruction::OpDefCfaOffset:
    OS << "def_cfa_offset ";
    if (CFI.getLabel())
      OS << "<mcsymbol> ";
    OS << CFI.getOffset();
    break;
  case MCCFIInstruction::OpDefCfa:
    OS << "def_cfa ";
    if (CFI.getLabel())
      OS << "<mcsymbol> ";
    printCFIRegister(CFI.getRegister(), OS, TRI);
    OS << ", " << CFI.getOffset();
    break;
  default:
    // TODO: Print the other CFI Operations.
    OS << "<unserializable cfi operation>";
    break;
  }
}

void llvm::printMIR(raw_ostream &OS, const Module &M) {
  yaml::Output Out(OS);
  Out << const_cast<Module &>(M);
}

void llvm::printMIR(raw_ostream &OS, const MachineFunction &MF) {
  MIRPrinter Printer(OS);
  Printer.print(MF);
}

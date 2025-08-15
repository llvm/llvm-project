//===- MIRPrinter.cpp - MIR serialization format printer ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the class that prints out the LLVM IR and machine
// functions using the MIR serialization format.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MIRPrinter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MIRYamlMapping.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleSlotTracker.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/Value.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

static cl::opt<bool> SimplifyMIR(
    "simplify-mir", cl::Hidden,
    cl::desc("Leave out unnecessary information when printing MIR"));

static cl::opt<bool> PrintLocations("mir-debug-loc", cl::Hidden, cl::init(true),
                                    cl::desc("Print MIR debug-locations"));

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

struct MFPrintState {
  MachineModuleSlotTracker MST;
  DenseMap<const uint32_t *, unsigned> RegisterMaskIds;
  /// Maps from stack object indices to operand indices which will be used when
  /// printing frame index machine operands.
  DenseMap<int, FrameIndexOperand> StackObjectOperandMapping;
  /// Synchronization scope names registered with LLVMContext.
  SmallVector<StringRef, 8> SSNs;

  MFPrintState(const MachineModuleInfo &MMI, const MachineFunction &MF)
      : MST(MMI, &MF) {}
};

} // end anonymous namespace

namespace llvm::yaml {

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

} // end namespace llvm::yaml

static void printRegMIR(Register Reg, yaml::StringValue &Dest,
                        const TargetRegisterInfo *TRI) {
  raw_string_ostream OS(Dest.Value);
  OS << printReg(Reg, TRI);
}

static DenseMap<const uint32_t *, unsigned>
initRegisterMaskIds(const MachineFunction &MF) {
  DenseMap<const uint32_t *, unsigned> RegisterMaskIds;
  const auto *TRI = MF.getSubtarget().getRegisterInfo();
  unsigned I = 0;
  for (const uint32_t *Mask : TRI->getRegMasks())
    RegisterMaskIds.insert(std::make_pair(Mask, I++));
  return RegisterMaskIds;
}

static void printMBB(raw_ostream &OS, MFPrintState &State,
                     const MachineBasicBlock &MBB);
static void convertMRI(yaml::MachineFunction &YamlMF, const MachineFunction &MF,
                       const MachineRegisterInfo &RegInfo,
                       const TargetRegisterInfo *TRI);
static void convertMCP(yaml::MachineFunction &MF,
                       const MachineConstantPool &ConstantPool);
static void convertMJTI(ModuleSlotTracker &MST, yaml::MachineJumpTable &YamlJTI,
                        const MachineJumpTableInfo &JTI);
static void convertMFI(ModuleSlotTracker &MST, yaml::MachineFrameInfo &YamlMFI,
                       const MachineFrameInfo &MFI);
static void
convertSRPoints(ModuleSlotTracker &MST,
                std::vector<yaml::SaveRestorePointEntry> &YamlSRPoints,
                ArrayRef<MachineBasicBlock *> SaveRestorePoints);
static void convertStackObjects(yaml::MachineFunction &YMF,
                                const MachineFunction &MF,
                                ModuleSlotTracker &MST, MFPrintState &State);
static void convertEntryValueObjects(yaml::MachineFunction &YMF,
                                     const MachineFunction &MF,
                                     ModuleSlotTracker &MST);
static void convertCallSiteObjects(yaml::MachineFunction &YMF,
                                   const MachineFunction &MF,
                                   ModuleSlotTracker &MST);
static void convertMachineMetadataNodes(yaml::MachineFunction &YMF,
                                        const MachineFunction &MF,
                                        MachineModuleSlotTracker &MST);
static void convertCalledGlobals(yaml::MachineFunction &YMF,
                                 const MachineFunction &MF,
                                 MachineModuleSlotTracker &MST);

static void printMF(raw_ostream &OS, const MachineModuleInfo &MMI,
                    const MachineFunction &MF) {
  MFPrintState State(MMI, MF);
  State.RegisterMaskIds = initRegisterMaskIds(MF);

  yaml::MachineFunction YamlMF;
  YamlMF.Name = MF.getName();
  YamlMF.Alignment = MF.getAlignment();
  YamlMF.ExposesReturnsTwice = MF.exposesReturnsTwice();
  YamlMF.HasWinCFI = MF.hasWinCFI();

  YamlMF.CallsEHReturn = MF.callsEHReturn();
  YamlMF.CallsUnwindInit = MF.callsUnwindInit();
  YamlMF.HasEHContTarget = MF.hasEHContTarget();
  YamlMF.HasEHScopes = MF.hasEHScopes();
  YamlMF.HasEHFunclets = MF.hasEHFunclets();
  YamlMF.HasFakeUses = MF.hasFakeUses();
  YamlMF.IsOutlined = MF.isOutlined();
  YamlMF.UseDebugInstrRef = MF.useDebugInstrRef();

  const MachineFunctionProperties &Props = MF.getProperties();
  YamlMF.Legalized = Props.hasLegalized();
  YamlMF.RegBankSelected = Props.hasRegBankSelected();
  YamlMF.Selected = Props.hasSelected();
  YamlMF.FailedISel = Props.hasFailedISel();
  YamlMF.FailsVerification = Props.hasFailsVerification();
  YamlMF.TracksDebugUserValues = Props.hasTracksDebugUserValues();
  YamlMF.NoPHIs = Props.hasNoPHIs();
  YamlMF.IsSSA = Props.hasIsSSA();
  YamlMF.NoVRegs = Props.hasNoVRegs();

  convertMRI(YamlMF, MF, MF.getRegInfo(), MF.getSubtarget().getRegisterInfo());
  MachineModuleSlotTracker &MST = State.MST;
  MST.incorporateFunction(MF.getFunction());
  convertMFI(MST, YamlMF.FrameInfo, MF.getFrameInfo());
  convertStackObjects(YamlMF, MF, MST, State);
  convertEntryValueObjects(YamlMF, MF, MST);
  convertCallSiteObjects(YamlMF, MF, MST);
  for (const auto &Sub : MF.DebugValueSubstitutions) {
    const auto &SubSrc = Sub.Src;
    const auto &SubDest = Sub.Dest;
    YamlMF.DebugValueSubstitutions.push_back({SubSrc.first, SubSrc.second,
                                              SubDest.first,
                                              SubDest.second,
                                              Sub.Subreg});
  }
  if (const auto *ConstantPool = MF.getConstantPool())
    convertMCP(YamlMF, *ConstantPool);
  if (const auto *JumpTableInfo = MF.getJumpTableInfo())
    convertMJTI(MST, YamlMF.JumpTableInfo, *JumpTableInfo);

  const TargetMachine &TM = MF.getTarget();
  YamlMF.MachineFuncInfo =
      std::unique_ptr<yaml::MachineFunctionInfo>(TM.convertFuncInfoToYAML(MF));

  raw_string_ostream StrOS(YamlMF.Body.Value.Value);
  bool IsNewlineNeeded = false;
  for (const auto &MBB : MF) {
    if (IsNewlineNeeded)
      StrOS << "\n";
    printMBB(StrOS, State, MBB);
    IsNewlineNeeded = true;
  }
  // Convert machine metadata collected during the print of the machine
  // function.
  convertMachineMetadataNodes(YamlMF, MF, MST);

  convertCalledGlobals(YamlMF, MF, MST);

  yaml::Output Out(OS);
  if (!SimplifyMIR)
      Out.setWriteDefaultValues(true);
  Out << YamlMF;
}

static void printCustomRegMask(const uint32_t *RegMask, raw_ostream &OS,
                               const TargetRegisterInfo *TRI) {
  assert(RegMask && "Can't print an empty register mask");
  OS << StringRef("CustomRegMask(");

  bool IsRegInRegMaskFound = false;
  for (int I = 0, E = TRI->getNumRegs(); I < E; I++) {
    // Check whether the register is asserted in regmask.
    if (RegMask[I / 32] & (1u << (I % 32))) {
      if (IsRegInRegMaskFound)
        OS << ',';
      OS << printReg(I, TRI);
      IsRegInRegMaskFound = true;
    }
  }

  OS << ')';
}

static void printRegClassOrBank(Register Reg, yaml::StringValue &Dest,
                                const MachineRegisterInfo &RegInfo,
                                const TargetRegisterInfo *TRI) {
  raw_string_ostream OS(Dest.Value);
  OS << printRegClassOrBank(Reg, RegInfo, TRI);
}

template <typename T>
static void
printStackObjectDbgInfo(const MachineFunction::VariableDbgInfo &DebugVar,
                        T &Object, ModuleSlotTracker &MST) {
  std::array<std::string *, 3> Outputs{{&Object.DebugVar.Value,
                                        &Object.DebugExpr.Value,
                                        &Object.DebugLoc.Value}};
  std::array<const Metadata *, 3> Metas{{DebugVar.Var,
                                        DebugVar.Expr,
                                        DebugVar.Loc}};
  for (unsigned i = 0; i < 3; ++i) {
    raw_string_ostream StrOS(*Outputs[i]);
    Metas[i]->printAsOperand(StrOS, MST);
  }
}

static void printRegFlags(Register Reg,
                          std::vector<yaml::FlowStringValue> &RegisterFlags,
                          const MachineFunction &MF,
                          const TargetRegisterInfo *TRI) {
  auto FlagValues = TRI->getVRegFlagsOfReg(Reg, MF);
  for (auto &Flag : FlagValues)
    RegisterFlags.push_back(yaml::FlowStringValue(Flag.str()));
}

static void convertMRI(yaml::MachineFunction &YamlMF, const MachineFunction &MF,
                       const MachineRegisterInfo &RegInfo,
                       const TargetRegisterInfo *TRI) {
  YamlMF.TracksRegLiveness = RegInfo.tracksLiveness();

  // Print the virtual register definitions.
  for (unsigned I = 0, E = RegInfo.getNumVirtRegs(); I < E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    yaml::VirtualRegisterDefinition VReg;
    VReg.ID = I;
    if (RegInfo.getVRegName(Reg) != "")
      continue;
    ::printRegClassOrBank(Reg, VReg.Class, RegInfo, TRI);
    Register PreferredReg = RegInfo.getSimpleHint(Reg);
    if (PreferredReg)
      printRegMIR(PreferredReg, VReg.PreferredRegister, TRI);
    printRegFlags(Reg, VReg.RegisterFlags, MF, TRI);
    YamlMF.VirtualRegisters.push_back(std::move(VReg));
  }

  // Print the live ins.
  for (std::pair<MCRegister, Register> LI : RegInfo.liveins()) {
    yaml::MachineFunctionLiveIn LiveIn;
    printRegMIR(LI.first, LiveIn.Register, TRI);
    if (LI.second)
      printRegMIR(LI.second, LiveIn.VirtualRegister, TRI);
    YamlMF.LiveIns.push_back(std::move(LiveIn));
  }

  // Prints the callee saved registers.
  if (RegInfo.isUpdatedCSRsInitialized()) {
    const MCPhysReg *CalleeSavedRegs = RegInfo.getCalleeSavedRegs();
    std::vector<yaml::FlowStringValue> CalleeSavedRegisters;
    for (const MCPhysReg *I = CalleeSavedRegs; *I; ++I) {
      yaml::FlowStringValue Reg;
      printRegMIR(*I, Reg, TRI);
      CalleeSavedRegisters.push_back(std::move(Reg));
    }
    YamlMF.CalleeSavedRegisters = std::move(CalleeSavedRegisters);
  }
}

static void convertMFI(ModuleSlotTracker &MST, yaml::MachineFrameInfo &YamlMFI,
                       const MachineFrameInfo &MFI) {
  YamlMFI.IsFrameAddressTaken = MFI.isFrameAddressTaken();
  YamlMFI.IsReturnAddressTaken = MFI.isReturnAddressTaken();
  YamlMFI.HasStackMap = MFI.hasStackMap();
  YamlMFI.HasPatchPoint = MFI.hasPatchPoint();
  YamlMFI.StackSize = MFI.getStackSize();
  YamlMFI.OffsetAdjustment = MFI.getOffsetAdjustment();
  YamlMFI.MaxAlignment = MFI.getMaxAlign().value();
  YamlMFI.AdjustsStack = MFI.adjustsStack();
  YamlMFI.HasCalls = MFI.hasCalls();
  YamlMFI.MaxCallFrameSize = MFI.isMaxCallFrameSizeComputed()
    ? MFI.getMaxCallFrameSize() : ~0u;
  YamlMFI.CVBytesOfCalleeSavedRegisters =
      MFI.getCVBytesOfCalleeSavedRegisters();
  YamlMFI.HasOpaqueSPAdjustment = MFI.hasOpaqueSPAdjustment();
  YamlMFI.HasVAStart = MFI.hasVAStart();
  YamlMFI.HasMustTailInVarArgFunc = MFI.hasMustTailInVarArgFunc();
  YamlMFI.HasTailCall = MFI.hasTailCall();
  YamlMFI.IsCalleeSavedInfoValid = MFI.isCalleeSavedInfoValid();
  YamlMFI.LocalFrameSize = MFI.getLocalFrameSize();
  if (!MFI.getSavePoints().empty())
    convertSRPoints(MST, YamlMFI.SavePoints, MFI.getSavePoints());
  if (!MFI.getRestorePoints().empty())
    convertSRPoints(MST, YamlMFI.RestorePoints, MFI.getRestorePoints());
}

static void convertEntryValueObjects(yaml::MachineFunction &YMF,
                                     const MachineFunction &MF,
                                     ModuleSlotTracker &MST) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  for (const MachineFunction::VariableDbgInfo &DebugVar :
       MF.getEntryValueVariableDbgInfo()) {
    yaml::EntryValueObject &Obj = YMF.EntryValueObjects.emplace_back();
    printStackObjectDbgInfo(DebugVar, Obj, MST);
    MCRegister EntryValReg = DebugVar.getEntryValueRegister();
    printRegMIR(EntryValReg, Obj.EntryValueRegister, TRI);
  }
}

static void printStackObjectReference(raw_ostream &OS,
                                      const MFPrintState &State,
                                      int FrameIndex) {
  auto ObjectInfo = State.StackObjectOperandMapping.find(FrameIndex);
  assert(ObjectInfo != State.StackObjectOperandMapping.end() &&
         "Invalid frame index");
  const FrameIndexOperand &Operand = ObjectInfo->second;
  MachineOperand::printStackObjectReference(OS, Operand.ID, Operand.IsFixed,
                                            Operand.Name);
}

static void convertStackObjects(yaml::MachineFunction &YMF,
                                const MachineFunction &MF,
                                ModuleSlotTracker &MST, MFPrintState &State) {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  // Process fixed stack objects.
  assert(YMF.FixedStackObjects.empty());
  SmallVector<int, 32> FixedStackObjectsIdx;
  const int BeginIdx = MFI.getObjectIndexBegin();
  if (BeginIdx < 0)
    FixedStackObjectsIdx.reserve(-BeginIdx);

  unsigned ID = 0;
  for (int I = BeginIdx; I < 0; ++I, ++ID) {
    FixedStackObjectsIdx.push_back(-1); // Fill index for possible dead.
    if (MFI.isDeadObjectIndex(I))
      continue;

    yaml::FixedMachineStackObject YamlObject;
    YamlObject.ID = ID;
    YamlObject.Type = MFI.isSpillSlotObjectIndex(I)
                          ? yaml::FixedMachineStackObject::SpillSlot
                          : yaml::FixedMachineStackObject::DefaultType;
    YamlObject.Offset = MFI.getObjectOffset(I);
    YamlObject.Size = MFI.getObjectSize(I);
    YamlObject.Alignment = MFI.getObjectAlign(I);
    YamlObject.StackID = (TargetStackID::Value)MFI.getStackID(I);
    YamlObject.IsImmutable = MFI.isImmutableObjectIndex(I);
    YamlObject.IsAliased = MFI.isAliasedObjectIndex(I);
    // Save the ID' position in FixedStackObjects storage vector.
    FixedStackObjectsIdx[ID] = YMF.FixedStackObjects.size();
    YMF.FixedStackObjects.push_back(std::move(YamlObject));
    State.StackObjectOperandMapping.insert(
        std::make_pair(I, FrameIndexOperand::createFixed(ID)));
  }

  // Process ordinary stack objects.
  assert(YMF.StackObjects.empty());
  SmallVector<unsigned, 32> StackObjectsIdx;
  const int EndIdx = MFI.getObjectIndexEnd();
  if (EndIdx > 0)
    StackObjectsIdx.reserve(EndIdx);
  ID = 0;
  for (int I = 0; I < EndIdx; ++I, ++ID) {
    StackObjectsIdx.push_back(-1); // Fill index for possible dead.
    if (MFI.isDeadObjectIndex(I))
      continue;

    yaml::MachineStackObject YamlObject;
    YamlObject.ID = ID;
    if (const auto *Alloca = MFI.getObjectAllocation(I))
      YamlObject.Name.Value = std::string(
          Alloca->hasName() ? Alloca->getName() : "");
    YamlObject.Type = MFI.isSpillSlotObjectIndex(I)
                          ? yaml::MachineStackObject::SpillSlot
                          : MFI.isVariableSizedObjectIndex(I)
                                ? yaml::MachineStackObject::VariableSized
                                : yaml::MachineStackObject::DefaultType;
    YamlObject.Offset = MFI.getObjectOffset(I);
    YamlObject.Size = MFI.getObjectSize(I);
    YamlObject.Alignment = MFI.getObjectAlign(I);
    YamlObject.StackID = (TargetStackID::Value)MFI.getStackID(I);

    // Save the ID' position in StackObjects storage vector.
    StackObjectsIdx[ID] = YMF.StackObjects.size();
    YMF.StackObjects.push_back(YamlObject);
    State.StackObjectOperandMapping.insert(std::make_pair(
        I, FrameIndexOperand::create(YamlObject.Name.Value, ID)));
  }

  for (const auto &CSInfo : MFI.getCalleeSavedInfo()) {
    const int FrameIdx = CSInfo.getFrameIdx();
    if (!CSInfo.isSpilledToReg() && MFI.isDeadObjectIndex(FrameIdx))
      continue;

    yaml::StringValue Reg;
    printRegMIR(CSInfo.getReg(), Reg, TRI);
    if (!CSInfo.isSpilledToReg()) {
      assert(FrameIdx >= MFI.getObjectIndexBegin() &&
             FrameIdx < MFI.getObjectIndexEnd() &&
             "Invalid stack object index");
      if (FrameIdx < 0) { // Negative index means fixed objects.
        auto &Object =
            YMF.FixedStackObjects
                [FixedStackObjectsIdx[FrameIdx + MFI.getNumFixedObjects()]];
        Object.CalleeSavedRegister = std::move(Reg);
        Object.CalleeSavedRestored = CSInfo.isRestored();
      } else {
        auto &Object = YMF.StackObjects[StackObjectsIdx[FrameIdx]];
        Object.CalleeSavedRegister = std::move(Reg);
        Object.CalleeSavedRestored = CSInfo.isRestored();
      }
    }
  }
  for (unsigned I = 0, E = MFI.getLocalFrameObjectCount(); I < E; ++I) {
    auto LocalObject = MFI.getLocalFrameObjectMap(I);
    assert(LocalObject.first >= 0 && "Expected a locally mapped stack object");
    YMF.StackObjects[StackObjectsIdx[LocalObject.first]].LocalOffset =
        LocalObject.second;
  }

  // Print the stack object references in the frame information class after
  // converting the stack objects.
  if (MFI.hasStackProtectorIndex()) {
    raw_string_ostream StrOS(YMF.FrameInfo.StackProtector.Value);
    printStackObjectReference(StrOS, State, MFI.getStackProtectorIndex());
  }

  if (MFI.hasFunctionContextIndex()) {
    raw_string_ostream StrOS(YMF.FrameInfo.FunctionContext.Value);
    printStackObjectReference(StrOS, State, MFI.getFunctionContextIndex());
  }

  // Print the debug variable information.
  for (const MachineFunction::VariableDbgInfo &DebugVar :
       MF.getInStackSlotVariableDbgInfo()) {
    int Idx = DebugVar.getStackSlot();
    assert(Idx >= MFI.getObjectIndexBegin() && Idx < MFI.getObjectIndexEnd() &&
           "Invalid stack object index");
    if (Idx < 0) { // Negative index means fixed objects.
      auto &Object =
          YMF.FixedStackObjects[FixedStackObjectsIdx[Idx +
                                                     MFI.getNumFixedObjects()]];
      printStackObjectDbgInfo(DebugVar, Object, MST);
    } else {
      auto &Object = YMF.StackObjects[StackObjectsIdx[Idx]];
      printStackObjectDbgInfo(DebugVar, Object, MST);
    }
  }
}

static void convertCallSiteObjects(yaml::MachineFunction &YMF,
                                   const MachineFunction &MF,
                                   ModuleSlotTracker &MST) {
  const auto *TRI = MF.getSubtarget().getRegisterInfo();
  for (auto [MI, CallSiteInfo] : MF.getCallSitesInfo()) {
    yaml::CallSiteInfo YmlCS;
    yaml::MachineInstrLoc CallLocation;

    // Prepare instruction position.
    MachineBasicBlock::const_instr_iterator CallI = MI->getIterator();
    CallLocation.BlockNum = CallI->getParent()->getNumber();
    // Get call instruction offset from the beginning of block.
    CallLocation.Offset =
        std::distance(CallI->getParent()->instr_begin(), CallI);
    YmlCS.CallLocation = CallLocation;

    auto [ArgRegPairs, CalleeTypeIds] = CallSiteInfo;
    // Construct call arguments and theirs forwarding register info.
    for (auto ArgReg : ArgRegPairs) {
      yaml::CallSiteInfo::ArgRegPair YmlArgReg;
      YmlArgReg.ArgNo = ArgReg.ArgNo;
      printRegMIR(ArgReg.Reg, YmlArgReg.Reg, TRI);
      YmlCS.ArgForwardingRegs.emplace_back(YmlArgReg);
    }
    // Get type ids.
    for (auto *CalleeTypeId : CalleeTypeIds) {
      YmlCS.CalleeTypeIds.push_back(CalleeTypeId->getZExtValue());
    }
    YMF.CallSitesInfo.push_back(std::move(YmlCS));
  }

  // Sort call info by position of call instructions.
  llvm::sort(YMF.CallSitesInfo.begin(), YMF.CallSitesInfo.end(),
             [](yaml::CallSiteInfo A, yaml::CallSiteInfo B) {
               return std::tie(A.CallLocation.BlockNum, A.CallLocation.Offset) <
                      std::tie(B.CallLocation.BlockNum, B.CallLocation.Offset);
             });
}

static void convertMachineMetadataNodes(yaml::MachineFunction &YMF,
                                        const MachineFunction &MF,
                                        MachineModuleSlotTracker &MST) {
  MachineModuleSlotTracker::MachineMDNodeListType MDList;
  MST.collectMachineMDNodes(MDList);
  for (auto &MD : MDList) {
    std::string NS;
    raw_string_ostream StrOS(NS);
    MD.second->print(StrOS, MST, MF.getFunction().getParent());
    YMF.MachineMetadataNodes.push_back(std::move(NS));
  }
}

static void convertCalledGlobals(yaml::MachineFunction &YMF,
                                 const MachineFunction &MF,
                                 MachineModuleSlotTracker &MST) {
  for (const auto &[CallInst, CG] : MF.getCalledGlobals()) {
    yaml::MachineInstrLoc CallSite;
    CallSite.BlockNum = CallInst->getParent()->getNumber();
    CallSite.Offset = std::distance(CallInst->getParent()->instr_begin(),
                                    CallInst->getIterator());

    yaml::CalledGlobal YamlCG{CallSite, CG.Callee->getName().str(),
                              CG.TargetFlags};
    YMF.CalledGlobals.push_back(std::move(YamlCG));
  }

  // Sort by position of call instructions.
  llvm::sort(YMF.CalledGlobals.begin(), YMF.CalledGlobals.end(),
             [](yaml::CalledGlobal A, yaml::CalledGlobal B) {
               return std::tie(A.CallSite.BlockNum, A.CallSite.Offset) <
                      std::tie(B.CallSite.BlockNum, B.CallSite.Offset);
             });
}

static void convertMCP(yaml::MachineFunction &MF,
                       const MachineConstantPool &ConstantPool) {
  unsigned ID = 0;
  for (const MachineConstantPoolEntry &Constant : ConstantPool.getConstants()) {
    std::string Str;
    raw_string_ostream StrOS(Str);
    if (Constant.isMachineConstantPoolEntry())
      Constant.Val.MachineCPVal->print(StrOS);
    else
      Constant.Val.ConstVal->printAsOperand(StrOS);

    yaml::MachineConstantPoolValue YamlConstant;
    YamlConstant.ID = ID++;
    YamlConstant.Value = std::move(Str);
    YamlConstant.Alignment = Constant.getAlign();
    YamlConstant.IsTargetSpecific = Constant.isMachineConstantPoolEntry();

    MF.Constants.push_back(std::move(YamlConstant));
  }
}

static void
convertSRPoints(ModuleSlotTracker &MST,
                std::vector<yaml::SaveRestorePointEntry> &YamlSRPoints,
                ArrayRef<MachineBasicBlock *> SRPoints) {
  for (const auto &MBB : SRPoints) {
    SmallString<16> Str;
    yaml::SaveRestorePointEntry Entry;
    raw_svector_ostream StrOS(Str);
    StrOS << printMBBReference(*MBB);
    Entry.Point = StrOS.str().str();
    Str.clear();
    YamlSRPoints.push_back(Entry);
  }
}

static void convertMJTI(ModuleSlotTracker &MST, yaml::MachineJumpTable &YamlJTI,
                        const MachineJumpTableInfo &JTI) {
  YamlJTI.Kind = JTI.getEntryKind();
  unsigned ID = 0;
  for (const auto &Table : JTI.getJumpTables()) {
    std::string Str;
    yaml::MachineJumpTable::Entry Entry;
    Entry.ID = ID++;
    for (const auto *MBB : Table.MBBs) {
      raw_string_ostream StrOS(Str);
      StrOS << printMBBReference(*MBB);
      Entry.Blocks.push_back(Str);
      Str.clear();
    }
    YamlJTI.Entries.push_back(std::move(Entry));
  }
}

void llvm::guessSuccessors(const MachineBasicBlock &MBB,
                           SmallVectorImpl<MachineBasicBlock*> &Result,
                           bool &IsFallthrough) {
  SmallPtrSet<MachineBasicBlock*,8> Seen;

  for (const MachineInstr &MI : MBB) {
    if (MI.isPHI())
      continue;
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isMBB())
        continue;
      MachineBasicBlock *Succ = MO.getMBB();
      auto RP = Seen.insert(Succ);
      if (RP.second)
        Result.push_back(Succ);
    }
  }
  MachineBasicBlock::const_iterator I = MBB.getLastNonDebugInstr();
  IsFallthrough = I == MBB.end() || !I->isBarrier();
}

static bool canPredictSuccessors(const MachineBasicBlock &MBB) {
  SmallVector<MachineBasicBlock*,8> GuessedSuccs;
  bool GuessedFallthrough;
  guessSuccessors(MBB, GuessedSuccs, GuessedFallthrough);
  if (GuessedFallthrough) {
    const MachineFunction &MF = *MBB.getParent();
    MachineFunction::const_iterator NextI = std::next(MBB.getIterator());
    if (NextI != MF.end()) {
      MachineBasicBlock *Next = const_cast<MachineBasicBlock*>(&*NextI);
      if (!is_contained(GuessedSuccs, Next))
        GuessedSuccs.push_back(Next);
    }
  }
  if (GuessedSuccs.size() != MBB.succ_size())
    return false;
  return std::equal(MBB.succ_begin(), MBB.succ_end(), GuessedSuccs.begin());
}

static void printMI(raw_ostream &OS, MFPrintState &State,
                    const MachineInstr &MI);

static void printMIOperand(raw_ostream &OS, MFPrintState &State,
                           const MachineInstr &MI, unsigned OpIdx,
                           const TargetRegisterInfo *TRI,
                           const TargetInstrInfo *TII,
                           bool ShouldPrintRegisterTies,
                           SmallBitVector &PrintedTypes,
                           const MachineRegisterInfo &MRI, bool PrintDef);

void printMBB(raw_ostream &OS, MFPrintState &State,
              const MachineBasicBlock &MBB) {
  assert(MBB.getNumber() >= 0 && "Invalid MBB number");
  MBB.printName(OS,
                MachineBasicBlock::PrintNameIr |
                    MachineBasicBlock::PrintNameAttributes,
                &State.MST);
  OS << ":\n";

  bool HasLineAttributes = false;
  // Print the successors
  bool canPredictProbs = MBB.canPredictBranchProbabilities();
  // Even if the list of successors is empty, if we cannot guess it,
  // we need to print it to tell the parser that the list is empty.
  // This is needed, because MI model unreachable as empty blocks
  // with an empty successor list. If the parser would see that
  // without the successor list, it would guess the code would
  // fallthrough.
  if ((!MBB.succ_empty() && !SimplifyMIR) || !canPredictProbs ||
      !canPredictSuccessors(MBB)) {
    OS.indent(2) << "successors:";
    if (!MBB.succ_empty())
      OS << " ";
    ListSeparator LS;
    for (auto I = MBB.succ_begin(), E = MBB.succ_end(); I != E; ++I) {
      OS << LS << printMBBReference(**I);
      if (!SimplifyMIR || !canPredictProbs)
        OS << format("(0x%08" PRIx32 ")",
                     MBB.getSuccProbability(I).getNumerator());
    }
    OS << "\n";
    HasLineAttributes = true;
  }

  // Print the live in registers.
  const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  if (!MBB.livein_empty()) {
    const TargetRegisterInfo &TRI = *MRI.getTargetRegisterInfo();
    OS.indent(2) << "liveins: ";
    ListSeparator LS;
    for (const auto &LI : MBB.liveins_dbg()) {
      OS << LS << printReg(LI.PhysReg, &TRI);
      if (!LI.LaneMask.all())
        OS << ":0x" << PrintLaneMask(LI.LaneMask);
    }
    OS << "\n";
    HasLineAttributes = true;
  }

  if (HasLineAttributes && !MBB.empty())
    OS << "\n";
  bool IsInBundle = false;
  for (const MachineInstr &MI : MBB.instrs()) {
    if (IsInBundle && !MI.isInsideBundle()) {
      OS.indent(2) << "}\n";
      IsInBundle = false;
    }
    OS.indent(IsInBundle ? 4 : 2);
    printMI(OS, State, MI);
    if (!IsInBundle && MI.getFlag(MachineInstr::BundledSucc)) {
      OS << " {";
      IsInBundle = true;
    }
    OS << "\n";
  }
  if (IsInBundle)
    OS.indent(2) << "}\n";
}

static void printMI(raw_ostream &OS, MFPrintState &State,
                    const MachineInstr &MI) {
  const auto *MF = MI.getMF();
  const auto &MRI = MF->getRegInfo();
  const auto &SubTarget = MF->getSubtarget();
  const auto *TRI = SubTarget.getRegisterInfo();
  assert(TRI && "Expected target register info");
  const auto *TII = SubTarget.getInstrInfo();
  assert(TII && "Expected target instruction info");
  if (MI.isCFIInstruction())
    assert(MI.getNumOperands() == 1 && "Expected 1 operand in CFI instruction");

  SmallBitVector PrintedTypes(8);
  bool ShouldPrintRegisterTies = MI.hasComplexRegisterTies();
  ListSeparator LS;
  unsigned I = 0, E = MI.getNumOperands();
  for (; I < E; ++I) {
    const MachineOperand MO = MI.getOperand(I);
    if (!MO.isReg() || !MO.isDef() || MO.isImplicit())
      break;
    OS << LS;
    printMIOperand(OS, State, MI, I, TRI, TII, ShouldPrintRegisterTies,
                   PrintedTypes, MRI, /*PrintDef=*/false);
  }

  if (I)
    OS << " = ";
  if (MI.getFlag(MachineInstr::FrameSetup))
    OS << "frame-setup ";
  if (MI.getFlag(MachineInstr::FrameDestroy))
    OS << "frame-destroy ";
  if (MI.getFlag(MachineInstr::FmNoNans))
    OS << "nnan ";
  if (MI.getFlag(MachineInstr::FmNoInfs))
    OS << "ninf ";
  if (MI.getFlag(MachineInstr::FmNsz))
    OS << "nsz ";
  if (MI.getFlag(MachineInstr::FmArcp))
    OS << "arcp ";
  if (MI.getFlag(MachineInstr::FmContract))
    OS << "contract ";
  if (MI.getFlag(MachineInstr::FmAfn))
    OS << "afn ";
  if (MI.getFlag(MachineInstr::FmReassoc))
    OS << "reassoc ";
  if (MI.getFlag(MachineInstr::NoUWrap))
    OS << "nuw ";
  if (MI.getFlag(MachineInstr::NoSWrap))
    OS << "nsw ";
  if (MI.getFlag(MachineInstr::IsExact))
    OS << "exact ";
  if (MI.getFlag(MachineInstr::NoFPExcept))
    OS << "nofpexcept ";
  if (MI.getFlag(MachineInstr::NoMerge))
    OS << "nomerge ";
  if (MI.getFlag(MachineInstr::Unpredictable))
    OS << "unpredictable ";
  if (MI.getFlag(MachineInstr::NoConvergent))
    OS << "noconvergent ";
  if (MI.getFlag(MachineInstr::NonNeg))
    OS << "nneg ";
  if (MI.getFlag(MachineInstr::Disjoint))
    OS << "disjoint ";
  if (MI.getFlag(MachineInstr::NoUSWrap))
    OS << "nusw ";
  if (MI.getFlag(MachineInstr::SameSign))
    OS << "samesign ";
  if (MI.getFlag(MachineInstr::InBounds))
    OS << "inbounds ";

  // NOTE: Please add new MIFlags also to the MI_FLAGS_STR in
  // llvm/utils/update_mir_test_checks.py.

  OS << TII->getName(MI.getOpcode());

  LS = ListSeparator();

  if (I < E) {
    OS << ' ';
    for (; I < E; ++I) {
      OS << LS;
      printMIOperand(OS, State, MI, I, TRI, TII, ShouldPrintRegisterTies,
                     PrintedTypes, MRI, /*PrintDef=*/true);
    }
  }

  // Print any optional symbols attached to this instruction as-if they were
  // operands.
  if (MCSymbol *PreInstrSymbol = MI.getPreInstrSymbol()) {
    OS << LS << " pre-instr-symbol ";
    MachineOperand::printSymbol(OS, *PreInstrSymbol);
  }
  if (MCSymbol *PostInstrSymbol = MI.getPostInstrSymbol()) {
    OS << LS << " post-instr-symbol ";
    MachineOperand::printSymbol(OS, *PostInstrSymbol);
  }
  if (MDNode *HeapAllocMarker = MI.getHeapAllocMarker()) {
    OS << LS << " heap-alloc-marker ";
    HeapAllocMarker->printAsOperand(OS, State.MST);
  }
  if (MDNode *PCSections = MI.getPCSections()) {
    OS << LS << " pcsections ";
    PCSections->printAsOperand(OS, State.MST);
  }
  if (MDNode *MMRA = MI.getMMRAMetadata()) {
    OS << LS << " mmra ";
    MMRA->printAsOperand(OS, State.MST);
  }
  if (uint32_t CFIType = MI.getCFIType())
    OS << LS << " cfi-type " << CFIType;

  if (auto Num = MI.peekDebugInstrNum())
    OS << LS << " debug-instr-number " << Num;

  if (PrintLocations) {
    if (const DebugLoc &DL = MI.getDebugLoc()) {
      OS << LS << " debug-location ";
      DL->printAsOperand(OS, State.MST);
    }
  }

  if (!MI.memoperands_empty()) {
    OS << " :: ";
    const LLVMContext &Context = MF->getFunction().getContext();
    const MachineFrameInfo &MFI = MF->getFrameInfo();
    LS = ListSeparator();
    for (const auto *Op : MI.memoperands()) {
      OS << LS;
      Op->print(OS, State.MST, State.SSNs, Context, &MFI, TII);
    }
  }
}

static std::string formatOperandComment(std::string Comment) {
  if (Comment.empty())
    return Comment;
  return std::string(" /* " + Comment + " */");
}

static void printMIOperand(raw_ostream &OS, MFPrintState &State,
                           const MachineInstr &MI, unsigned OpIdx,
                           const TargetRegisterInfo *TRI,
                           const TargetInstrInfo *TII,
                           bool ShouldPrintRegisterTies,
                           SmallBitVector &PrintedTypes,
                           const MachineRegisterInfo &MRI, bool PrintDef) {
  LLT TypeToPrint = MI.getTypeToPrint(OpIdx, PrintedTypes, MRI);
  const MachineOperand &Op = MI.getOperand(OpIdx);
  std::string MOComment = TII->createMIROperandComment(MI, Op, OpIdx, TRI);

  switch (Op.getType()) {
  case MachineOperand::MO_Immediate:
    if (MI.isOperandSubregIdx(OpIdx)) {
      MachineOperand::printTargetFlags(OS, Op);
      MachineOperand::printSubRegIdx(OS, Op.getImm(), TRI);
      break;
    }
    [[fallthrough]];
  case MachineOperand::MO_Register:
  case MachineOperand::MO_CImmediate:
  case MachineOperand::MO_FPImmediate:
  case MachineOperand::MO_MachineBasicBlock:
  case MachineOperand::MO_ConstantPoolIndex:
  case MachineOperand::MO_TargetIndex:
  case MachineOperand::MO_JumpTableIndex:
  case MachineOperand::MO_ExternalSymbol:
  case MachineOperand::MO_GlobalAddress:
  case MachineOperand::MO_RegisterLiveOut:
  case MachineOperand::MO_Metadata:
  case MachineOperand::MO_MCSymbol:
  case MachineOperand::MO_CFIIndex:
  case MachineOperand::MO_IntrinsicID:
  case MachineOperand::MO_Predicate:
  case MachineOperand::MO_BlockAddress:
  case MachineOperand::MO_DbgInstrRef:
  case MachineOperand::MO_ShuffleMask: {
    unsigned TiedOperandIdx = 0;
    if (ShouldPrintRegisterTies && Op.isReg() && Op.isTied() && !Op.isDef())
      TiedOperandIdx = Op.getParent()->findTiedOperandIdx(OpIdx);
    Op.print(OS, State.MST, TypeToPrint, OpIdx, PrintDef,
             /*IsStandalone=*/false, ShouldPrintRegisterTies, TiedOperandIdx,
             TRI);
    OS << formatOperandComment(MOComment);
    break;
  }
  case MachineOperand::MO_FrameIndex:
    printStackObjectReference(OS, State, Op.getIndex());
    break;
  case MachineOperand::MO_RegisterMask: {
    const auto &RegisterMaskIds = State.RegisterMaskIds;
    auto RegMaskInfo = RegisterMaskIds.find(Op.getRegMask());
    if (RegMaskInfo != RegisterMaskIds.end())
      OS << StringRef(TRI->getRegMaskNames()[RegMaskInfo->second]).lower();
    else
      printCustomRegMask(Op.getRegMask(), OS, TRI);
    break;
  }
  }
}

void MIRFormatter::printIRValue(raw_ostream &OS, const Value &V,
                                ModuleSlotTracker &MST) {
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
  int Slot = MST.getCurrentFunction() ? MST.getLocalSlot(&V) : -1;
  MachineOperand::printIRSlotNumber(OS, Slot);
}

void llvm::printMIR(raw_ostream &OS, const Module &M) {
  yaml::Output Out(OS);
  Out << const_cast<Module &>(M);
}

void llvm::printMIR(raw_ostream &OS, const MachineModuleInfo &MMI,
                    const MachineFunction &MF) {
  printMF(OS, MMI, MF);
}

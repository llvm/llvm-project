//=- RISCVMachineFunctionInfo.cpp - RISC-V machine function info --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares RISCV-specific per-machine-function information.
//
//===----------------------------------------------------------------------===//

#include "RISCVMachineFunctionInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/RISCVZicfilpHash.h"

using namespace llvm;

enum class RawZicfilpLabelSource {
  Missing,
  CFITypeHashAttribute,
  GlobalCtorOrDtorCallback,
};

struct RawZicfilpLabel {
  uint32_t Value = 0;
  RawZicfilpLabelSource Source = RawZicfilpLabelSource::Missing;

  bool isAvailable() const { return Source != RawZicfilpLabelSource::Missing; }
};

static bool isDlRuntimeResolveFunction(const Function &F) {
  return F.getName() == "_dl_runtime_resolve" && !F.hasLocalLinkage();
}

static RawZicfilpLabel
getRawFunctionEntryLabel(const Function &F,
                         RISCVCFI::ZicfilpLabelSchemeKind Scheme);

static RISCVCFI::ZicfilpLabelSchemeKind
resolveZicfilpLabelScheme(const Module &M);

yaml::RISCVMachineFunctionInfo::RISCVMachineFunctionInfo(
    const llvm::RISCVMachineFunctionInfo &MFI)
    : VarArgsFrameIndex(MFI.getVarArgsFrameIndex()),
      VarArgsSaveSize(MFI.getVarArgsSaveSize()) {}

MachineFunctionInfo *RISCVMachineFunctionInfo::clone(
    BumpPtrAllocator &Allocator, MachineFunction &DestMF,
    const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
    const {
  return DestMF.cloneInfo<RISCVMachineFunctionInfo>(*this);
}

RISCVMachineFunctionInfo::RISCVMachineFunctionInfo(const Function &F,
                                                   const RISCVSubtarget *STI) {
  if (const auto *CFB = mdconst::extract_or_null<ConstantInt>(
          F.getParent()->getModuleFlag("cf-protection-branch")))
    CFProtectionBranch = CFB->getZExtValue() != 0;

  // The default stack probe size is 4096 if the function has no
  // stack-probe-size attribute. This is a safe default because it is the
  // smallest possible guard page size.
  uint64_t ProbeSize = 4096;
  if (F.hasFnAttribute("stack-probe-size"))
    ProbeSize = F.getFnAttributeAsParsedInteger("stack-probe-size");
  else if (const auto *PS = mdconst::extract_or_null<ConstantInt>(
               F.getParent()->getModuleFlag("stack-probe-size")))
    ProbeSize = PS->getZExtValue();
  assert(int64_t(ProbeSize) > 0 && "Invalid stack probe size");

  // Round down to the stack alignment.
  uint64_t StackAlign =
      STI->getFrameLowering()->getTransientStackAlign().value();
  ProbeSize = std::max(StackAlign, alignDown(ProbeSize, StackAlign));
  StringRef ProbeKind;
  if (F.hasFnAttribute("probe-stack"))
    ProbeKind = F.getFnAttribute("probe-stack").getValueAsString();
  else if (const auto *PS = dyn_cast_or_null<MDString>(
               F.getParent()->getModuleFlag("probe-stack")))
    ProbeKind = PS->getString();
  if (ProbeKind.size()) {
    StackProbeSize = ProbeSize;
  }

  NeedsFunctionEntryLpad = CFProtectionBranch &&
                           !F.hasFnAttribute("interrupt") &&
                           (F.hasAddressTaken() || !F.hasLocalLinkage());

  // abort here if cf-protection-branch is not enabled
  if (!CFProtectionBranch)
    return;

  PreferredLandingPadLabel = STI->getCLOpts().landing_pad_label;

  // 1. get scheme
  ZicfilpLabelScheme = resolveZicfilpLabelScheme(*F.getParent());

  // 1.1 __dl_runtime_resolve special case
  // FIXME: this check doesn't match the check for the callsite label
  if (isDlRuntimeResolveFunction(F)) {
    ZicfilpFnEntryLabel = 0;
    ZicfilpLocalJumpLabel = 0;
    return;
  }

  // 2. establish the function entry label
  const RawZicfilpLabel RawEntryLabel =
      getRawFunctionEntryLabel(F, ZicfilpLabelScheme);

  if (getNeedsFunctionEntryLpad()) {
    if (ZicfilpLabelScheme == RISCVCFI::ZicfilpLabelSchemeKind::FuncSig &&
        !RawEntryLabel.isAvailable()) {
      // FIXME: Absent metadata is tolerated for now
      // it's unknown wether this could be legitimately missing (i.e.
      // backend-generated function) report_fatal_error("missing cfi_type_hash
      // attribute");
      ZicfilpFnEntryLabel = 0;
    } else {
      ZicfilpFnEntryLabel =
          resolveZicfilpLabel(RawEntryLabel.Value, ZicfilpLabelScheme, PreferredLandingPadLabel);
    }
  } else {
    ZicfilpFnEntryLabel = 0;
  }

  // 3. establish the local jump label
  const uint32_t RawLocalJumpLabel =
      RawEntryLabel.isAvailable()
          ? generateRawLocalJumpLabel(RawEntryLabel.Value)
          : 0;

  if (!RawEntryLabel.isAvailable() &&
      ZicfilpLabelScheme == RISCVCFI::ZicfilpLabelSchemeKind::FuncSig) {
    ZicfilpLocalJumpLabel = 0;
  } else {
    ZicfilpLocalJumpLabel =
        resolveZicfilpLabel(RawLocalJumpLabel, ZicfilpLabelScheme, PreferredLandingPadLabel);
  }
}

RISCVMachineFunctionInfo::InterruptStackKind
RISCVMachineFunctionInfo::getInterruptStackKind(
    const MachineFunction &MF) const {
  if (!MF.getFunction().hasFnAttribute("interrupt"))
    return InterruptStackKind::None;

  assert(VarArgsSaveSize == 0 &&
         "Interrupt functions should not having incoming varargs");

  StringRef InterruptVal =
      MF.getFunction().getFnAttribute("interrupt").getValueAsString();

  return StringSwitch<RISCVMachineFunctionInfo::InterruptStackKind>(
             InterruptVal)
      .Case("qci-nest", InterruptStackKind::QCINest)
      .Case("qci-nonest", InterruptStackKind::QCINoNest)
      .Case("SiFive-CLIC-preemptible",
            InterruptStackKind::SiFiveCLICPreemptible)
      .Case("SiFive-CLIC-stack-swap", InterruptStackKind::SiFiveCLICStackSwap)
      .Case("SiFive-CLIC-preemptible-stack-swap",
            InterruptStackKind::SiFiveCLICPreemptibleStackSwap)
      .Default(InterruptStackKind::None);
}

void yaml::RISCVMachineFunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<RISCVMachineFunctionInfo>::mapping(YamlIO, *this);
}

RISCVMachineFunctionInfo::ShadowStackKind
RISCVMachineFunctionInfo::getShadowStackKind(const MachineFunction &MF) const {
  // Prefer HW Shadow Stack
  //
  // Zicfiss is encoded using a Zimop, so that's the only extension we have to
  // check here. MOPs will be compressed if they have the right structure to.
  if (MF.getSubtarget<RISCVSubtarget>().hasStdExtZimop() &&
      MF.getFunction().hasFnAttribute("hw-shadow-stack"))
    return ShadowStackKind::Hardware;

  // ShadowCallStack attribute is used for software shadow call stack
  if (MF.getFunction().hasFnAttribute(Attribute::ShadowCallStack))
    return ShadowStackKind::Software;

  // Otherwise, none
  return ShadowStackKind::None;
}

RISCVMachineFunctionInfo::PushPopKind
RISCVMachineFunctionInfo::getPushPopKind(const MachineFunction &MF) const {
  // We cannot use fixed locations for the callee saved spill slots if the
  // function uses a varargs save area.
  // TODO: Use a separate placement for vararg registers to enable Zcmp.
  if (VarArgsSaveSize != 0)
    return PushPopKind::None;

  // SiFive interrupts are not compatible with push/pop.
  if (useSiFiveInterrupt(MF))
    return PushPopKind::None;

  // Zcmp is not compatible with the frame pointer convention.
  if (MF.getSubtarget<RISCVSubtarget>().hasStdExtZcmp() &&
      !MF.disableFramePointerElim())
    return PushPopKind::StdExtZcmp;

  // Xqccmp is Zcmp but has a push order compatible with the frame-pointer
  // convention.
  if (MF.getSubtarget<RISCVSubtarget>().hasVendorXqccmp())
    return PushPopKind::VendorXqccmp;

  return PushPopKind::None;
}

bool RISCVMachineFunctionInfo::hasImplicitFPUpdates(
    const MachineFunction &MF) const {
  switch (getInterruptStackKind(MF)) {
  case InterruptStackKind::QCINest:
  case InterruptStackKind::QCINoNest:
    // QC.C.MIENTER and QC.C.MIENTER.NEST both update FP on function entry.
    return true;
  default:
    break;
  }

  switch (getPushPopKind(MF)) {
  case PushPopKind::VendorXqccmp:
    // When using Xqccmp, we will use `QC.CM.PUSHFP` when Frame Pointers are
    // enabled, which will update FP.
    return true;
  default:
    break;
  }

  return false;
}

void RISCVMachineFunctionInfo::initializeBaseYamlFields(
    const yaml::RISCVMachineFunctionInfo &YamlMFI) {
  VarArgsFrameIndex = YamlMFI.VarArgsFrameIndex;
  VarArgsSaveSize = YamlMFI.VarArgsSaveSize;
}

void RISCVMachineFunctionInfo::addSExt32Register(Register Reg) {
  SExt32Registers.push_back(Reg);
}

bool RISCVMachineFunctionInfo::isSExt32Register(Register Reg) const {
  return is_contained(SExt32Registers, Reg);
}

uint32_t RISCVMachineFunctionInfo::resolveZicfilpLabel(uint32_t Label,
                                    RISCVCFI::ZicfilpLabelSchemeKind Scheme,
                                    std::optional<uint32_t> PreferredLabel) {
  if (Label == 0) {
    if (Scheme != RISCVCFI::ZicfilpLabelSchemeKind::Unlabeled)
      report_fatal_error("supplied RISC-V landing pad label is 0 but selected "
                         "scheme is 'func-sig'");
    if (PreferredLabel) {
      if (!isUInt<20>(*PreferredLabel))
        report_fatal_error("riscv-landing-pad-label=<val>, <val> needs to "
                           "fit in unsigned 20-bits");
      Label = *PreferredLabel;
    }
  } else if (Scheme == RISCVCFI::ZicfilpLabelSchemeKind::Unlabeled) {
    report_fatal_error("supplied RISC-V landing pad label is not 0 but "
                       "selected scheme is 'unlabeled'");
  } else if (!isUInt<20>(Label)) {
    report_fatal_error("landing pad Label needs to fit in unsigned 20-bits");
  }
  return Label;
}

// Raw label from F's cfi_type_hash attribute. 0 if absent.
// rejects a present-but-unparsable value.
uint32_t RISCVMachineFunctionInfo::generateRawFnEntryLabel(const Function &F) {
  if (!F.hasFnAttribute("cfi_type_hash"))
    return 0;
  uint32_t ParsedLabel = 0;
  if (F.getFnAttribute("cfi_type_hash")
          .getValueAsString()
          .getAsInteger(10, ParsedLabel))
    report_fatal_error("malformed cfi_type_hash attribute");
  return ParsedLabel;
}

// Raw local-jump label, derived from the raw entry label via the +/-1
// anti-collision offset. 0 maps to 0.
uint32_t
RISCVMachineFunctionInfo::generateRawLocalJumpLabel(uint32_t RawFnEntryLabel) {
  if (RawFnEntryLabel == 0)
    return 0;
  return RawFnEntryLabel == 0xFFFFF ? RawFnEntryLabel - 1 : RawFnEntryLabel + 1;
}

uint32_t
RISCVMachineFunctionInfo::getZicfilpCallSiteLabel(const CallBase *CB) const {
  if (!CB || !CB->isIndirectCall() || !hasCFProtectionBranch())
    return 0;
  uint32_t Label = 0;

  auto *MD = CB->getMetadata("riscv_lpad_label");
  if (!MD) {
    if (getZicfilpLabelScheme() == RISCVCFI::ZicfilpLabelSchemeKind::FuncSig)
      return 0; // FIXME: Absent metadata under func-sig is tolerated for now
                // it's unknown wether this could be legitimately missing (i.e.
                // backend-generated call)
  } else {

    if (MD->getNumOperands() != 1)
      report_fatal_error(
          "riscv_lpad_label metadata must have exactly one operand");

    auto *CAM = dyn_cast<ConstantAsMetadata>(MD->getOperand(0));
    if (!CAM)
      report_fatal_error(
          "riscv_lpad_label metadata operand must be a constant");

    auto *CI = dyn_cast<ConstantInt>(CAM->getValue());
    if (!CI)
      report_fatal_error(
          "riscv_lpad_label metadata operand must be an integer constant");

    if (!CI->getType()->isIntegerTy(32))
      report_fatal_error("riscv_lpad_label metadata operand must be i32");

    Label = CI->getZExtValue();
    if (!isUInt<20>(Label))
      report_fatal_error("riscv_lpad_label metadata value out of range");
  }

  return resolveZicfilpLabel(Label, getZicfilpLabelScheme(), PreferredLandingPadLabel);
}

uint32_t RISCVMachineFunctionInfo::resolveZicfilpIndirectCallLabel(
    const MachineInstr &MI) {
  assert((MI.getOpcode() == RISCV::PseudoCALLIndirect ||
          MI.getOpcode() == RISCV::PseudoTAILIndirect) &&
         "expected an indirect call/tail-call pseudo");
  if (MI.getNumOperands() <= 1 || !MI.getOperand(1).isImm())
    report_fatal_error(
        "indirect call/tail-call pseudo missing landing-pad label operand");

  int64_t RawLabel = MI.getOperand(1).getImm();
  if (!isUInt<20>(RawLabel))
    report_fatal_error("landing pad Label needs to fit in unsigned 20-bits");

  return static_cast<uint32_t>(RawLabel);
  // actual resolving against scheme was done earlier.
}

static bool isFunctionListedInCtorDtorArray(const Function &F,
                                            StringRef ArrayName) {
  const Module *M = F.getParent();
  const GlobalVariable *GV = M->getNamedGlobal(ArrayName);
  if (!GV || !GV->hasInitializer())
    return false;

  const auto *Entries = dyn_cast<ConstantArray>(GV->getInitializer());
  if (!Entries)
    return false;

  for (const Use &Op : Entries->operands()) {
    const auto *Entry = dyn_cast<ConstantStruct>(Op.get());
    // { i32 Priority, ptr Func, ptr Data }
    if (!Entry || Entry->getNumOperands() < 2)
      continue;

    const auto *Callback =
        dyn_cast<Function>(Entry->getOperand(1)->stripPointerCastsAndAliases());
    if (Callback == &F)
      return true;
  }

  return false;
}

static bool isListedInGlobalCtorOrDtorArray(const Function &F) {
  return isFunctionListedInCtorDtorArray(F, "llvm.global_ctors") ||
         isFunctionListedInCtorDtorArray(F, "llvm.global_dtors");
}

static RawZicfilpLabel
getRawFunctionEntryLabel(const Function &F,
                         RISCVCFI::ZicfilpLabelSchemeKind Scheme) {
  if (F.hasFnAttribute("cfi_type_hash"))
    return {RISCVMachineFunctionInfo::generateRawFnEntryLabel(F),
            RawZicfilpLabelSource::CFITypeHashAttribute};

  // llvm.global_ctors / llvm.global_dtors callbacks have ABI type void().
  // Their psABI func-sig signature is therefore FvvE.
  if (Scheme == RISCVCFI::ZicfilpLabelSchemeKind::FuncSig &&
      isListedInGlobalCtorOrDtorArray(F))
    return {RISCVCFI::getZicfilpFuncSigLabel("FvvE"),
            RawZicfilpLabelSource::GlobalCtorOrDtorCallback};

  return {};
}

static RISCVCFI::ZicfilpLabelSchemeKind
resolveZicfilpLabelScheme(const Module &M) {
  const Metadata *MD = M.getModuleFlag("cf-branch-label-scheme");

  // Deliberate compatibility default for legacy IR with branch protection but
  // no scheme module flag. Clang-produced func-sig IR must carry the flag.
  if (!MD)
    return RISCVCFI::ZicfilpLabelSchemeKind::Unlabeled;

  const auto *MDS = dyn_cast<MDString>(MD);
  if (!MDS)
    report_fatal_error("invalid RISC-V landing pad label scheme");

  auto Scheme = RISCVCFI::getZicfilpLabelScheme(MDS->getString());
  if (Scheme == RISCVCFI::ZicfilpLabelSchemeKind::Invalid)
    report_fatal_error("unrecognized RISC-V landing pad label scheme");

  return Scheme;
}
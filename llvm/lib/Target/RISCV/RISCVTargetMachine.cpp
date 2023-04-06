//===-- RISCVTargetMachine.cpp - Define TargetMachine for RISC-V ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the info about RISC-V target spec.
//
//===----------------------------------------------------------------------===//

#include "RISCVTargetMachine.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVMacroFusion.h"
#include "RISCVTargetObjectFile.h"
#include "RISCVTargetTransformInfo.h"
#include "TargetInfo/RISCVTargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/MIRParser/MIParser.h"
#include "llvm/CodeGen/MIRYamlMapping.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO.h"
#include <optional>
using namespace llvm;

static cl::opt<bool> EnableRedundantCopyElimination(
    "riscv-enable-copyelim",
    cl::desc("Enable the redundant copy elimination pass"), cl::init(true),
    cl::Hidden);

// FIXME: Unify control over GlobalMerge.
static cl::opt<cl::boolOrDefault>
    EnableGlobalMerge("riscv-enable-global-merge", cl::Hidden,
                      cl::desc("Enable the global merge pass"));

static cl::opt<bool>
    EnableMachineCombiner("riscv-enable-machine-combiner",
                          cl::desc("Enable the machine combiner pass"),
                          cl::init(true), cl::Hidden);

static cl::opt<unsigned> RVVVectorBitsMaxOpt(
    "riscv-v-vector-bits-max",
    cl::desc("Assume V extension vector registers are at most this big, "
             "with zero meaning no maximum size is assumed."),
    cl::init(0), cl::Hidden);

static cl::opt<int> RVVVectorBitsMinOpt(
    "riscv-v-vector-bits-min",
    cl::desc("Assume V extension vector registers are at least this big, "
             "with zero meaning no minimum size is assumed. A value of -1 "
             "means use Zvl*b extension. This is primarily used to enable "
             "autovectorization with fixed width vectors."),
    cl::init(-1), cl::Hidden);

static cl::opt<bool> EnableRISCVCopyPropagation(
    "riscv-enable-copy-propagation",
    cl::desc("Enable the copy propagation with RISC-V copy instr"),
    cl::init(true), cl::Hidden);

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRISCVTarget() {
  RegisterTargetMachine<RISCVTargetMachine> X(getTheRISCV32Target());
  RegisterTargetMachine<RISCVTargetMachine> Y(getTheRISCV64Target());
  auto *PR = PassRegistry::getPassRegistry();
  initializeGlobalISel(*PR);
  initializeRISCVMakeCompressibleOptPass(*PR);
  initializeRISCVGatherScatterLoweringPass(*PR);
  initializeRISCVCodeGenPreparePass(*PR);
  initializeRISCVMergeBaseOffsetOptPass(*PR);
  initializeRISCVOptWInstrsPass(*PR);
  initializeRISCVPreRAExpandPseudoPass(*PR);
  initializeRISCVExpandPseudoPass(*PR);
  initializeRISCVInsertNTLHInstsPass(*PR);
  initializeRISCVInsertVSETVLIPass(*PR);
  initializeRISCVDAGToDAGISelPass(*PR);
  initializeRISCVInitUndefPass(*PR);
}

static StringRef computeDataLayout(const Triple &TT) {
  if (TT.isArch64Bit())
    return "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128";
  assert(TT.isArch32Bit() && "only RV32 and RV64 are currently supported");
  return "e-m:e-p:32:32-i64:64-n32-S128";
}

static Reloc::Model getEffectiveRelocModel(const Triple &TT,
                                           std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::Static);
}

RISCVTargetMachine::RISCVTargetMachine(const Target &T, const Triple &TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       std::optional<Reloc::Model> RM,
                                       std::optional<CodeModel::Model> CM,
                                       CodeGenOpt::Level OL, bool JIT)
    : LLVMTargetMachine(T, computeDataLayout(TT), TT, CPU, FS, Options,
                        getEffectiveRelocModel(TT, RM),
                        getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<RISCVELFTargetObjectFile>()) {
  initAsmInfo();

  // RISC-V supports the MachineOutliner.
  setMachineOutliner(true);
  setSupportsDefaultOutlining(true);

  if (TT.isOSFuchsia() && !TT.isArch64Bit())
    report_fatal_error("Fuchsia is only supported for 64-bit");
}

const RISCVSubtarget *
RISCVTargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute TuneAttr = F.getFnAttribute("tune-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");

  std::string CPU =
      CPUAttr.isValid() ? CPUAttr.getValueAsString().str() : TargetCPU;
  std::string TuneCPU =
      TuneAttr.isValid() ? TuneAttr.getValueAsString().str() : CPU;
  std::string FS =
      FSAttr.isValid() ? FSAttr.getValueAsString().str() : TargetFS;

  unsigned RVVBitsMin = RVVVectorBitsMinOpt;
  unsigned RVVBitsMax = RVVVectorBitsMaxOpt;

  Attribute VScaleRangeAttr = F.getFnAttribute(Attribute::VScaleRange);
  if (VScaleRangeAttr.isValid()) {
    if (!RVVVectorBitsMinOpt.getNumOccurrences())
      RVVBitsMin = VScaleRangeAttr.getVScaleRangeMin() * RISCV::RVVBitsPerBlock;
    std::optional<unsigned> VScaleMax = VScaleRangeAttr.getVScaleRangeMax();
    if (VScaleMax.has_value() && !RVVVectorBitsMaxOpt.getNumOccurrences())
      RVVBitsMax = *VScaleMax * RISCV::RVVBitsPerBlock;
  }

  if (RVVBitsMin != -1U) {
    // FIXME: Change to >= 32 when VLEN = 32 is supported.
    assert((RVVBitsMin == 0 || (RVVBitsMin >= 64 && RVVBitsMin <= 65536 &&
                                isPowerOf2_32(RVVBitsMin))) &&
           "V or Zve* extension requires vector length to be in the range of "
           "64 to 65536 and a power 2!");
    assert((RVVBitsMax >= RVVBitsMin || RVVBitsMax == 0) &&
           "Minimum V extension vector length should not be larger than its "
           "maximum!");
  }
  assert((RVVBitsMax == 0 || (RVVBitsMax >= 64 && RVVBitsMax <= 65536 &&
                              isPowerOf2_32(RVVBitsMax))) &&
         "V or Zve* extension requires vector length to be in the range of "
         "64 to 65536 and a power 2!");

  if (RVVBitsMin != -1U) {
    if (RVVBitsMax != 0) {
      RVVBitsMin = std::min(RVVBitsMin, RVVBitsMax);
      RVVBitsMax = std::max(RVVBitsMin, RVVBitsMax);
    }

    RVVBitsMin = llvm::bit_floor(
        (RVVBitsMin < 64 || RVVBitsMin > 65536) ? 0 : RVVBitsMin);
  }
  RVVBitsMax =
      llvm::bit_floor((RVVBitsMax < 64 || RVVBitsMax > 65536) ? 0 : RVVBitsMax);

  SmallString<512> Key;
  Key += "RVVMin";
  Key += std::to_string(RVVBitsMin);
  Key += "RVVMax";
  Key += std::to_string(RVVBitsMax);
  Key += CPU;
  Key += TuneCPU;
  Key += FS;
  auto &I = SubtargetMap[Key];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    auto ABIName = Options.MCOptions.getABIName();
    if (const MDString *ModuleTargetABI = dyn_cast_or_null<MDString>(
            F.getParent()->getModuleFlag("target-abi"))) {
      auto TargetABI = RISCVABI::getTargetABI(ABIName);
      if (TargetABI != RISCVABI::ABI_Unknown &&
          ModuleTargetABI->getString() != ABIName) {
        report_fatal_error("-target-abi option != target-abi module flag");
      }
      ABIName = ModuleTargetABI->getString();
    }
    I = std::make_unique<RISCVSubtarget>(
        TargetTriple, CPU, TuneCPU, FS, ABIName, RVVBitsMin, RVVBitsMax, *this);
  }
  return I.get();
}

MachineFunctionInfo *RISCVTargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return RISCVMachineFunctionInfo::create<RISCVMachineFunctionInfo>(Allocator,
                                                                    F, STI);
}

TargetTransformInfo
RISCVTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(RISCVTTIImpl(this, F));
}

// A RISC-V hart has a single byte-addressable address space of 2^XLEN bytes
// for all memory accesses, so it is reasonable to assume that an
// implementation has no-op address space casts. If an implementation makes a
// change to this, they can override it here.
bool RISCVTargetMachine::isNoopAddrSpaceCast(unsigned SrcAS,
                                             unsigned DstAS) const {
  return true;
}

namespace {
class RISCVPassConfig : public TargetPassConfig {
public:
  RISCVPassConfig(RISCVTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  RISCVTargetMachine &getRISCVTargetMachine() const {
    return getTM<RISCVTargetMachine>();
  }

  ScheduleDAGInstrs *
  createMachineScheduler(MachineSchedContext *C) const override {
    const RISCVSubtarget &ST = C->MF->getSubtarget<RISCVSubtarget>();
    if (ST.hasMacroFusion()) {
      ScheduleDAGMILive *DAG = createGenericSchedLive(C);
      DAG->addMutation(createRISCVMacroFusionDAGMutation());
      return DAG;
    }
    return nullptr;
  }

  ScheduleDAGInstrs *
  createPostMachineScheduler(MachineSchedContext *C) const override {
    const RISCVSubtarget &ST = C->MF->getSubtarget<RISCVSubtarget>();
    if (ST.hasMacroFusion()) {
      ScheduleDAGMI *DAG = createGenericSchedPostRA(C);
      DAG->addMutation(createRISCVMacroFusionDAGMutation());
      return DAG;
    }
    return nullptr;
  }

  void addIRPasses() override;
  bool addPreISel() override;
  bool addInstSelector() override;
  bool addIRTranslator() override;
  bool addLegalizeMachineIR() override;
  bool addRegBankSelect() override;
  bool addGlobalInstructionSelect() override;
  void addPreEmitPass() override;
  void addPreEmitPass2() override;
  void addPreSched2() override;
  void addMachineSSAOptimization() override;
  void addPreRegAlloc() override;
  void addPostRegAlloc() override;
  void addOptimizedRegAlloc() override;
};
} // namespace

TargetPassConfig *RISCVTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new RISCVPassConfig(*this, PM);
}

void RISCVPassConfig::addIRPasses() {
  addPass(createAtomicExpandPass());

  if (getOptLevel() != CodeGenOpt::None) {
    addPass(createRISCVGatherScatterLoweringPass());
    addPass(createInterleavedAccessPass());
    addPass(createRISCVCodeGenPreparePass());
  }

  TargetPassConfig::addIRPasses();
}

bool RISCVPassConfig::addPreISel() {
  if (TM->getOptLevel() != CodeGenOpt::None) {
    // Add a barrier before instruction selection so that we will not get
    // deleted block address after enabling default outlining. See D99707 for
    // more details.
    addPass(createBarrierNoopPass());
  }

  if (EnableGlobalMerge == cl::BOU_TRUE) {
    addPass(createGlobalMergePass(TM, /* MaxOffset */ 2047,
                                  /* OnlyOptimizeForSize */ false,
                                  /* MergeExternalByDefault */ true));
  }

  return false;
}

bool RISCVPassConfig::addInstSelector() {
  addPass(createRISCVISelDag(getRISCVTargetMachine(), getOptLevel()));

  return false;
}

bool RISCVPassConfig::addIRTranslator() {
  addPass(new IRTranslator(getOptLevel()));
  return false;
}

bool RISCVPassConfig::addLegalizeMachineIR() {
  addPass(new Legalizer());
  return false;
}

bool RISCVPassConfig::addRegBankSelect() {
  addPass(new RegBankSelect());
  return false;
}

bool RISCVPassConfig::addGlobalInstructionSelect() {
  addPass(new InstructionSelect(getOptLevel()));
  return false;
}

void RISCVPassConfig::addPreSched2() {}

void RISCVPassConfig::addPreEmitPass() {
  addPass(&BranchRelaxationPassID);
  addPass(createRISCVMakeCompressibleOptPass());

  // TODO: It would potentially be better to schedule copy propagation after
  // expanding pseudos (in addPreEmitPass2). However, performing copy
  // propagation after the machine outliner (which runs after addPreEmitPass)
  // currently leads to incorrect code-gen, where copies to registers within
  // outlined functions are removed erroneously.
  if (TM->getOptLevel() >= CodeGenOpt::Default && EnableRISCVCopyPropagation)
    addPass(createMachineCopyPropagationPass(true));
}

void RISCVPassConfig::addPreEmitPass2() {
  addPass(createRISCVExpandPseudoPass());
  addPass(createRISCVInsertNTLHInstsPass());

  // Schedule the expansion of AMOs at the last possible moment, avoiding the
  // possibility for other passes to break the requirements for forward
  // progress in the LR/SC block.
  addPass(createRISCVExpandAtomicPseudoPass());
}

void RISCVPassConfig::addMachineSSAOptimization() {
  TargetPassConfig::addMachineSSAOptimization();
  if (EnableMachineCombiner)
    addPass(&MachineCombinerID);

  if (TM->getTargetTriple().getArch() == Triple::riscv64) {
    addPass(createRISCVOptWInstrsPass());
  }
}

void RISCVPassConfig::addPreRegAlloc() {
  addPass(createRISCVPreRAExpandPseudoPass());
  if (TM->getOptLevel() != CodeGenOpt::None)
    addPass(createRISCVMergeBaseOffsetOptPass());
  addPass(createRISCVInsertVSETVLIPass());
}

void RISCVPassConfig::addOptimizedRegAlloc() {
  if (getOptimizeRegAlloc())
    insertPass(&DetectDeadLanesID, &RISCVInitUndefID);

  TargetPassConfig::addOptimizedRegAlloc();
}

void RISCVPassConfig::addPostRegAlloc() {
  if (TM->getOptLevel() != CodeGenOpt::None && EnableRedundantCopyElimination)
    addPass(createRISCVRedundantCopyEliminationPass());
}

yaml::MachineFunctionInfo *
RISCVTargetMachine::createDefaultFuncInfoYAML() const {
  return new yaml::RISCVMachineFunctionInfo();
}

yaml::MachineFunctionInfo *
RISCVTargetMachine::convertFuncInfoToYAML(const MachineFunction &MF) const {
  const auto *MFI = MF.getInfo<RISCVMachineFunctionInfo>();
  return new yaml::RISCVMachineFunctionInfo(*MFI);
}

bool RISCVTargetMachine::parseMachineFunctionInfo(
    const yaml::MachineFunctionInfo &MFI, PerFunctionMIParsingState &PFS,
    SMDiagnostic &Error, SMRange &SourceRange) const {
  const auto &YamlMFI =
      static_cast<const yaml::RISCVMachineFunctionInfo &>(MFI);
  PFS.MF.getInfo<RISCVMachineFunctionInfo>()->initializeBaseYamlFields(YamlMFI);
  return false;
}

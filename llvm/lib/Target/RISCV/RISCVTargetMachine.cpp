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
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVMachineScheduler.h"
#include "RISCVTargetObjectFile.h"
#include "RISCVTargetTransformInfo.h"
#include "TargetInfo/RISCVTargetInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/MIRParser/MIParser.h"
#include "llvm/CodeGen/MIRYamlMapping.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/MacroFusion.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/LibraryOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include <optional>
using namespace llvm;

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeRISCVTarget() {
  RegisterTargetMachine<RISCVTargetMachine> X(getTheRISCV32Target());
  RegisterTargetMachine<RISCVTargetMachine> Y(getTheRISCV64Target());
  RegisterTargetMachine<RISCVTargetMachine> A(getTheRISCV32beTarget());
  RegisterTargetMachine<RISCVTargetMachine> B(getTheRISCV64beTarget());
  static opt::RegisterLibraryOptions<RISCVOptions> O;
  auto *PR = PassRegistry::getPassRegistry();
  initializeGlobalISel(*PR);
  initializeRISCVO0PreLegalizerCombinerPass(*PR);
  initializeRISCVPreLegalizerCombinerPass(*PR);
  initializeRISCVPostLegalizerCombinerPass(*PR);
  initializeMachineKCFILegacyPass(*PR);
  initializeRISCVDeadRegisterDefinitionsPass(*PR);
  initializeRISCVLateBranchOptPass(*PR);
  initializeRISCVMakeCompressibleOptPass(*PR);
  initializeRISCVQCRelaxMarkingPass(*PR);
  initializeRISCVGatherScatterLoweringLegacyPass(*PR);
  initializeRISCVCodeGenPrepareLegacyPass(*PR);
  initializeRISCVZacasABIFixLegacyPass(*PR);
  initializeRISCVExpandPseudoPostRALegacyPass(*PR);
  initializeRISCVMergeBaseOffsetOptPass(*PR);
  initializeRISCVOptWInstrsLegacyPass(*PR);
  initializeRISCVFoldMemOffsetLegacyPass(*PR);
  initializeRISCVExpandPseudoPreRALegacyPass(*PR);
  initializeRISCVExpandPseudoPreEmitLegacyPass(*PR);
  initializeRISCVVectorPeepholeLegacyPass(*PR);
  initializeRISCVVLOptimizerLegacyPass(*PR);
  initializeRISCVVMV0EliminationPass(*PR);
  initializeRISCVInsertVSETVLIPass(*PR);
  initializeRISCVInsertReadWriteCSRPass(*PR);
  initializeRISCVInsertWriteVXRMPass(*PR);
  initializeRISCVDAGToDAGISelLegacyPass(*PR);
  initializeRISCVMoveMergePass(*PR);
  initializeRISCVPushPopOptPass(*PR);
  initializeRISCVIndirectBranchTrackingPass(*PR);
  initializeRISCVLoadStoreOptPass(*PR);
  initializeRISCVPreAllocZilsdOptPass(*PR);
  initializeRISCVExpandPseudoAtomicsLegacyPass(*PR);
  initializeRISCVRedundantCopyEliminationPass(*PR);
  initializeRISCVAsmPrinterPass(*PR);
  initializeRISCVPromoteConstantPass(*PR);
}

static Reloc::Model getEffectiveRelocModel(const Triple &TT,
                                           std::optional<Reloc::Model> RM) {
  if (TT.isOSBinFormatMachO())
    return RM.value_or(Reloc::PIC_);

  return RM.value_or(Reloc::Static);
}

static std::unique_ptr<TargetLoweringObjectFile> createTLOF(const Triple &TT) {
  if (TT.isOSBinFormatMachO())
    return std::make_unique<RISCVMachOTargetObjectFile>();
  return std::make_unique<RISCVELFTargetObjectFile>();
}

RISCVTargetMachine::RISCVTargetMachine(const Target &T, const Triple &TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       std::optional<Reloc::Model> RM,
                                       std::optional<CodeModel::Model> CM,
                                       CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, TT, CPU, FS, Options,
                               getEffectiveRelocModel(TT, RM),
                               getEffectiveCodeModel(CM, CodeModel::Small), OL),
      CLOpts(RISCVOptions::Global), TLOF(createTLOF(TT)) {
  initAsmInfo();

  // RISC-V supports the MachineOutliner.
  setMachineOutliner(true);
  setSupportsDefaultOutlining(true);

  // RISC-V supports the debug entry values.
  setSupportsDebugEntryValues(true);

  if (TT.isOSFuchsia() && !TT.isArch64Bit())
    report_fatal_error("Fuchsia is only supported for 64-bit");

  setCFIFixup(!CLOpts.enable_cfi_instr_inserter);
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

  unsigned RVVBitsMin = CLOpts.v_vector_bits_min.value_or(-1);
  unsigned RVVBitsMax = CLOpts.v_vector_bits_max.value_or(0);

  Attribute VScaleRangeAttr = F.getFnAttribute(Attribute::VScaleRange);
  if (VScaleRangeAttr.isValid()) {
    if (!CLOpts.v_vector_bits_min)
      RVVBitsMin = VScaleRangeAttr.getVScaleRangeMin() * RISCV::RVVBitsPerBlock;
    std::optional<unsigned> VScaleMax = VScaleRangeAttr.getVScaleRangeMax();
    if (VScaleMax.has_value() && !CLOpts.v_vector_bits_max)
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
  raw_svector_ostream(Key) << "RVVMin" << RVVBitsMin << "RVVMax" << RVVBitsMax
                           << CPU << TuneCPU << FS;
  auto &I = SubtargetMap[Key];
  if (!I) {
    StringRef ABIName = getTargetABIName(*F.getParent());
    I = std::make_unique<RISCVSubtarget>(
        TargetTriple, CPU, TuneCPU, FS, ABIName, RVVBitsMin, RVVBitsMax, *this);
  }
  return I.get();
}

MachineFunctionInfo *RISCVTargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return RISCVMachineFunctionInfo::create<RISCVMachineFunctionInfo>(
      Allocator, F, static_cast<const RISCVSubtarget *>(STI));
}

TargetTransformInfo
RISCVTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(std::make_unique<RISCVTTIImpl>(this, F));
}

// A RISC-V hart has a single byte-addressable address space of 2^XLEN bytes
// for all memory accesses, so it is reasonable to assume that an
// implementation has no-op address space casts. If an implementation makes a
// change to this, they can override it here.
bool RISCVTargetMachine::isNoopAddrSpaceCast(const DataLayout &DL,
                                             unsigned SrcAS,
                                             unsigned DstAS) const {
  return true;
}

ScheduleDAGInstrs *
RISCVTargetMachine::createMachineScheduler(MachineSchedContext *C) const {
  const RISCVSubtarget &ST = C->MF->getSubtarget<RISCVSubtarget>();
  ScheduleDAGMILive *DAG = createSchedLive<RISCVPreRAMachineSchedStrategy>(C);

  // Add MacroFusion mutation first with a higher priority than later clustering
  const auto &MacroFusions = ST.getMacroFusions();
  if (!MacroFusions.empty())
    DAG->addMutation(createMacroFusionDAGMutation(MacroFusions));

  if (ST.enableMISchedLoadClustering())
    DAG->addMutation(createLoadClusterDAGMutation(
        DAG->TII, /*ReorderWhileClustering=*/true));

  if (ST.enableMISchedStoreClustering())
    DAG->addMutation(createStoreClusterDAGMutation(
        DAG->TII, /*ReorderWhileClustering=*/true));

  if (CLOpts.vector_mask_mutation && ST.hasVInstructions())
    DAG->addMutation(createRISCVVectorMaskDAGMutation(DAG->TRI));

  return DAG;
}

ScheduleDAGInstrs *
RISCVTargetMachine::createPostMachineScheduler(MachineSchedContext *C) const {
  const RISCVSubtarget &ST = C->MF->getSubtarget<RISCVSubtarget>();
  ScheduleDAGMI *DAG = createSchedPostRA(C);

  // Add MacroFusion mutation first with a higher priority than later clustering
  const auto &MacroFusions = ST.getMacroFusions();
  if (!MacroFusions.empty())
    DAG->addMutation(createMacroFusionDAGMutation(MacroFusions));

  if (ST.enablePostMISchedLoadClustering())
    DAG->addMutation(createLoadClusterDAGMutation(
        DAG->TII, /*ReorderWhileClustering=*/true));

  if (ST.enablePostMISchedStoreClustering())
    DAG->addMutation(createStoreClusterDAGMutation(
        DAG->TII, /*ReorderWhileClustering=*/true));

  return DAG;
}

namespace {

class RVVRegisterRegAlloc : public RegisterRegAllocBase<RVVRegisterRegAlloc> {
public:
  RVVRegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
      : RegisterRegAllocBase(N, D, C) {}
};

static bool onlyAllocateRVVReg(const TargetRegisterInfo &TRI,
                               const MachineRegisterInfo &MRI,
                               const Register Reg) {
  const TargetRegisterClass *RC = MRI.getRegClass(Reg);
  return RISCVRegisterInfo::isRVVRegClass(RC);
}

static FunctionPass *useDefaultRegisterAllocator() { return nullptr; }

static llvm::once_flag InitializeDefaultRVVRegisterAllocatorFlag;

/// -riscv-rvv-regalloc=<fast|basic|greedy> command line option.
/// This option could designate the rvv register allocator only.
/// For example: -riscv-rvv-regalloc=basic
static cl::opt<RVVRegisterRegAlloc::FunctionPassCtor, false,
               RegisterPassParser<RVVRegisterRegAlloc>>
    RVVRegAlloc("riscv-rvv-regalloc", cl::Hidden,
                cl::init(&useDefaultRegisterAllocator),
                cl::desc("Register allocator to use for RVV register."));

static void initializeDefaultRVVRegisterAllocatorOnce() {
  RegisterRegAlloc::FunctionPassCtor Ctor = RVVRegisterRegAlloc::getDefault();

  if (!Ctor) {
    Ctor = RVVRegAlloc;
    RVVRegisterRegAlloc::setDefault(RVVRegAlloc);
  }
}

static FunctionPass *createBasicRVVRegisterAllocator() {
  return createBasicRegisterAllocator(onlyAllocateRVVReg);
}

static FunctionPass *createGreedyRVVRegisterAllocator() {
  return createGreedyRegisterAllocator(onlyAllocateRVVReg);
}

static FunctionPass *createFastRVVRegisterAllocator() {
  return createFastRegisterAllocator(onlyAllocateRVVReg, false);
}

static RVVRegisterRegAlloc basicRegAllocRVVReg("basic",
                                               "basic register allocator",
                                               createBasicRVVRegisterAllocator);
static RVVRegisterRegAlloc
    greedyRegAllocRVVReg("greedy", "greedy register allocator",
                         createGreedyRVVRegisterAllocator);

static RVVRegisterRegAlloc fastRegAllocRVVReg("fast", "fast register allocator",
                                              createFastRVVRegisterAllocator);

class RISCVPassConfig : public TargetPassConfig {
  const RISCVOptions &CLOpts;

public:
  RISCVPassConfig(RISCVTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM), CLOpts(TM.getCLOpts()) {
    if (TM.getOptLevel() != CodeGenOptLevel::None)
      substitutePass(&PostRASchedulerID, &PostMachineSchedulerID);
    setEnableSinkAndFold(CLOpts.enable_sink_fold);
    EnableLoopTermFold = true;
  }

  RISCVTargetMachine &getRISCVTargetMachine() const {
    return getTM<RISCVTargetMachine>();
  }

  void addIRPasses() override;
  bool addPreISel() override;
  void addCodeGenPrepare() override;
  bool addInstSelector() override;
  bool addIRTranslator() override;
  void addPreLegalizeMachineIR() override;
  bool addLegalizeMachineIR() override;
  void addPreRegBankSelect() override;
  bool addRegBankSelect() override;
  bool addGlobalInstructionSelect() override;
  void addPreEmitPass() override;
  void addPreEmitPass2() override;
  void addPreSched2() override;
  void addMachineSSAOptimization() override;
  FunctionPass *createRVVRegAllocPass(bool Optimized);
  bool addRegAssignAndRewriteFast() override;
  bool addRegAssignAndRewriteOptimized() override;
  void addPreRegAlloc() override;
  void addPostRegAlloc() override;
  void addFastRegAlloc() override;
  bool addILPOpts() override;

  std::unique_ptr<CSEConfigBase> getCSEConfig() const override;
};
} // namespace

TargetPassConfig *RISCVTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new RISCVPassConfig(*this, PM);
}

std::unique_ptr<CSEConfigBase> RISCVPassConfig::getCSEConfig() const {
  return getStandardCSEConfigForOpt(TM->getOptLevel());
}

FunctionPass *RISCVPassConfig::createRVVRegAllocPass(bool Optimized) {
  // Initialize the global default.
  llvm::call_once(InitializeDefaultRVVRegisterAllocatorFlag,
                  initializeDefaultRVVRegisterAllocatorOnce);

  RegisterRegAlloc::FunctionPassCtor Ctor = RVVRegisterRegAlloc::getDefault();
  if (Ctor != useDefaultRegisterAllocator)
    return Ctor();

  if (Optimized)
    return createGreedyRVVRegisterAllocator();

  return createFastRVVRegisterAllocator();
}

bool RISCVPassConfig::addRegAssignAndRewriteFast() {
  addPass(createRVVRegAllocPass(false));
  addPass(createRISCVInsertVSETVLIPass());
  if (TM->getOptLevel() != CodeGenOptLevel::None && CLOpts.enable_dead_defs)
    addPass(createRISCVDeadRegisterDefinitionsPass());
  return TargetPassConfig::addRegAssignAndRewriteFast();
}

bool RISCVPassConfig::addRegAssignAndRewriteOptimized() {
  addPass(createRVVRegAllocPass(true));
  addPass(createVirtRegRewriter(false));
  addPass(createRISCVInsertVSETVLIPass());
  if (TM->getOptLevel() != CodeGenOptLevel::None && CLOpts.enable_dead_defs)
    addPass(createRISCVDeadRegisterDefinitionsPass());
  return TargetPassConfig::addRegAssignAndRewriteOptimized();
}

void RISCVPassConfig::addIRPasses() {
  addPass(createAtomicExpandLegacyPass());
  addPass(createRISCVZacasABIFixLegacyPass());

  if (getOptLevel() != CodeGenOptLevel::None) {
    if (CLOpts.enable_loop_data_prefetch)
      addPass(createLoopDataPrefetchPass());

    addPass(createRISCVGatherScatterLoweringLegacyPass());
    addPass(createInterleavedAccessPass());
    addPass(createRISCVCodeGenPrepareLegacyPass());
  }

  TargetPassConfig::addIRPasses();

  if (getOptLevel() == CodeGenOptLevel::Aggressive && CLOpts.select_opt)
    addPass(createSelectOptimizePass());
}

bool RISCVPassConfig::addPreISel() {
  if (TM->getOptLevel() != CodeGenOptLevel::None)
    addPass(createRISCVPromoteConstantPass());
  if (TM->getOptLevel() != CodeGenOptLevel::None) {
    // Add a barrier before instruction selection so that we will not get
    // deleted block address after enabling default outlining. See D99707 for
    // more details.
    addPass(createBarrierNoopPass());
  }

  if (valueOr(CLOpts.enable_global_merge,
              TM->getOptLevel() != CodeGenOptLevel::None)) {
    // FIXME: Like AArch64, we disable extern global merging by default due to
    // concerns it might regress some workloads. Unlike AArch64, we don't
    // currently support enabling the pass in an "OnlyOptimizeForSize" mode.
    // Investigating and addressing both items are TODO.
    addPass(createGlobalMergePass(TM, /* MaxOffset */ 2047,
                                  /* OnlyOptimizeForSize */ false,
                                  /* MergeExternalByDefault */ true));
  }

  return false;
}

void RISCVPassConfig::addCodeGenPrepare() {
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createTypePromotionLegacyPass());
  TargetPassConfig::addCodeGenPrepare();
}

bool RISCVPassConfig::addInstSelector() {
  addPass(createRISCVISelDagLegacyPass(getRISCVTargetMachine(), getOptLevel()));

  return false;
}

bool RISCVPassConfig::addIRTranslator() {
  addPass(new IRTranslatorLegacy(getOptLevel()));
  return false;
}

void RISCVPassConfig::addPreLegalizeMachineIR() {
  if (getOptLevel() == CodeGenOptLevel::None) {
    addPass(createRISCVO0PreLegalizerCombiner());
  } else {
    addPass(createRISCVPreLegalizerCombiner());
  }
}

bool RISCVPassConfig::addLegalizeMachineIR() {
  addPass(new LegalizerLegacy());
  return false;
}

void RISCVPassConfig::addPreRegBankSelect() {
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createRISCVPostLegalizerCombiner());
}

bool RISCVPassConfig::addRegBankSelect() {
  addPass(new RegBankSelectLegacy());
  return false;
}

bool RISCVPassConfig::addGlobalInstructionSelect() {
  addPass(new InstructionSelectLegacy(getOptLevel()));
  return false;
}

void RISCVPassConfig::addPreSched2() {
  addPass(createRISCVExpandPseudoPostRALegacyPass());

  // Emit KCFI checks for indirect calls.
  addPass(createKCFIPass());
  if (TM->getOptLevel() != CodeGenOptLevel::None)
    addPass(createRISCVLoadStoreOptPass());
}

void RISCVPassConfig::addPreEmitPass() {
  // TODO: It would potentially be better to schedule copy propagation after
  // expanding pseudos (in addPreEmitPass2). However, performing copy
  // propagation after the machine outliner (which runs after addPreEmitPass)
  // currently leads to incorrect code-gen, where copies to registers within
  // outlined functions are removed erroneously.
  if (TM->getOptLevel() >= CodeGenOptLevel::Default &&
      CLOpts.enable_copy_propagation)
    addPass(createMachineCopyPropagationPass(true));
  if (TM->getOptLevel() >= CodeGenOptLevel::Default)
    addPass(createRISCVLateBranchOptPass());
  // The IndirectBranchTrackingPass inserts lpad and could have changed the
  // basic block alignment. It must be done before Branch Relaxation to
  // prevent the adjusted offset exceeding the branch range.
  addPass(createRISCVIndirectBranchTrackingPass());
  addPass(&BranchRelaxationPassID);
  addPass(createRISCVMakeCompressibleOptPass());
}

void RISCVPassConfig::addPreEmitPass2() {
  if (TM->getOptLevel() != CodeGenOptLevel::None) {
    addPass(createRISCVMoveMergePass());
    // Schedule PushPop Optimization before expansion of Pseudo instruction,
    // ensuring return instruction is detected correctly.
    addPass(createRISCVPushPopOptimizationPass());
  }
  addPass(createRISCVExpandPseudoPreEmitLegacyPass());

  // Add QC Relaxation Markers as late as possible, and only for RV32
  if (TM->getOptLevel() != CodeGenOptLevel::None &&
      TM->getTargetTriple().isRISCV32())
    addPass(createRISCVQCRelaxMarkingPass());

  // Schedule the expansion of AMOs at the last possible moment, avoiding the
  // possibility for other passes to break the requirements for forward
  // progress in the LR/SC block.
  addPass(createRISCVExpandPseudoAtomicsLegacyPass());

  // KCFI indirect call checks are lowered to a bundle.
  addPass(createUnpackMachineBundlesLegacy([&](const MachineFunction &MF) {
    return MF.getFunction().getParent()->getModuleFlag("kcfi");
  }));

  if (CLOpts.enable_cfi_instr_inserter)
    addPass(createCFIInstrInserterLegacy());
}

void RISCVPassConfig::addMachineSSAOptimization() {
  // It's beneficial to reduce the VL to enable more
  // Machine SSA optimizations.
  if (TM->getOptLevel() != CodeGenOptLevel::None) {
    // RISCVVLOptimizer can make loop invariant instructions like vmv.v.i
    // loop variant by propagating a VL defined inside the loop. Run LICM and
    // hoist them early. Don't do this at -O0 to avoid the compile-time
    // overhead. Not reducing the VL of loop invariant pseudos results in more
    // vsetvli toggles, and still requires the MachineLoopInfo analysis to be
    // run.
    addPass(&EarlyMachineLICMID);
    addPass(createRISCVVLOptimizerLegacyPass());
  }

  addPass(createRISCVVectorPeepholeLegacyPass());
  addPass(createRISCVFoldMemOffsetLegacyPass());

  TargetPassConfig::addMachineSSAOptimization();

  if (TM->getTargetTriple().isRISCV64()) {
    addPass(createRISCVOptWInstrsLegacyPass());
  }
}

void RISCVPassConfig::addPreRegAlloc() {
  addPass(createRISCVExpandPseudoPreRALegacyPass());
  if (TM->getOptLevel() != CodeGenOptLevel::None) {
    addPass(createRISCVMergeBaseOffsetOptPass());
    // Add Zilsd pre-allocation load/store optimization
    addPass(createRISCVPreAllocZilsdOptPass());
  }

  addPass(createRISCVInsertReadWriteCSRPass());
  addPass(createRISCVInsertWriteVXRMPass());
  addPass(createRISCVLandingPadSetupPass());

  if (TM->getOptLevel() != CodeGenOptLevel::None && CLOpts.enable_pipeliner)
    addPass(&MachinePipelinerID);

  addPass(createRISCVVMV0EliminationPass());
}

void RISCVPassConfig::addFastRegAlloc() {
  addPass(&InitUndefID);
  TargetPassConfig::addFastRegAlloc();
}


void RISCVPassConfig::addPostRegAlloc() {
  if (TM->getOptLevel() != CodeGenOptLevel::None && CLOpts.enable_copyelim)
    addPass(createRISCVRedundantCopyEliminationPass());
}

bool RISCVPassConfig::addILPOpts() {
  if (CLOpts.enable_machine_combiner)
    addPass(&MachineCombinerID);

  return true;
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

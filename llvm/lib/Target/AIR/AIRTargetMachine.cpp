//===- AIRTargetMachine.cpp - AIR Target Implementation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the AIR (Apple Intermediate Representation) target initializer and pipeline.
///
//===----------------------------------------------------------------------===//

#include "AIRTargetMachine.h"
#include "AIRAliasAnalysis.h"
#include "AIRWriter/AIREmbedderPass.h"
#include "AIRWriter/AIRWriterPass.h"
#include "AIR.h"
#include "AIRArgumentBuffer.h"
#include "AIRSystemValues.h"
#include "AIRAsyncEventToAlloca.h"
#include "AIRBFloat16CastDecompose.h"
#include "AIRBarrierRename.h"
#include "AIRCrossBufferStoreSeparate.h"
#include "AIRAliasAnnotate.h"
#include "AIRDemoteF64.h"
#include "AIREmulateF64.h"
#include "AIRDeviceLoadsVolatile.h"
#include "AIRInlineNonKernel.h"
#include "LLVMToAIRIntrinsics.h"
#include "AIRLowerAtomicRMW.h"
#include "AIRLowerFNeg.h"
#include "AIRSimdDivergenceGuard.h"
#include "AIRLegalizeUnsupportedIR.h"
#include "AIRNaNMinMax.h"
#include "AIRNormalizeAllocas.h"
#include "AIRPrepare.h"
#include "AIRScalarArgsToConstantBuffers.h"
#include "AIRScalarBufferPacking.h"
#include "AIRScalarStoreGuard.h"
#include "AIRSplitI64Shuffle.h"
#include "AIRSubtarget.h"
#include "AIRTGBarrierInsert.h"
#include "AIRTGGlobalCoalesce.h"
#include "AIRTargetTransformInfo.h"
#include "TargetInfo/AIRTargetInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionAIRLib.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Transforms/Utils.h"
#include <optional>

using namespace llvm;

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeAIRTarget() {
  RegisterTargetMachine<AIRTargetMachine> X(getTheAIRTarget());
  auto *PR = PassRegistry::getPassRegistry();
  // TargetPassConfig::addIRPasses() schedules these through the legacy PM.
  // They must be registered in THIS image's PassRegistry: the target lib is
  // a dylib with its own statically-linked LLVM copy, so registrations done
  // in the driver binary land in a different registry singleton.
  initializeCore(*PR);
  initializeCodeGen(*PR);
  initializeScalarOpts(*PR);
  initializeTransformUtils(*PR);
  initializeAnalysis(*PR);
  initializeLoopStrengthReducePass(*PR);
  initializeUnreachableBlockElimLegacyPassPass(*PR);
  initializeConstantHoistingLegacyPassPass(*PR);
  initializeScalarizeMaskedMemIntrinLegacyPassPass(*PR);
  initializePostInlineEntryExitInstrumenterPass(*PR);
  initializeAIRFromGenericGPUPass(*PR);
  initializeAIRInlineNonKernelLegacyPass(*PR);
  initializeAIRDemoteF64LegacyPass(*PR);
  initializeAIREmulateF64LegacyPass(*PR);
  initializeAIRLowerFNegLegacyPass(*PR);
  initializeAIRSimdDivergenceGuardLegacyPass(*PR);
  initializeAIRLegalizeUnsupportedIRLegacyPass(*PR);
  initializeAIRNaNMinMaxLegacyPass(*PR);
  initializeLLVMToAIRIntrinsicsLegacyPass(*PR);
  initializeAIRBarrierRenameLegacyPass(*PR);
  initializeAIRLowerAtomicRMWLegacyPass(*PR);
  initializeAIRSplitI64ShuffleLegacyPass(*PR);
  initializeAIRScalarStoreGuardLegacyPass(*PR);
  initializeAIRTGGlobalCoalesceLegacyPass(*PR);
  initializeAIRTGBarrierInsertLegacyPass(*PR);
  initializeAIRDeviceLoadsVolatileLegacyPass(*PR);
  initializeAIRAsyncEventToAllocaLegacyPass(*PR);
  initializeAIRNormalizeAllocasLegacyPass(*PR);
  initializeAIRBFloat16CastDecomposeLegacyPass(*PR);
  initializeAIRScalarBufferPackingLegacyPass(*PR);
  initializeAIRScalarArgsToConstantBuffersLegacyPass(*PR);
  initializeAIRArgumentBufferLegacyPass(*PR);
  initializeAIRSystemValuesLegacyPass(*PR);
  initializeAIRAliasAnnotateLegacyPass(*PR);
  initializeAIRPrepareLegacyPass(*PR);
  initializeAIRCrossBufferStoreSeparateLegacyPass(*PR);
  initializeAIREmbedderLegacyPassPass(*PR);
}

namespace {
/// Object-file lowering stub: the AIR backend serializes to a .metallib
/// container rather than going through the normal section machinery.
class AIRTargetObjectFile : public TargetLoweringObjectFile {
public:
  AIRTargetObjectFile() = default;

  MCSection *getExplicitSectionGlobal(const GlobalObject *GO, SectionKind Kind,
                                      const TargetMachine &) const override {
    return getContext().getAIRLibSection(GO->getSection(), Kind);
  }

protected:
  MCSection *SelectSectionForGlobal(const GlobalObject *, SectionKind,
                                    const TargetMachine &) const override {
    llvm_unreachable("Not supported!");
  }
};

class AIRPassConfig : public TargetPassConfig {
public:
  AIRPassConfig(AIRTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  AIRTargetMachine &getAIRTargetMachine() const {
    return getTM<AIRTargetMachine>();
  }

  FunctionPass *createTargetRegisterAllocator(bool) override { return nullptr; }

  void addCodeGenPrepare() override {
    // IR-to-AIR conformance passes, in order.
    // AIR bitcode has no switch encoding; lower to branch chains first.
    addPass(createLowerSwitchPass());
    // Remap generic GPU (OpenCL/SPIR-V) builtins -> AIR intrinsics so non-Triton
    // frontends (MLIR gpu dialect, SYCL) target AIR. No-op on Triton IR.
    addPass(createAIRFromGenericGPULegacyPass());
    // AGX-1 (cross-buffer same-offset device-store warp-0 miscompile).
    // Run EARLY, while the in-bounds predicate icmp (and its assume) are still
    // intact, so the separation guard can use the REAL mask. Sinks the run of
    // conflicting cross-buffer stores behind one shared in-bounds branch; a
    // no-op on single-output kernels (needs >=2 device-output buffers writing
    // the same per-thread offset). See AIRCrossBufferStoreSeparate.cpp.
    addPass(createAIRCrossBufferStoreSeparateLegacyPass());
    addPass(createAIRInlineNonKernelLegacyPass());
    addPass(createAIRDemoteF64LegacyPass());
    addPass(createAIRLowerFNegLegacyPass());
    addPass(createAIRLegalizeUnsupportedIRLegacyPass());
    addPass(createAIRNaNMinMaxLegacyPass(/*SignedZeros=*/true));
    addPass(createLLVMToAIRIntrinsicsLegacyPass());
    addPass(createAIRBarrierRenameLegacyPass());
    addPass(createAIRLowerAtomicRMWLegacyPass());
    addPass(createAIRSplitI64ShuffleLegacyPass());
    addPass(createAIRScalarStoreGuardLegacyPass());
    addPass(createAIRTGGlobalCoalesceLegacyPass());
    addPass(createAIRTGBarrierInsertLegacyPass());
    addPass(createAIRDeviceLoadsVolatileLegacyPass());
    addPass(createAIRAsyncEventToAllocaLegacyPass());
    addPass(createAIRNormalizeAllocasLegacyPass());
    addPass(createAIRBFloat16CastDecomposeLegacyPass());
    // Pack device/constant buffers into a Metal argument buffer (for the
    // gpu-dialect/IREE path that lacks pre-baked !air.kernel metadata). Must run
    // BEFORE AIRSystemValues so the indirect-buffer metadata is emitted against
    // the packed signature.
    addPass(createAIRArgumentBufferLegacyPass());
    // Must run BEFORE AIRSystemValues so that !air.kernel metadata is
    // emitted against the post-packing signature.
    addPass(createAIRScalarBufferPackingLegacyPass());
    addPass(createAIRSystemValuesLegacyPass());
    // Emit Apple-style alias-scope MD + "air-buffer-no-alias" param attrs
    // after the IR shape is final but before final normalisations.
    addPass(createAIRAliasAnnotateLegacyPass());
    // Keep blocks with SIMD-group calls under divergent control flow from
    // being if-converted by Metal's compiler (they would run with every
    // thread active). Late, after every pass that creates or moves them.
    addPass(createAIRSimdDivergenceGuardLegacyPass());
    // Final pre-serialization normalizations.
    addPass(createAIRPrepareLegacyPass());
    // (AGX-1 cross-buffer store separation now runs early, after LowerSwitch.)
    // AIRPrepare's mergeByteGlobals now emits the identity bitcast
    // inline on bfloat/half/float-through-bfloat typed-base GEPs, so the
    // post-Prepare NormalizeAllocas re-run is no longer needed.
    // See test_scan2d[cum{sum,prod}-bfloat16-*].
  }
};
} // namespace

AIRTargetMachine::AIRTargetMachine(const Target &T, const Triple &TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       std::optional<Reloc::Model> RM,
                                       std::optional<CodeModel::Model> CM,
                                       CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, TT, CPU, FS, Options,
                               Reloc::Static, CodeModel::Small, OL),
      TLOF(std::make_unique<AIRTargetObjectFile>()),
      Subtarget(std::make_unique<AIRSubtarget>(TT, CPU, FS, *this)) {
  initAsmInfo();
}

AIRTargetMachine::~AIRTargetMachine() {}

static cl::opt<bool> AIRPortable(
    "air-portable",
    cl::desc("Run only the passes shared by every AIR consumer "
             "(addAIRPortablePasses) and print the module; with "
             "-filetype=asm only"),
    cl::init(false));

// The passes of the full pipeline (AIRPassConfig::addCodeGenPrepare) that
// define AIR rather than work around the Metal driver, in the same order.
// Scalar arguments become constant buffers, AIR's own form, where the full
// pipeline packs them into _scalar_buf for the Triton host. AIR has no double
// and no floating point environment: doubles and constrained intrinsics are
// emulated first, before the passes that write AIR's functions (the soft-f64
// library's intrinsics too). A module without them is left as it is (the
// AIR agx-raise-to-air writes).
void llvm::addAIRPortablePasses(legacy::PassManagerBase &PM) {
  PM.add(createAIRFromGenericGPULegacyPass());
  PM.add(createAIREmulateF64LegacyPass());
  PM.add(createAIRNaNMinMaxLegacyPass());
  PM.add(createLLVMToAIRIntrinsicsLegacyPass());
  PM.add(createAIRBarrierRenameLegacyPass());
  PM.add(createAIRLowerAtomicRMWLegacyPass(/*OrderingHints=*/true));
  PM.add(createAIRScalarArgsToConstantBuffersLegacyPass());
  PM.add(createAIRSystemValuesLegacyPass());
}

namespace {
/// addAIRPortablePasses for the new pass manager ("air-portable"), for
/// consumers that run AIR's portable passes by name (MLIR's AIR target for
/// AGX output, opt).
struct AIRPortablePass : RequiredPassInfoMixin<AIRPortablePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    legacy::PassManager PM;
    addAIRPortablePasses(PM);
    return PM.run(M) ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};
} // namespace

void AIRTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB){
#define GET_PASS_REGISTRY "AIRPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
  PB.registerAnalysisRegistrationCallback([](FunctionAnalysisManager &FAM) {
    FAM.registerPass([] { return AIRAA(); });
  });
  PB.registerParseAACallback([](StringRef Name, AAManager &AAM) {
    if (Name != "air-aa")
      return false;
    AAM.registerFunctionAnalysis<AIRAA>();
    return true;
  });
}

void AIRTargetMachine::registerEarlyDefaultAliasAnalyses(AAManager &AAM) {
  AAM.registerFunctionAnalysis<AIRAA>();
}

bool AIRTargetMachine::addPassesToEmitFile(
    PassManagerBase &PM, raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
    CodeGenFileType FileType, bool DisableVerify,
    MachineModuleInfoWrapperPass *MMIWP) {
  if (AIRPortable) {
    if (FileType != CodeGenFileType::AssemblyFile)
      report_fatal_error("-air-portable prints the module; use -filetype=asm");
    addAIRPortablePasses(PM);
    PM.add(createPrintModulePass(Out, "", true));
    return false;
  }

  TargetPassConfig *PassConfig = createPassConfig(PM);
  // Standard llc IR prologue (verifier, LSR + codegen-prep IR passes) at
  // -O1+, same as every upstream backend; -O0 / -disable-lsr opt out.
  PassConfig->addIRPasses();
  PassConfig->addCodeGenPrepare();

  switch (FileType) {
  case CodeGenFileType::AssemblyFile:
    // Emit the transformed LLVM IR. Object-file emission below produces the
    // .metallib via the MC ObjectWriter and embedder pass.
    PM.add(createPrintModulePass(Out, "", true));
    break;
  case CodeGenFileType::ObjectFile:
    if (PassConfig->willCompleteCodeGenPipeline()) {
      PM.add(createAIREmbedderPass());
      if (!MMIWP)
        MMIWP = new MachineModuleInfoWrapperPass(this);
      PM.add(MMIWP);
      if (addAsmPrinter(PM, Out, DwoOut, FileType,
                        MMIWP->getMMI().getContext()))
        return true;
    } else {
      // Raw-bitcode fallback (mirrors DXIL keeping createDXILWriterPass).
      PM.add(createAIRWriterPass(Out));
    }
    break;
  case CodeGenFileType::Null:
    break;
  }
  return false;
}

bool AIRTargetMachine::addPassesToEmitMC(PassManagerBase &, MCContext *&,
                                           raw_pwrite_stream &, bool) {
  return true;
}

TargetPassConfig *AIRTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new AIRPassConfig(*this, PM);
}

const AIRSubtarget *
AIRTargetMachine::getSubtargetImpl(const Function &) const {
  return Subtarget.get();
}

TargetTransformInfo
AIRTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(std::make_unique<AIRTTIImpl>(this, F));
}

AIRTargetLowering::AIRTargetLowering(const AIRTargetMachine &TM,
                                         const AIRSubtarget &STI)
    : TargetLowering(TM, STI) {
  // AIR has no instruction selection, but the cost model the optimizations
  // of portable AIR use asks which types and operations the target has: a
  // GPU's 16- and 32-bit integers and floats, with their minimum, maximum,
  // absolute value and bit counts (otherwise each costs a legalization, and
  // simplifycfg keeps branches a GPU would rather select). 64-bit integers
  // too, as register pairs (comparisons and selects of addresses are
  // everywhere).
  const TargetRegisterClass *RC = STI.getRegisterInfo()->getRegClass(0);
  for (MVT VT : {MVT::i1, MVT::i16, MVT::i32, MVT::i64, MVT::f16, MVT::f32})
    addRegisterClass(VT, RC);
  computeRegisterProperties(STI.getRegisterInfo());
  for (MVT VT : {MVT::i16, MVT::i32})
    setOperationAction({ISD::SMIN, ISD::SMAX, ISD::UMIN, ISD::UMAX, ISD::ABS,
                        ISD::CTLZ, ISD::CTTZ, ISD::CTPOP, ISD::BITREVERSE},
                       VT, Legal);
  for (MVT VT : {MVT::f16, MVT::f32})
    setOperationAction({ISD::FMINNUM, ISD::FMAXNUM, ISD::FMINIMUM,
                        ISD::FMAXIMUM, ISD::FMA, ISD::FABS, ISD::FCOPYSIGN},
                       VT, Legal);
}

bool AIRTargetLowering::isLegalAddressingMode(const DataLayout &DL,
                                              const AddrMode &AM, Type *Ty,
                                              unsigned AS, Instruction *I) const {
  if (!APInt(64, AM.BaseOffs).isSignedIntN(32))
    return false;

  if (AM.BaseGV)
    return !AM.BaseOffs && !AM.HasBaseReg && !AM.Scale;

  switch (AM.Scale) {
  case 0:
    break;
  case 1:
    break;
  default:
    return false;
  }
  return true;
}

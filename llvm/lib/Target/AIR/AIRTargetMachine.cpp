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
#include "AIRDeviceLoadsVolatile.h"
#include "AIRInlineNonKernel.h"
#include "LLVMToAIRIntrinsics.h"
#include "AIRLowerAtomicRMW.h"
#include "AIRLowerFNeg.h"
#include "AIRNaNMinMax.h"
#include "AIRNormalizeAllocas.h"
#include "AIRPrepare.h"
#include "AIRScalarBufferPacking.h"
#include "AIRScalarizeShuffleOperands.h"
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
  initializeAIRLowerFNegLegacyPass(*PR);
  initializeAIRNaNMinMaxLegacyPass(*PR);
  initializeLLVMToAIRIntrinsicsLegacyPass(*PR);
  initializeAIRBarrierRenameLegacyPass(*PR);
  initializeAIRLowerAtomicRMWLegacyPass(*PR);
  initializeAIRSplitI64ShuffleLegacyPass(*PR);
  initializeAIRScalarStoreGuardLegacyPass(*PR);
  initializeAIRScalarizeShuffleOperandsLegacyPass(*PR);
  initializeAIRTGGlobalCoalesceLegacyPass(*PR);
  initializeAIRTGBarrierInsertLegacyPass(*PR);
  initializeAIRDeviceLoadsVolatileLegacyPass(*PR);
  initializeAIRAsyncEventToAllocaLegacyPass(*PR);
  initializeAIRNormalizeAllocasLegacyPass(*PR);
  initializeAIRBFloat16CastDecomposeLegacyPass(*PR);
  initializeAIRScalarBufferPackingLegacyPass(*PR);
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
    // The Apple AGX GPU JIT miscompiles cross-lane `air.simd_shuffle*` when the
    // shuffle's scalar operand is sourced via `extractelement` from a vector
    // SSA value (a vector register): the permute reads the wrong physical lane
    // for some SIMD threads, corrupting cross-lane reductions. Apple's own
    // `metal` frontend never feeds vector-extracted values into shuffles. The
    // SLP vectorizer (O1+) creates exactly this pattern in reduce/scan kernels,
    // so scalarize the vector chains entangled with shuffle operands back to
    // scalars before AIR emission. GEMM's pure load/store vectors are
    // untouched.
    addPass(createAIRScalarizeShuffleOperandsLegacyPass());
    addPass(createAIRInlineNonKernelLegacyPass());
    addPass(createAIRDemoteF64LegacyPass());
    addPass(createAIRLowerFNegLegacyPass());
    addPass(createAIRNaNMinMaxLegacyPass());
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
    : CodeGenTargetMachineImpl(T, TT.computeDataLayout(), TT, CPU, FS, Options,
                               Reloc::Static, CodeModel::Small, OL),
      TLOF(std::make_unique<AIRTargetObjectFile>()),
      Subtarget(std::make_unique<AIRSubtarget>(TT, CPU, FS, *this)) {
  initAsmInfo();
}

AIRTargetMachine::~AIRTargetMachine() {}

void AIRTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB){
#define GET_PASS_REGISTRY "AIRPassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"
}

bool AIRTargetMachine::addPassesToEmitFile(
    PassManagerBase &PM, raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
    CodeGenFileType FileType, bool DisableVerify,
    MachineModuleInfoWrapperPass *MMIWP) {
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
    if (TargetPassConfig::willCompleteCodeGenPipeline()) {
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
    : TargetLowering(TM, STI) {}

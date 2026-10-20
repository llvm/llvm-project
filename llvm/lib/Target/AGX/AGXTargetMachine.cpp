//===-- AGXTargetMachine.cpp - Define TargetMachine for AGX ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The AGX code generation pipeline. Like AMDGPU, all calls are inlined and
// the control flow graph is structurized before instruction selection.
// Unlike AMDGPU, AGX has no scalar unit and no scalar branches: every
// conditional branch, uniform or not, is lowered to execution mask
// operations on the per-thread nesting counter r0l.
//
//===----------------------------------------------------------------------===//

#include "AGXTargetMachine.h"
#include "AGX.h"
#include "AGXMachineFunctionInfo.h"
#include "AGXTargetTransformInfo.h"
#include "TargetInfo/AGXTargetInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/Scalarizer.h"
#include "llvm/Transforms/Utils.h"

using namespace llvm;

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeAGXTarget() {
  RegisterTargetMachine<AGXTargetMachine> X(getTheAGXTarget());
  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeAGXAlwaysInlinePass(PR);
  initializeAGXPromoteKernelArgsPass(PR);
  initializeAGXCodeGenPreparePass(PR);
  initializeAGXStrictFPPass(PR);
  initializeAGXLowerFPEnvPass(PR);
  initializeAGXSoftF64Pass(PR);
  initializeAGXExpandWideAtomicsPass(PR);
  initializeAGXAnnotateControlFlowPass(PR);
  initializeAGXLowerGenericAddressesPass(PR);
  initializeAGXDAGToDAGISelLegacyPass(PR);
  initializeAGXFoldOperandsPass(PR);
  initializeAGXHoistFromElseFlowPass(PR);
  initializeAGXLowerControlFlowPass(PR);
  initializeAGXInsertWaitsPass(PR);
  initializeAGXAsmPrinterPass(PR);
}

static Reloc::Model getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::PIC_);
}

AGXTargetMachine::AGXTargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   std::optional<Reloc::Model> RM,
                                   std::optional<CodeModel::Model> CM,
                                   CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, TT, CPU.empty() ? "generic" : CPU, FS,
                               Options,
                               getEffectiveRelocModel(RM),
                               getEffectiveCodeModel(CM, CodeModel::Small),
                               OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()) {
  setRequiresStructuredCFG(true);
  initAsmInfo();
}

AGXTargetMachine::~AGXTargetMachine() = default;

const AGXSubtarget *
AGXTargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");
  std::string CPU =
      CPUAttr.isValid() ? CPUAttr.getValueAsString().str() : TargetCPU;
  std::string FS =
      FSAttr.isValid() ? FSAttr.getValueAsString().str() : TargetFS;
  // The module flag "agx-mesa-isa" selects Mesa's instruction forms for every
  // function of the module.
  if (const auto *Flag = mdconst::extract_or_null<ConstantInt>(
          F.getParent()->getModuleFlag("agx-mesa-isa")))
    if (!Flag->isZero())
      FS = FS.empty() ? "+mesa-isa" : FS + ",+mesa-isa";

  auto &I = SubtargetMap[CPU + FS];
  if (!I)
    I = std::make_unique<AGXSubtarget>(TargetTriple, CPU, FS, *this);
  return I.get();
}

TargetTransformInfo
AGXTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(std::make_unique<AGXTTIImpl>(this, F));
}

MachineFunctionInfo *AGXTargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return AGXMachineFunctionInfo::create<AGXMachineFunctionInfo>(Allocator, F,
                                                                STI);
}

bool AGXTargetMachine::isNoopAddrSpaceCast(const DataLayout &DL, unsigned SrcAS,
                                           unsigned DestAS) const {
  // Generic, device and constant pointers all address device memory.
  auto IsDevice = [](unsigned AS) {
    return AS == AGX::GENERIC_ADDRESS || AS == AGX::DEVICE_ADDRESS ||
           AS == AGX::CONSTANT_ADDRESS;
  };
  return IsDevice(SrcAS) && IsDevice(DestAS);
}

namespace {

class AGXPassConfig : public TargetPassConfig {
public:
  AGXPassConfig(AGXTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {
    // There are no calls, no stack unwinding and no shrink wrapping.
    disablePass(&StackMapLivenessID);
    disablePass(&FuncletLayoutID);
    disablePass(&PatchableFunctionID);
    disablePass(&ShrinkWrapID);
    disablePass(&PostRAMachineSinkingID);
    // Keep the block order of the structurizer: execution mask code falls
    // through from each construct into the next, as the vendor and Mesa
    // compilers lay it out.
    disablePass(&MachineBlockPlacementID);
  }

  AGXTargetMachine &getAGXTargetMachine() const {
    return getTM<AGXTargetMachine>();
  }

  void addIRPasses() override;
  bool addPreISel() override;
  bool addInstSelector() override;
  void addMachineSSAOptimization() override;
  void addPreEmitPass() override;
};

} // namespace

TargetPassConfig *AGXTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new AGXPassConfig(*this, PM);
}

void AGXPassConfig::addIRPasses() {
  // There are no function calls: inline everything into the kernels.
  addPass(createAGXAlwaysInlinePass());
  addPass(createAlwaysInlinerLegacyPass());

  // 64- and 128-bit atomics take locks; the others are hardware atomics.
  addPass(createAGXExpandWideAtomicsPass(&getAGXTargetMachine()));
  addPass(createAtomicExpandLegacyPass());
  if (getOptLevel() != CodeGenOptLevel::None) {
    addPass(createInferAddressSpacesPass());
    addPass(createSROAPass());
  }
  // Generic pointers that may name threadgroup memory become tagged integers.
  addPass(createAGXLowerGenericAddressesPass());
  // By-reference kernel arguments become uniform register values.
  addPass(createAGXPromoteKernelArgsPass());
  // Vector arithmetic is scalar code (as in instruction selection), which the
  // software floating point lowerings below then see.
  addPass(createScalarizerPass());
  // The floating point mode register and constrained floating point
  // intrinsics (directed rounding in software).
  addPass(createAGXLowerFPEnvPass());
  addPass(createAGXSoftF64Pass());
  addPass(createAGXCodeGenPreparePass());
  // Software IEEE binary32 arithmetic and canonical NaNs, when enabled.
  addPass(createAGXStrictFPPass());

  TargetPassConfig::addIRPasses();
}

bool AGXPassConfig::addPreISel() {
  // Bring the CFG into the structured form AGXAnnotateControlFlow expects.
  addPass(createLowerSwitchPass());
  addPass(createFixIrreduciblePass());
  addPass(createUnifyLoopExitsPass());
  // Every branch is lowered to execution mask operations, so uniform
  // regions are structurized too.
  addPass(createStructurizeCFGPass(/*SkipUniformRegions=*/false));
  addPass(createAGXAnnotateControlFlowPass());
  addPass(createLCSSAPass());
  return false;
}

bool AGXPassConfig::addInstSelector() {
  addPass(createAGXISelDag(getAGXTargetMachine()));
  return false;
}

void AGXPassConfig::addMachineSSAOptimization() {
  TargetPassConfig::addMachineSSAOptimization();
  addPass(createAGXFoldOperandsPass());
  // MachineCSE and MachineSink may have put code where only the threads of
  // a then side run it.
  addPass(createAGXHoistFromElseFlowPass());
}

void AGXPassConfig::addPreEmitPass() {
  addPass(createAGXLowerControlFlowPass());
  addPass(createAGXInsertWaitsPass());
}

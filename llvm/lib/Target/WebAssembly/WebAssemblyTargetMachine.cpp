//===- WebAssemblyTargetMachine.cpp - Define TargetMachine for WebAssembly -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file defines the WebAssembly-specific subclass of TargetMachine.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyTargetMachine.h"
#include "WebAssemblyTargetObjectFile.h"
#include "WebAssemblyTargetTransformInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

#define DEBUG_TYPE "wasm"

extern "C" void LLVMInitializeWebAssemblyTarget() {
  // Register the target.
  RegisterTargetMachine<WebAssemblyTargetMachine> X(TheWebAssemblyTarget32);
  RegisterTargetMachine<WebAssemblyTargetMachine> Y(TheWebAssemblyTarget64);
}

//===----------------------------------------------------------------------===//
// WebAssembly Lowering public interface.
//===----------------------------------------------------------------------===//

/// Create an WebAssembly architecture model.
///
WebAssemblyTargetMachine::WebAssemblyTargetMachine(
    const Target &T, const Triple &TT, StringRef CPU, StringRef FS,
    const TargetOptions &Options, Reloc::Model RM, CodeModel::Model CM,
    CodeGenOpt::Level OL)
    : LLVMTargetMachine(T, TT.isArch64Bit() ? "e-p:64:64-i64:64-n32:64-S128"
                                            : "e-p:32:32-i64:64-n32:64-S128",
                        TT, CPU, FS, Options, RM, CM, OL),
      TLOF(make_unique<WebAssemblyTargetObjectFile>()) {
  // WebAssembly type-checks expressions, but a noreturn function with a return
  // type that doesn't match the context will cause a check failure. So we lower
  // LLVM 'unreachable' to ISD::TRAP and then lower that to WebAssembly's
  // 'unreachable' expression which is meant for that case.
  this->Options.TrapUnreachable = true;

  initAsmInfo();

  // We need a reducible CFG, so disable some optimizations which tend to
  // introduce irreducibility.
  setRequiresStructuredCFG(true);
}

WebAssemblyTargetMachine::~WebAssemblyTargetMachine() {}

const WebAssemblySubtarget *
WebAssemblyTargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute FSAttr = F.getFnAttribute("target-features");

  std::string CPU = !CPUAttr.hasAttribute(Attribute::None)
                        ? CPUAttr.getValueAsString().str()
                        : TargetCPU;
  std::string FS = !FSAttr.hasAttribute(Attribute::None)
                       ? FSAttr.getValueAsString().str()
                       : TargetFS;

  auto &I = SubtargetMap[CPU + FS];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    I = llvm::make_unique<WebAssemblySubtarget>(TargetTriple, CPU, FS, *this);
  }
  return I.get();
}

namespace {
/// WebAssembly Code Generator Pass Configuration Options.
class WebAssemblyPassConfig final : public TargetPassConfig {
public:
  WebAssemblyPassConfig(WebAssemblyTargetMachine *TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  WebAssemblyTargetMachine &getWebAssemblyTargetMachine() const {
    return getTM<WebAssemblyTargetMachine>();
  }

  FunctionPass *createTargetRegisterAllocator(bool) override;

  void addIRPasses() override;
  bool addInstSelector() override;
  bool addILPOpts() override;
  void addPreRegAlloc() override;
  void addPostRegAlloc() override;
  void addPreEmitPass() override;
};
} // end anonymous namespace

TargetIRAnalysis WebAssemblyTargetMachine::getTargetIRAnalysis() {
  return TargetIRAnalysis([this](const Function &F) {
    return TargetTransformInfo(WebAssemblyTTIImpl(this, F));
  });
}

TargetPassConfig *
WebAssemblyTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new WebAssemblyPassConfig(this, PM);
}

FunctionPass *WebAssemblyPassConfig::createTargetRegisterAllocator(bool) {
  return nullptr; // No reg alloc
}

//===----------------------------------------------------------------------===//
// The following functions are called from lib/CodeGen/Passes.cpp to modify
// the CodeGen pass sequence.
//===----------------------------------------------------------------------===//

void WebAssemblyPassConfig::addIRPasses() {
  if (TM->Options.ThreadModel == ThreadModel::Single)
    // In "single" mode, atomics get lowered to non-atomics.
    addPass(createLowerAtomicPass());
  else
    // Expand some atomic operations. WebAssemblyTargetLowering has hooks which
    // control specifically what gets lowered.
    addPass(createAtomicExpandPass(TM));

  // Optimize "returned" function attributes.
  addPass(createWebAssemblyOptimizeReturned());

  TargetPassConfig::addIRPasses();
}

bool WebAssemblyPassConfig::addInstSelector() {
  addPass(
      createWebAssemblyISelDag(getWebAssemblyTargetMachine(), getOptLevel()));
  return false;
}

bool WebAssemblyPassConfig::addILPOpts() { return true; }

void WebAssemblyPassConfig::addPreRegAlloc() {
  // Prepare store instructions for register stackifying.
  addPass(createWebAssemblyStoreResults());

  // Mark registers as representing wasm's expression stack.
  addPass(createWebAssemblyRegStackify());
}

void WebAssemblyPassConfig::addPostRegAlloc() {
  // TODO: The following CodeGen passes don't currently support code containing
  // virtual registers. Consider removing their restrictions and re-enabling
  // them.
  //
  // Fails with: Regalloc must assign all vregs.
  disablePass(&PrologEpilogCodeInserterID);
  // Fails with: should be run after register allocation.
  disablePass(&MachineCopyPropagationID);

  // Run the register coloring pass to reduce the total number of registers.
  addPass(createWebAssemblyRegColoring());
}

void WebAssemblyPassConfig::addPreEmitPass() {
  // Put the CFG in structured form; insert BLOCK and LOOP markers.
  addPass(createWebAssemblyCFGStackify());

  // Lower br_unless into br_if.
  addPass(createWebAssemblyLowerBrUnless());

  // Create a mapping from LLVM CodeGen virtual registers to wasm registers.
  addPass(createWebAssemblyRegNumbering());

  // Perform the very last peephole optimizations on the code.
  addPass(createWebAssemblyPeephole());
}

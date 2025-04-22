//===-- X86CodeGenPassBuilder.cpp ---------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains X86 CodeGen pipeline builder.
/// TODO: Port CodeGen passes to new pass manager.
//===----------------------------------------------------------------------===//

#include "X86ISelDAGToDAG.h"
#include "X86TargetMachine.h"

#include "llvm/MC/MCStreamer.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"

using namespace llvm;

namespace {

class X86CodeGenPassBuilder
    : public CodeGenPassBuilder<X86CodeGenPassBuilder, X86TargetMachine> {
public:
  explicit X86CodeGenPassBuilder(X86TargetMachine &TM,
                                 const CGPassBuilderOption &Opts,
                                 PassInstrumentationCallbacks *PIC,
                                 PassBuilder &PB)
      : CodeGenPassBuilder(TM, Opts, PIC, PB) {}
  using Base = CodeGenPassBuilder<X86CodeGenPassBuilder, X86TargetMachine>;
  void addPreISel(AddIRPass &addPass) const;
  void addAsmPrinter(AddMachinePass &, CreateMCStreamer) const;
  Error addInstSelector(AddMachinePass &) const;
  Error addRegAssignmentOptimized(AddMachinePass &addPass) const;
};

Error X86CodeGenPassBuilder::addRegAssignmentOptimized(
    AddMachinePass &addPass) const {
  if (EnableTileRAPass) {
    addRegAllocPassOrOpt(addPass, []() {
      return RAGreedyPass({onlyAllocateTileRegisters, "tile-reg"});
    });
    // TODO: addPass(X86TileConfigPass());
  }
  return Base::addRegAssignmentOptimized(addPass);
}

void X86CodeGenPassBuilder::addPreISel(AddIRPass &addPass) const {
  // TODO: Add passes pre instruction selection.
}

void X86CodeGenPassBuilder::addAsmPrinter(AddMachinePass &addPass,
                                          CreateMCStreamer) const {
  // TODO: Add AsmPrinter.
}

Error X86CodeGenPassBuilder::addInstSelector(AddMachinePass &addPass) const {
  // TODO: Add instruction selector related passes.
  addPass(X86ISelDAGToDAGPass(TM));
  return Error::success();
}

} // namespace

void X86TargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
#define GET_PASS_REGISTRY "X86PassRegistry.def"
#include "llvm/Passes/TargetPassRegistry.inc"

  PB.registerRegClassFilterParsingCallback(
      [](StringRef FilterName) -> RegAllocFilterFunc {
        if (FilterName == "tile-reg") {
          return onlyAllocateTileRegisters;
        }
        return nullptr;
      });
}

Error X86TargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
    CodeGenFileType FileType, const CGPassBuilderOption &Opt,
    PassInstrumentationCallbacks *PIC, PassBuilder &PB) {
  auto CGPB = X86CodeGenPassBuilder(*this, Opt, PIC, PB);
  return CGPB.buildPipeline(MPM, Out, DwoOut, FileType);
}

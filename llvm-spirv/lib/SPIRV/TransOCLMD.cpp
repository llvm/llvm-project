//===- TransOCLMD.cpp - Transform OCL metadata to SPIR-V metadata - C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file implements translation of OCL metadata to SPIR-V metadata.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "clmdtospv"

#include "OCLUtil.h"
#include "SPIRVInternal.h"
#include "SPIRVMDBuilder.h"
#include "SPIRVMDWalker.h"

#include "llvm/ADT/Triple.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {

cl::opt<bool> EraseOCLMD("spirv-erase-cl-md", cl::init(true),
                         cl::desc("Erase OpenCL metadata"));

class TransOCLMD : public ModulePass {
public:
  TransOCLMD() : ModulePass(ID), M(nullptr), Ctx(nullptr), CLVer(0) {
    initializeTransOCLMDPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;
  void visit(Module *M);

  static char ID;

private:
  Module *M;
  LLVMContext *Ctx;
  unsigned CLVer; /// OpenCL version as major*10+minor
};

char TransOCLMD::ID = 0;

bool TransOCLMD::runOnModule(Module &Module) {
  M = &Module;
  Ctx = &M->getContext();
  CLVer = getOCLVersion(M, true);
  if (CLVer == 0)
    return false;

  LLVM_DEBUG(dbgs() << "Enter TransOCLMD:\n");
  visit(M);

  LLVM_DEBUG(dbgs() << "After TransOCLMD:\n" << *M);
  std::string Err;
  raw_string_ostream ErrorOS(Err);
  if (verifyModule(*M, &ErrorOS)) {
    LLVM_DEBUG(errs() << "Fails to verify module: " << ErrorOS.str());
  }
  return true;
}

void TransOCLMD::visit(Module *M) {
  SPIRVMDBuilder B(*M);
  SPIRVMDWalker W(*M);
  // !spirv.Source = !{!x}
  // !{x} = !{i32 3, i32 102000}
  B.addNamedMD(kSPIRVMD::Source)
      .addOp()
      .add(CLVer < kOCLVer::CL21 ? spv::SourceLanguageOpenCL_C
                                 : spv::SourceLanguageOpenCL_CPP)
      .add(CLVer)
      .done();
  if (EraseOCLMD)
    B.eraseNamedMD(kSPIR2MD::OCLVer).eraseNamedMD(kSPIR2MD::SPIRVer);

  // !spirv.MemoryModel = !{!x}
  // !{x} = !{i32 1, i32 2}
  Triple TT(M->getTargetTriple());
  auto Arch = TT.getArch();
  assert((Arch == Triple::spir || Arch == Triple::spir64) && "Invalid triple");
  B.addNamedMD(kSPIRVMD::MemoryModel)
      .addOp()
      .add(Arch == Triple::spir ? spv::AddressingModelPhysical32
                                : spv::AddressingModelPhysical64)
      .add(spv::MemoryModelOpenCL)
      .done();

  // Add source extensions
  // !spirv.SourceExtension = !{!x, !y, ...}
  // !x = {!"cl_khr_..."}
  // !y = {!"cl_khr_..."}
  auto Exts = getNamedMDAsStringSet(M, kSPIR2MD::Extensions);
  if (!Exts.empty()) {
    auto N = B.addNamedMD(kSPIRVMD::SourceExtension);
    for (auto &I : Exts)
      N.addOp().add(I).done();
  }
  if (EraseOCLMD)
    B.eraseNamedMD(kSPIR2MD::Extensions).eraseNamedMD(kSPIR2MD::OptFeatures);

  if (EraseOCLMD)
    B.eraseNamedMD(kSPIR2MD::FPContract);

  // Create metadata representing (empty so far) list
  // of OpEntryPoint and OpExecutionMode instructions
  auto EP = B.addNamedMD(kSPIRVMD::EntryPoint);    // !spirv.EntryPoint = {}
  auto EM = B.addNamedMD(kSPIRVMD::ExecutionMode); // !spirv.ExecutionMode = {}

  // Add execution modes for kernels. We take it from metadata attached to
  // the kernel functions.
  for (Function &Kernel : *M) {
    if (Kernel.getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;

    // Add EntryPoint(which actually is adding its operands) to the list of
    // entry points:
    // !{i32 6, void (i32 addrspace(1)*)* @kernel, !"kernel" }
    MDNode *EPNode;
    EP.addOp()
        .add(spv::ExecutionModelKernel)
        .add(&Kernel)
        .add(Kernel.getName())
        .done(&EPNode);

    // Specifing execution modes for the Kernel and adding it to the list
    // of ExecutionMode instructions.

    // !{void (i32 addrspace(1)*)* @kernel, i32 17, i32 X, i32 Y, i32 Z}
    if (MDNode *WGSize = Kernel.getMetadata(kSPIR2MD::WGSize)) {
      unsigned X, Y, Z;
      decodeMDNode(WGSize, X, Y, Z);
      EM.addOp()
          .add(&Kernel)
          .add(spv::ExecutionModeLocalSize)
          .add(X)
          .add(Y)
          .add(Z)
          .done();
    }

    // !{void (i32 addrspace(1)*)* @kernel, i32 18, i32 X, i32 Y, i32 Z}
    if (MDNode *WGSizeHint = Kernel.getMetadata(kSPIR2MD::WGSizeHint)) {
      unsigned X, Y, Z;
      decodeMDNode(WGSizeHint, X, Y, Z);
      EM.addOp()
          .add(&Kernel)
          .add(spv::ExecutionModeLocalSizeHint)
          .add(X)
          .add(Y)
          .add(Z)
          .done();
    }

    // !{void (i32 addrspace(1)*)* @kernel, i32 30, i32 hint}
    if (MDNode *VecTypeHint = Kernel.getMetadata(kSPIR2MD::VecTyHint)) {
      EM.addOp()
          .add(&Kernel)
          .add(spv::ExecutionModeVecTypeHint)
          .add(transVecTypeHint(VecTypeHint))
          .done();
    }

    // !{void (i32 addrspace(1)*)* @kernel, i32 35, i32 size}
    if (MDNode *ReqdSubgroupSize = Kernel.getMetadata(kSPIR2MD::SubgroupSize)) {
      EM.addOp()
          .add(&Kernel)
          .add(spv::ExecutionModeSubgroupSize)
          .add(getMDOperandAsInt(ReqdSubgroupSize, 0))
          .done();
    }
  }
}

} // namespace SPIRV

INITIALIZE_PASS(TransOCLMD, "clmdtospv",
                "Transform OCL metadata format to SPIR-V metadata format",
                false, false)

ModulePass *llvm::createTransOCLMD() { return new TransOCLMD(); }

//===- SPIRVLowerOCLBlocks.cpp - OCL Utilities ----------------------------===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2018 Intel Corporation. All rights reserved.
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
// Neither the names of Intel Corporation, nor the names of its
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
// SPIR-V specification doesn't allow function pointers, so SPIR-V translator
// is designed to fail if a value with function type (except calls) is occured.
// Currently there is only two cases, when function pointers are generating in
// LLVM IR in OpenCL - block calls and device side enqueue built-in calls.
//
// In both cases values with function type used as intermediate representation
// for block literal structure.
//
// In LLVM IR produced by clang, blocks are represented with the following
// structure:
// %struct.__opencl_block_literal_generic = type { i32, i32, i8 addrspace(4)* }
// Pointers to block invoke functions are stored in the third field. Clang
// replaces inderect function calls in all cases except if block is passed as a
// function argument. Note that it is somewhat unclear if the OpenCL C spec
// should allow passing blocks as function argumernts. This pass is not supposed
// to work correctly with such functions.
// Clang though has to store function pointers to this structure. Purpose of
// this pass is to replace store of function pointers(not allowed in SPIR-V)
// with null pointers.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "spv-lower-ocl-blocks"

#include "SPIRVInternal.h"

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Regex.h"

using namespace llvm;

namespace {

static bool isBlockInvoke(Function &F) {
  static Regex BlockInvokeRegex("_block_invoke_?[0-9]*$");
  return BlockInvokeRegex.match(F.getName());
}

class SPIRVLowerOCLBlocks : public ModulePass {

public:
  SPIRVLowerOCLBlocks() : ModulePass(ID) {}

  bool runOnModule(Module &M) {
    bool Changed = false;
    for (Function &F : M) {
      if (!isBlockInvoke(F))
        continue;
      for (User *U : F.users()) {
        if (!isa<Constant>(U))
          continue;
        Constant *Null = Constant::getNullValue(U->getType());
        if (U != Null) {
          U->replaceAllUsesWith(Null);
          Changed = true;
        }
      }
    }
    return Changed;
  }

  static char ID;
};

char SPIRVLowerOCLBlocks::ID = 0;

} // namespace

INITIALIZE_PASS(
    SPIRVLowerOCLBlocks, "spv-lower-ocl-blocks",
    "Remove function pointers occured in case of using OpenCL blocks", false,
    false)

llvm::ModulePass *llvm::createSPIRVLowerOCLBlocks() {
  return new SPIRVLowerOCLBlocks();
}

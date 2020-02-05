//===- ReduceAttributes.cpp - Specialized Delta Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce Attributes in the provided Module.
//
//===----------------------------------------------------------------------===//

#include "ReduceAttributes.h"

#include "Delta.h"
#include "TestRunner.h"

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include <vector>

static void removeAttr(llvm::Function &F, const llvm::Attribute &A) {
  if (A.isStringAttribute())
    F.removeFnAttr(A.getKindAsString());
  else
    F.removeFnAttr(A.getKindAsEnum());
}

static void extractAttributes(const std::vector<llvm::Chunk> &ChunksToKeep,
                              llvm::Module *M) {
  int AttributeIndex = 0;
  unsigned ChunkIndex = 0;
  // TODO: global variables may also have attributes.
  for (llvm::Function &F : M->getFunctionList()) {
    for (const llvm::Attribute &A : F.getAttributes().getFnAttributes()) {
      ++AttributeIndex;
      if (!ChunksToKeep[ChunkIndex].contains(AttributeIndex))
        removeAttr(F, A);
      if (AttributeIndex == ChunksToKeep[ChunkIndex].end)
        ++ChunkIndex;
    }
  }
}

static int countAttributes(llvm::Module *M) {
  int TotalAttributeCount = 0;
  for (const llvm::Function &F : M->getFunctionList())
    TotalAttributeCount +=
        F.getAttributes().getFnAttributes().getNumAttributes();
  // TODO: global variables may also have attributes.
  return TotalAttributeCount;
}

void llvm::reduceAttributesDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing Attributes...\n";
  int AttributeCount = countAttributes(Test.getProgram());
  runDeltaPass(Test, AttributeCount, extractAttributes);
}

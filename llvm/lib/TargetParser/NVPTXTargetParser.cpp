//===-- NVPTXTargetParser - Parser for NVPTX target ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser for the NVPTX (CUDA) GPU list.
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/NVPTXTargetParser.h"
#include "llvm/ADT/StringSwitch.h"

using namespace llvm;
using namespace NVPTX;

GPUKind llvm::NVPTX::parseArch(StringRef CPU) {
  return StringSwitch<GPUKind>(CPU)
#define NVPTX_GPU(NAME, KIND, VIRTUAL, SM_ID, MIN_VER, MAX_VER, SUFFIX)        \
  .Case(NAME, GK_##KIND)
#include "llvm/TargetParser/NVPTXTargetParser.def"
      .Default(GK_NONE);
}

StringRef llvm::NVPTX::getArchName(GPUKind Kind) {
  switch (Kind) {
  case GK_NONE:
    return "";
#define NVPTX_GPU(NAME, KIND, VIRTUAL, SM_ID, MIN_VER, MAX_VER, SUFFIX)        \
  case GK_##KIND:                                                              \
    return NAME;
#include "llvm/TargetParser/NVPTXTargetParser.def"
  }
  llvm_unreachable("invalid NVPTX GPUKind");
}

StringRef llvm::NVPTX::getVirtualArch(GPUKind Kind) {
  switch (Kind) {
  case GK_NONE:
    return "";
#define NVPTX_GPU(NAME, KIND, VIRTUAL, SM_ID, MIN_VER, MAX_VER, SUFFIX)        \
  case GK_##KIND:                                                              \
    return VIRTUAL;
#include "llvm/TargetParser/NVPTXTargetParser.def"
  }
  llvm_unreachable("invalid NVPTX GPUKind");
}

unsigned llvm::NVPTX::getSmVersion(GPUKind Kind) {
  switch (Kind) {
  case GK_NONE:
    return 0;
#define NVPTX_GPU(NAME, KIND, VIRTUAL, SM_ID, MIN_VER, MAX_VER, SUFFIX)        \
  case GK_##KIND:                                                              \
    return SM_ID;
#include "llvm/TargetParser/NVPTXTargetParser.def"
  }
  llvm_unreachable("invalid NVPTX GPUKind");
}

ArchSuffix llvm::NVPTX::getArchSuffix(GPUKind Kind) {
  switch (Kind) {
  case GK_NONE:
    return ArchSuffix::NONE;
#define NVPTX_GPU(NAME, KIND, VIRTUAL, SM_ID, MIN_VER, MAX_VER, SUFFIX)        \
  case GK_##KIND:                                                              \
    return ArchSuffix::SUFFIX;
#include "llvm/TargetParser/NVPTXTargetParser.def"
  }
  llvm_unreachable("invalid NVPTX GPUKind");
}

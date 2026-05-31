//===-- AIRContainerObjectWriter.h - AIR target object writer ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the AIRLib object target writer for the AIR backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_MCTARGETDESC_AIRCONTAINEROBJECTWRITER_H
#define LLVM_LIB_TARGET_AIR_MCTARGETDESC_AIRCONTAINEROBJECTWRITER_H

#include "llvm/MC/MCObjectWriter.h"

namespace llvm {

std::unique_ptr<MCObjectTargetWriter> createAIRContainerTargetObjectWriter();

}

#endif // LLVM_LIB_TARGET_AIR_MCTARGETDESC_AIRCONTAINEROBJECTWRITER_H

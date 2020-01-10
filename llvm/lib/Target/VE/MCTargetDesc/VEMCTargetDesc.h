//===-- VEMCTargetDesc.h - VE Target Descriptions ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides VE specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_VE_MCTARGETDESC_VEMCTARGETDESC_H
#define LLVM_LIB_TARGET_VE_MCTARGETDESC_VEMCTARGETDESC_H

#include "llvm/Support/DataTypes.h"

#include <memory>

namespace llvm {

class Target;
Target &getTheVETarget();

} // end llvm namespace

#endif

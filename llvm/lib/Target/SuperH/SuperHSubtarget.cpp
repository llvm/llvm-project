//===-- SuperHSubtarget.h - Define Subtarget for SuperH ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the SuperH specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "SuperHSubtarget.h"
#include "MCTargetDesc/SuperHMCTargetDesc.h"

using namespace llvm;

#define DEBUG_TYPE "sh-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "SuperHGenSubtargetInfo.inc"
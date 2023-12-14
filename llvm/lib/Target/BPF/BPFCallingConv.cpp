//=== BPFCallingConv.cpp ---- BPF Calling Convention Routines ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements BPF Calling Convention Routines
//
//===----------------------------------------------------------------------===//

#include "BPFCallingConv.h"
#include "BPFRegisterInfo.h"
#include "BPFSubtarget.h"

using namespace llvm;

#include "BPFGenCallingConv.inc"

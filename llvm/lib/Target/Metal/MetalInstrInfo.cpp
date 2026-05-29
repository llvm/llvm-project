//===-- MetalInstrInfo.cpp - InstrInfo for Metal -*- C++ --------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Metal specific subclass of TargetInstrInfo.
//
//===----------------------------------------------------------------------===//

#include "MetalInstrInfo.h"
#include "MetalSubtarget.h"

#define GET_INSTRINFO_CTOR_DTOR
#include "MetalGenInstrInfo.inc"

using namespace llvm;

MetalInstrInfo::MetalInstrInfo(const MetalSubtarget &STI)
    : MetalGenInstrInfo(STI, RI) {}

MetalInstrInfo::~MetalInstrInfo() {}

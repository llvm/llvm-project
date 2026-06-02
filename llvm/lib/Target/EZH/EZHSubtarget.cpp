//===- EZHSubtarget.cpp - EZH Subtarget Information -----------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the EZH specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "EZHSubtarget.h"
#include "MCTargetDesc/EZHMCTargetDesc.h"

#define DEBUG_TYPE "ezh-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "EZHGenSubtargetInfo.inc"

using namespace llvm;

void EZHSubtarget::initSubtargetFeatures(StringRef CPU, StringRef FS) {
  StringRef CPUName = CPU.empty() ? "generic" : CPU;
  ParseSubtargetFeatures(CPUName, /*TuneCPU*/ CPUName, FS);
}

EZHSubtarget &EZHSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                            StringRef FS) {
  initSubtargetFeatures(CPU, FS);
  return *this;
}

EZHSubtarget::EZHSubtarget(const Triple &TargetTriple, StringRef Cpu,
                           StringRef FeatureString, const TargetMachine &TM,
                           const TargetOptions & /*Options*/,
                           CodeModel::Model /*CodeModel*/,
                           CodeGenOptLevel /*OptLevel*/)
    : EZHGenSubtargetInfo(TargetTriple, Cpu, /*TuneCPU*/ Cpu, FeatureString),
      InstrInfo(initializeSubtargetDependencies(Cpu, FeatureString)),
      FrameLowering(*this), TLInfo(TM, *this) {}

void EZHSubtarget::initLibcallLoweringInfo(LibcallLoweringInfo &Info) const {
  Info.setLibcallImpl(RTLIB::SDIV_I32, RTLIB::impl___divsi3);
  Info.setLibcallImpl(RTLIB::UDIV_I32, RTLIB::impl___udivsi3);
  Info.setLibcallImpl(RTLIB::SREM_I32, RTLIB::impl___modsi3);
  Info.setLibcallImpl(RTLIB::UREM_I32, RTLIB::impl___umodsi3);
  Info.setLibcallImpl(RTLIB::MUL_I32, RTLIB::impl___mulsi3);
  Info.setLibcallImpl(RTLIB::SHL_I32, RTLIB::impl___ashlsi3);
  Info.setLibcallImpl(RTLIB::SRL_I32, RTLIB::impl___lshrsi3);
  Info.setLibcallImpl(RTLIB::SRA_I32, RTLIB::impl___ashrsi3);
  Info.setLibcallImpl(RTLIB::CTLZ_I32, RTLIB::impl___clzsi2);
  Info.setLibcallImpl(RTLIB::CTPOP_I32, RTLIB::impl___popcountsi2);

  Info.setLibcallImpl(RTLIB::SDIV_I64, RTLIB::impl___divdi3);
  Info.setLibcallImpl(RTLIB::UDIV_I64, RTLIB::impl___udivdi3);
  Info.setLibcallImpl(RTLIB::SREM_I64, RTLIB::impl___moddi3);
  Info.setLibcallImpl(RTLIB::UREM_I64, RTLIB::impl___umoddi3);
  Info.setLibcallImpl(RTLIB::MUL_I64, RTLIB::impl___muldi3);
  Info.setLibcallImpl(RTLIB::SHL_I64, RTLIB::impl___ashldi3);
  Info.setLibcallImpl(RTLIB::SRL_I64, RTLIB::impl___lshrdi3);
  Info.setLibcallImpl(RTLIB::SRA_I64, RTLIB::impl___ashrdi3);
  Info.setLibcallImpl(RTLIB::CTLZ_I64, RTLIB::impl___clzdi2);
}

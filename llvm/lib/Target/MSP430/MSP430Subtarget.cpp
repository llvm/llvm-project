//===-- MSP430Subtarget.cpp - MSP430 Subtarget Information ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MSP430 specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "MSP430Subtarget.h"
#include "MSP430SelectionDAGInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "msp430-subtarget"

static cl::opt<MSP430Subtarget::HWMultEnum>
HWMultModeOption("mhwmult", cl::Hidden,
           cl::desc("Hardware multiplier use mode for MSP430"),
           cl::init(MSP430Subtarget::NoHWMult),
           cl::values(
             clEnumValN(MSP430Subtarget::NoHWMult, "none",
                "Do not use hardware multiplier"),
             clEnumValN(MSP430Subtarget::HWMult16, "16bit",
                "Use 16-bit hardware multiplier"),
             clEnumValN(MSP430Subtarget::HWMult32, "32bit",
                "Use 32-bit hardware multiplier"),
             clEnumValN(MSP430Subtarget::HWMultF5, "f5series",
                "Use F5 series hardware multiplier")));

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "MSP430GenSubtargetInfo.inc"

void MSP430Subtarget::anchor() { }

MSP430Subtarget &
MSP430Subtarget::initializeSubtargetDependencies(StringRef CPU, StringRef FS) {
  ExtendedInsts = false;
  HWMultMode = NoHWMult;

  StringRef CPUName = CPU;
  if (CPUName.empty())
    CPUName = "msp430";

  ParseSubtargetFeatures(CPUName, /*TuneCPU*/ CPUName, FS);

  if (HWMultModeOption != NoHWMult)
    HWMultMode = HWMultModeOption;

  return *this;
}

MSP430Subtarget::MSP430Subtarget(const Triple &TT, const std::string &CPU,
                                 const std::string &FS, const TargetMachine &TM)
    : MSP430GenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)), TLInfo(TM, *this),
      FrameLowering(*this) {
  TSInfo = std::make_unique<MSP430SelectionDAGInfo>();
}

MSP430Subtarget::~MSP430Subtarget() = default;

const SelectionDAGTargetInfo *MSP430Subtarget::getSelectionDAGInfo() const {
  return TSInfo.get();
}

void MSP430Subtarget::initLibcallLoweringInfo(LibcallLoweringInfo &Info) const {
  if (hasHWMult16()) {
    const struct {
      const RTLIB::Libcall Op;
      const RTLIB::LibcallImpl Impl;
    } LibraryCalls[] = {
        // Integer Multiply - EABI Table 9
        {RTLIB::MUL_I16, RTLIB::impl___mspabi_mpyi_hw},
        {RTLIB::MUL_I32, RTLIB::impl___mspabi_mpyl_hw},
        {RTLIB::MUL_I64, RTLIB::impl___mspabi_mpyll_hw},
        // TODO The __mspabi_mpysl*_hw functions ARE implemented in libgcc
        // TODO The __mspabi_mpyul*_hw functions ARE implemented in libgcc
    };
    for (const auto &LC : LibraryCalls) {
      Info.setLibcallImpl(LC.Op, LC.Impl);
    }
  } else if (hasHWMult32()) {
    const struct {
      const RTLIB::Libcall Op;
      const RTLIB::LibcallImpl Impl;
    } LibraryCalls[] = {
        // Integer Multiply - EABI Table 9
        {RTLIB::MUL_I16, RTLIB::impl___mspabi_mpyi_hw},
        {RTLIB::MUL_I32, RTLIB::impl___mspabi_mpyl_hw32},
        {RTLIB::MUL_I64, RTLIB::impl___mspabi_mpyll_hw32},
        // TODO The __mspabi_mpysl*_hw32 functions ARE implemented in libgcc
        // TODO The __mspabi_mpyul*_hw32 functions ARE implemented in libgcc
    };
    for (const auto &LC : LibraryCalls) {
      Info.setLibcallImpl(LC.Op, LC.Impl);
    }
  } else if (hasHWMultF5()) {
    const struct {
      const RTLIB::Libcall Op;
      const RTLIB::LibcallImpl Impl;
    } LibraryCalls[] = {
        // Integer Multiply - EABI Table 9
        {RTLIB::MUL_I16, RTLIB::impl___mspabi_mpyi_f5hw},
        {RTLIB::MUL_I32, RTLIB::impl___mspabi_mpyl_f5hw},
        {RTLIB::MUL_I64, RTLIB::impl___mspabi_mpyll_f5hw},
        // TODO The __mspabi_mpysl*_f5hw functions ARE implemented in libgcc
        // TODO The __mspabi_mpyul*_f5hw functions ARE implemented in libgcc
    };
    for (const auto &LC : LibraryCalls) {
      Info.setLibcallImpl(LC.Op, LC.Impl);
    }
  } else { // NoHWMult
    const struct {
      const RTLIB::Libcall Op;
      const RTLIB::LibcallImpl Impl;
    } LibraryCalls[] = {
        // Integer Multiply - EABI Table 9
        {RTLIB::MUL_I16, RTLIB::impl___mspabi_mpyi},
        {RTLIB::MUL_I32, RTLIB::impl___mspabi_mpyl},
        {RTLIB::MUL_I64, RTLIB::impl___mspabi_mpyll},
        // The __mspabi_mpysl* functions are NOT implemented in libgcc
        // The __mspabi_mpyul* functions are NOT implemented in libgcc
    };
    for (const auto &LC : LibraryCalls) {
      Info.setLibcallImpl(LC.Op, LC.Impl);
    }
  }
}

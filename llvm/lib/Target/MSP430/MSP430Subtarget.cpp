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

  // The generic soft-float/integer helper routines (__addsf3, __divli, ...)
  // exist in msp430 libgcc alongside the __mspabi_* variants, so both are
  // available. The __mspabi_* variants are the one that should be used.
  static const struct {
    const RTLIB::Libcall Op;
    const RTLIB::LibcallImpl Impl;
  } EABISelected[] = {
      // Floating point conversions - EABI Table 6.
      {RTLIB::FPROUND_F64_F32, RTLIB::impl___mspabi_cvtdf},
      {RTLIB::FPEXT_F32_F64, RTLIB::impl___mspabi_cvtfd},
      {RTLIB::FPTOSINT_F64_I32, RTLIB::impl___mspabi_fixdli},
      {RTLIB::FPTOSINT_F64_I64, RTLIB::impl___mspabi_fixdlli},
      {RTLIB::FPTOUINT_F64_I32, RTLIB::impl___mspabi_fixdul},
      {RTLIB::FPTOUINT_F64_I64, RTLIB::impl___mspabi_fixdull},
      {RTLIB::FPTOSINT_F32_I32, RTLIB::impl___mspabi_fixfli},
      {RTLIB::FPTOSINT_F32_I64, RTLIB::impl___mspabi_fixflli},
      {RTLIB::FPTOUINT_F32_I32, RTLIB::impl___mspabi_fixful},
      {RTLIB::FPTOUINT_F32_I64, RTLIB::impl___mspabi_fixfull},
      {RTLIB::SINTTOFP_I32_F64, RTLIB::impl___mspabi_fltlid},
      {RTLIB::SINTTOFP_I64_F64, RTLIB::impl___mspabi_fltllid},
      {RTLIB::UINTTOFP_I32_F64, RTLIB::impl___mspabi_fltuld},
      {RTLIB::UINTTOFP_I64_F64, RTLIB::impl___mspabi_fltulld},
      {RTLIB::SINTTOFP_I32_F32, RTLIB::impl___mspabi_fltlif},
      {RTLIB::SINTTOFP_I64_F32, RTLIB::impl___mspabi_fltllif},
      {RTLIB::UINTTOFP_I32_F32, RTLIB::impl___mspabi_fltulf},
      {RTLIB::UINTTOFP_I64_F32, RTLIB::impl___mspabi_fltullf},
      // Floating point comparisons - EABI Table 7.
      {RTLIB::OEQ_F64, RTLIB::impl___mspabi_cmpd__oeq},
      {RTLIB::OGE_F64, RTLIB::impl___mspabi_cmpd__oge},
      {RTLIB::OLT_F64, RTLIB::impl___mspabi_cmpd__olt},
      {RTLIB::OLE_F64, RTLIB::impl___mspabi_cmpd__ole},
      {RTLIB::OGT_F64, RTLIB::impl___mspabi_cmpd__ogt},
      {RTLIB::OEQ_F32, RTLIB::impl___mspabi_cmpf__oeq},
      {RTLIB::UNE_F32, RTLIB::impl___mspabi_cmpf__une},
      {RTLIB::OGE_F32, RTLIB::impl___mspabi_cmpf__oge},
      {RTLIB::OLT_F32, RTLIB::impl___mspabi_cmpf__olt},
      {RTLIB::OLE_F32, RTLIB::impl___mspabi_cmpf__ole},
      {RTLIB::OGT_F32, RTLIB::impl___mspabi_cmpf__ogt},
      // Floating point arithmetic - EABI Table 8.
      {RTLIB::ADD_F64, RTLIB::impl___mspabi_addd},
      {RTLIB::SUB_F64, RTLIB::impl___mspabi_subd},
      {RTLIB::MUL_F64, RTLIB::impl___mspabi_mpyd},
      {RTLIB::DIV_F64, RTLIB::impl___mspabi_divd},
      {RTLIB::ADD_F32, RTLIB::impl___mspabi_addf},
      {RTLIB::SUB_F32, RTLIB::impl___mspabi_subf},
      {RTLIB::MUL_F32, RTLIB::impl___mspabi_mpyf},
      {RTLIB::DIV_F32, RTLIB::impl___mspabi_divf},
      // Universal Integer Operations - EABI Table 9.
      {RTLIB::SDIV_I16, RTLIB::impl___mspabi_divi},
      {RTLIB::SDIV_I32, RTLIB::impl___mspabi_divli},
      {RTLIB::SDIV_I64, RTLIB::impl___mspabi_divlli},
      {RTLIB::UDIV_I16, RTLIB::impl___mspabi_divu},
      {RTLIB::UDIV_I32, RTLIB::impl___mspabi_divul},
      {RTLIB::UDIV_I64, RTLIB::impl___mspabi_divull},
      {RTLIB::SREM_I16, RTLIB::impl___mspabi_remi},
      {RTLIB::SREM_I32, RTLIB::impl___mspabi_remli},
      {RTLIB::SREM_I64, RTLIB::impl___mspabi_remlli},
      {RTLIB::UREM_I16, RTLIB::impl___mspabi_remu},
      {RTLIB::UREM_I32, RTLIB::impl___mspabi_remul},
      {RTLIB::UREM_I64, RTLIB::impl___mspabi_remull},
      // Bitwise Operations - EABI Table 10.
      {RTLIB::SHL_I32, RTLIB::impl___mspabi_slll},
      {RTLIB::SRA_I32, RTLIB::impl___mspabi_sral},
  };
  for (const auto &LC : EABISelected)
    Info.setLibcallImpl(LC.Op, LC.Impl);
}

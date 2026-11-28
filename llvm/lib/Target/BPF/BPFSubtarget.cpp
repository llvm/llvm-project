//===-- BPFSubtarget.cpp - BPF Subtarget Information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the BPF specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "BPFSubtarget.h"
#include "BPF.h"
#include "BPFTargetMachine.h"
#include "GISel/BPFCallLowering.h"
#include "GISel/BPFLegalizerInfo.h"
#include "GISel/BPFRegisterBankInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Host.h"

using namespace llvm;

#define DEBUG_TYPE "bpf-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "BPFGenSubtargetInfo.inc"

static cl::opt<bool> Disable_ldsx("disable-ldsx", cl::Hidden, cl::init(false),
  cl::desc("Disable ldsx insns"));
static cl::opt<bool> Disable_movsx("disable-movsx", cl::Hidden, cl::init(false),
  cl::desc("Disable movsx insns"));
static cl::opt<bool> Disable_bswap("disable-bswap", cl::Hidden, cl::init(false),
  cl::desc("Disable bswap insns"));
static cl::opt<bool> Disable_sdiv_smod("disable-sdiv-smod", cl::Hidden,
  cl::init(false), cl::desc("Disable sdiv/smod insns"));
static cl::opt<bool> Disable_gotol("disable-gotol", cl::Hidden, cl::init(false),
  cl::desc("Disable gotol insn"));
static cl::opt<bool>
    Disable_StoreImm("disable-storeimm", cl::Hidden, cl::init(false),
                     cl::desc("Disable BPF_ST (immediate store) insn"));
static cl::opt<bool> Disable_load_acq_store_rel(
    "disable-load-acq-store-rel", cl::Hidden, cl::init(false),
    cl::desc("Disable load-acquire and store-release insns"));
static cl::opt<bool> Disable_gotox("disable-gotox", cl::Hidden, cl::init(false),
                                   cl::desc("Disable gotox insn"));
static cl::opt<bool>
    Disable_hmul("disable-hmul", cl::Hidden, cl::init(false),
                 cl::desc("Disable high-half multiply (UHMUL/SHMUL) insns"));

static cl::opt<bool> BPFStackSpilling(
    "bpf-stack-spilling", cl::init(false),
    cl::desc("Pass extra BPF arguments in upward-growing r10 stack frames"));

static cl::opt<int>
    BPFStackSizeOption("bpf-stack-size",
                       cl::desc("Specify the BPF stack size limit"),
                       cl::init(512));

bool BPFSubtarget::useStackSpilling() const { return BPFStackSpilling; }

int BPFSubtarget::getStackSizeLimit() const {
  if (useStackSpilling() && !BPFStackSizeOption.getNumOccurrences())
    return 4096;
  return BPFStackSizeOption;
}

void BPFSubtarget::anchor() {}

BPFSubtarget &BPFSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                            StringRef FS) {
  initializeEnvironment();
  initSubtargetFeatures(CPU, FS);
  ParseSubtargetFeatures(CPU, /*TuneCPU*/ CPU, FS);
  // Stack spilling accesses the bottom of the frame at r10 - frame_size.
  // Load/store offsets are signed 16-bit, so frame_size must not exceed 32768.
  // Larger frames would require materializing an adjusted base register, which
  // frame index elimination does not currently support.
  if (useStackSpilling() &&
      (getStackSizeLimit() < 64 || getStackSizeLimit() > 32768 ||
       getStackSizeLimit() % 64))
    report_fatal_error("BPF stack spilling requires a stack size that is a "
                       "multiple of 64 between 64 and 32768 bytes");
  return *this;
}

void BPFSubtarget::initializeEnvironment() {
  HasJmpExt = false;
  HasJmp32 = false;
  HasAlu32 = false;
  UseDwarfRIS = false;
  HasLdsx = false;
  HasMovsx = false;
  HasBswap = false;
  HasSdivSmod = false;
  HasGotol = false;
  HasStoreImm = false;
  HasLoadAcqStoreRel = false;
  HasGotox = false;
  HasHmul = false;
  AllowsMisalignedMemAccess = false;
  HasFuncTable = false;
}

void BPFSubtarget::initSubtargetFeatures(StringRef CPU, StringRef FS) {
  if (CPU.empty())
    CPU = "v3";
  if (CPU == "probe")
    CPU = sys::detail::getHostCPUNameForBPF();
  if (CPU == "generic" || CPU == "v1")
    return;
  if (CPU == "v2") {
    HasJmpExt = true;
    return;
  }
  if (CPU == "v3") {
    HasJmpExt = true;
    HasJmp32 = true;
    HasAlu32 = true;
    return;
  }
  if (CPU == "v4" || CPU == "v5") {
    HasJmpExt = true;
    HasJmp32 = true;
    HasAlu32 = true;
    HasLdsx = !Disable_ldsx;
    HasMovsx = !Disable_movsx;
    HasBswap = !Disable_bswap;
    HasSdivSmod = !Disable_sdiv_smod;
    HasGotol = !Disable_gotol;
    HasStoreImm = !Disable_StoreImm;
    HasLoadAcqStoreRel = !Disable_load_acq_store_rel;
    HasGotox = !Disable_gotox;
    HasHmul = CPU == "v5" && !Disable_hmul;
    return;
  }
}

BPFSubtarget::BPFSubtarget(const Triple &TT, const std::string &CPU,
                           const std::string &FS, const TargetMachine &TM)
    : BPFGenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)), FrameLowering(*this),
      TLInfo(TM, *this) {
  IsLittleEndian = TT.isLittleEndian();

  CallLoweringInfo.reset(new BPFCallLowering(*getTargetLowering()));
  Legalizer.reset(new BPFLegalizerInfo(*this));
  auto *RBI = new BPFRegisterBankInfo(*getRegisterInfo());
  RegBankInfo.reset(RBI);

  InstSelector.reset(createBPFInstructionSelector(
      *static_cast<const BPFTargetMachine *>(&TM), *this, *RBI));
}

const CallLowering *BPFSubtarget::getCallLowering() const {
  return CallLoweringInfo.get();
}

InstructionSelector *BPFSubtarget::getInstructionSelector() const {
  return InstSelector.get();
}

const LegalizerInfo *BPFSubtarget::getLegalizerInfo() const {
  return Legalizer.get();
}

const RegisterBankInfo *BPFSubtarget::getRegBankInfo() const {
  return RegBankInfo.get();
}

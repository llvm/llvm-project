//===-- XtensaMCTargetDesc.cpp - Xtensa target descriptions ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "XtensaMCTargetDesc.h"
#include "TargetInfo/XtensaTargetInfo.h"
#include "XtensaInstPrinter.h"
#include "XtensaMCAsmInfo.h"
#include "XtensaTargetStreamer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_INSTRINFO_MC_DESC
#include "XtensaGenInstrInfo.inc"

#define GET_REGINFO_MC_DESC
#include "XtensaGenRegisterInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "XtensaGenSubtargetInfo.inc"

using namespace llvm;

bool Xtensa::isValidAddrOffset(int Scale, int64_t OffsetVal) {
  bool Valid = false;

  switch (Scale) {
  case 1:
    Valid = (OffsetVal >= 0 && OffsetVal <= 255);
    break;
  case 2:
    Valid = (OffsetVal >= 0 && OffsetVal <= 510) && ((OffsetVal & 0x1) == 0);
    break;
  case 4:
    Valid = (OffsetVal >= 0 && OffsetVal <= 1020) && ((OffsetVal & 0x3) == 0);
    break;
  default:
    break;
  }
  return Valid;
}

bool Xtensa::isValidAddrOffsetForOpcode(unsigned Opcode, int64_t Offset) {
  int Scale = 0;

  switch (Opcode) {
  case Xtensa::L8UI:
  case Xtensa::S8I:
    Scale = 1;
    break;
  case Xtensa::L16SI:
  case Xtensa::L16UI:
  case Xtensa::S16I:
    Scale = 2;
    break;
  case Xtensa::LEA_ADD:
    return (Offset >= -128 && Offset <= 127);
  default:
    // assume that MI is 32-bit load/store operation
    Scale = 4;
    break;
  }
  return isValidAddrOffset(Scale, Offset);
}

// Verify Special Register
bool Xtensa::checkRegister(MCRegister RegNo, const FeatureBitset &FeatureBits,
                           RegisterAccessType RAType) {
  switch (RegNo) {
  case Xtensa::BREG:
    return FeatureBits[Xtensa::FeatureBoolean];
  case Xtensa::CCOUNT:
  case Xtensa::CCOMPARE0:
    if (FeatureBits[Xtensa::FeatureTimers1])
      return true;
    [[fallthrough]];
  case Xtensa::CCOMPARE1:
    if (FeatureBits[Xtensa::FeatureTimers2])
      return true;
    [[fallthrough]];
  case Xtensa::CCOMPARE2:
    if (FeatureBits[Xtensa::FeatureTimers3])
      return true;
    return false;
  case Xtensa::CONFIGID0:
    return RAType != Xtensa::REGISTER_EXCHANGE;
  case Xtensa::CONFIGID1:
    return RAType == Xtensa::REGISTER_READ;
  case Xtensa::CPENABLE:
    return FeatureBits[Xtensa::FeatureCoprocessor];
  case Xtensa::DEBUGCAUSE:
    return RAType == Xtensa::REGISTER_READ && FeatureBits[Xtensa::FeatureDebug];
  case Xtensa::DEPC:
  case Xtensa::EPC1:
  case Xtensa::EXCCAUSE:
  case Xtensa::EXCSAVE1:
  case Xtensa::EXCVADDR:
    return FeatureBits[Xtensa::FeatureException];
    [[fallthrough]];
  case Xtensa::EPC2:
  case Xtensa::EPS2:
  case Xtensa::EXCSAVE2:
    if (FeatureBits[Xtensa::FeatureHighPriInterrupts])
      return true;
    [[fallthrough]];
  case Xtensa::EPC3:
  case Xtensa::EPS3:
  case Xtensa::EXCSAVE3:
    if (FeatureBits[Xtensa::FeatureHighPriInterruptsLevel3])
      return true;
    [[fallthrough]];
  case Xtensa::EPC4:
  case Xtensa::EPS4:
  case Xtensa::EXCSAVE4:
    if (FeatureBits[Xtensa::FeatureHighPriInterruptsLevel4])
      return true;
    [[fallthrough]];
  case Xtensa::EPC5:
  case Xtensa::EPS5:
  case Xtensa::EXCSAVE5:
    if (FeatureBits[Xtensa::FeatureHighPriInterruptsLevel5])
      return true;
    [[fallthrough]];
  case Xtensa::EPC6:
  case Xtensa::EPS6:
  case Xtensa::EXCSAVE6:
    if (FeatureBits[Xtensa::FeatureHighPriInterruptsLevel6])
      return true;
    [[fallthrough]];
  case Xtensa::EPC7:
  case Xtensa::EPS7:
  case Xtensa::EXCSAVE7:
    if (FeatureBits[Xtensa::FeatureHighPriInterruptsLevel7])
      return true;
    return false;
  case Xtensa::INTENABLE:
    return FeatureBits[Xtensa::FeatureInterrupt];
  case Xtensa::INTERRUPT:
    return RAType == Xtensa::REGISTER_READ &&
           FeatureBits[Xtensa::FeatureInterrupt];
  case Xtensa::INTSET:
  case Xtensa::INTCLEAR:
    return RAType == Xtensa::REGISTER_WRITE &&
           FeatureBits[Xtensa::FeatureInterrupt];
  case Xtensa::ICOUNT:
  case Xtensa::ICOUNTLEVEL:
  case Xtensa::IBREAKENABLE:
  case Xtensa::DDR:
  case Xtensa::IBREAKA0:
  case Xtensa::IBREAKA1:
  case Xtensa::DBREAKA0:
  case Xtensa::DBREAKA1:
  case Xtensa::DBREAKC0:
  case Xtensa::DBREAKC1:
    return FeatureBits[Xtensa::FeatureDebug];
  case Xtensa::LBEG:
  case Xtensa::LEND:
  case Xtensa::LCOUNT:
    return FeatureBits[Xtensa::FeatureLoop];
  case Xtensa::LITBASE:
    return FeatureBits[Xtensa::FeatureExtendedL32R];
  case Xtensa::MEMCTL:
    return FeatureBits[Xtensa::FeatureDataCache];
  case Xtensa::ACCLO:
  case Xtensa::ACCHI:
  case Xtensa::M0:
  case Xtensa::M1:
  case Xtensa::M2:
  case Xtensa::M3:
    return FeatureBits[Xtensa::FeatureMAC16];
  case Xtensa::MISC0:
  case Xtensa::MISC1:
  case Xtensa::MISC2:
  case Xtensa::MISC3:
    return FeatureBits[Xtensa::FeatureMiscSR];
  case Xtensa::PRID:
    return RAType == Xtensa::REGISTER_READ && FeatureBits[Xtensa::FeaturePRID];
  case Xtensa::THREADPTR:
    return FeatureBits[FeatureTHREADPTR];
  case Xtensa::VECBASE:
    return FeatureBits[Xtensa::FeatureRelocatableVector];
  case Xtensa::FCR:
  case Xtensa::FSR:
    return FeatureBits[FeatureSingleFloat];
  case Xtensa::F64R_LO:
  case Xtensa::F64R_HI:
  case Xtensa::F64S:
    return FeatureBits[FeatureDFPAccel];
  case Xtensa::WINDOWBASE:
  case Xtensa::WINDOWSTART:
    return FeatureBits[Xtensa::FeatureWindowed];
  case Xtensa::ATOMCTL:
  case Xtensa::SCOMPARE1:
    return FeatureBits[Xtensa::FeatureWindowed];
  case Xtensa::NoRegister:
    return false;
  }

  return true;
}

// Get Xtensa User Register by encoding value.
MCRegister Xtensa::getUserRegister(unsigned Code, const MCRegisterInfo &MRI) {
  MCRegister UserReg = Xtensa::NoRegister;

  if (MRI.getEncodingValue(Xtensa::FCR) == Code) {
    UserReg = Xtensa::FCR;
  } else if (MRI.getEncodingValue(Xtensa::FSR) == Code) {
    UserReg = Xtensa::FSR;
  } else if (MRI.getEncodingValue(Xtensa::F64R_LO) == Code) {
    UserReg = Xtensa::F64R_LO;
  } else if (MRI.getEncodingValue(Xtensa::F64R_HI) == Code) {
    UserReg = Xtensa::F64R_HI;
  } else if (MRI.getEncodingValue(Xtensa::F64S) == Code) {
    UserReg = Xtensa::F64S;
  } else if (MRI.getEncodingValue(Xtensa::THREADPTR) == Code) {
    UserReg = Xtensa::THREADPTR;
  }

  return UserReg;
}

static MCAsmInfo *createXtensaMCAsmInfo(const MCRegisterInfo &MRI,
                                        const Triple &TT,
                                        const MCTargetOptions &Options) {
  MCAsmInfo *MAI = new XtensaMCAsmInfo(TT);
  return MAI;
}

static MCInstrInfo *createXtensaMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitXtensaMCInstrInfo(X);
  return X;
}

static MCInstPrinter *createXtensaMCInstPrinter(const Triple &TT,
                                                unsigned SyntaxVariant,
                                                const MCAsmInfo &MAI,
                                                const MCInstrInfo &MII,
                                                const MCRegisterInfo &MRI) {
  return new XtensaInstPrinter(MAI, MII, MRI);
}

static MCRegisterInfo *createXtensaMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitXtensaMCRegisterInfo(X, Xtensa::SP);
  return X;
}

static MCSubtargetInfo *
createXtensaMCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  return createXtensaMCSubtargetInfoImpl(TT, CPU, CPU, FS);
}

static MCTargetStreamer *
createXtensaAsmTargetStreamer(MCStreamer &S, formatted_raw_ostream &OS,
                              MCInstPrinter *InstPrint) {
  return new XtensaTargetAsmStreamer(S, OS);
}

static MCTargetStreamer *
createXtensaObjectTargetStreamer(MCStreamer &S, const MCSubtargetInfo &STI) {
  return new XtensaTargetELFStreamer(S);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeXtensaTargetMC() {
  // Register the MCAsmInfo.
  TargetRegistry::RegisterMCAsmInfo(getTheXtensaTarget(),
                                    createXtensaMCAsmInfo);

  // Register the MCCodeEmitter.
  TargetRegistry::RegisterMCCodeEmitter(getTheXtensaTarget(),
                                        createXtensaMCCodeEmitter);

  // Register the MCInstrInfo.
  TargetRegistry::RegisterMCInstrInfo(getTheXtensaTarget(),
                                      createXtensaMCInstrInfo);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(getTheXtensaTarget(),
                                        createXtensaMCInstPrinter);

  // Register the MCRegisterInfo.
  TargetRegistry::RegisterMCRegInfo(getTheXtensaTarget(),
                                    createXtensaMCRegisterInfo);

  // Register the MCSubtargetInfo.
  TargetRegistry::RegisterMCSubtargetInfo(getTheXtensaTarget(),
                                          createXtensaMCSubtargetInfo);

  // Register the MCAsmBackend.
  TargetRegistry::RegisterMCAsmBackend(getTheXtensaTarget(),
                                       createXtensaAsmBackend);

  // Register the asm target streamer.
  TargetRegistry::RegisterAsmTargetStreamer(getTheXtensaTarget(),
                                            createXtensaAsmTargetStreamer);

  // Register the ELF target streamer.
  TargetRegistry::RegisterObjectTargetStreamer(
      getTheXtensaTarget(), createXtensaObjectTargetStreamer);
}

//=====-- NVPTXSubtarget.h - Define Subtarget for the NVPTX ---*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the NVPTX specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXSUBTARGET_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXSUBTARGET_H

#include "NVPTX.h"
#include "NVPTXFrameLowering.h"
#include "NVPTXISelLowering.h"
#include "NVPTXInstrInfo.h"
#include "NVPTXRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <string>

#define GET_SUBTARGETINFO_HEADER
#include "NVPTXGenSubtargetInfo.inc"

namespace llvm {

class NVPTXSubtarget : public NVPTXGenSubtargetInfo {
  virtual void anchor();
  std::string TargetName;

  // PTX version x.y is represented as 10*x+y, e.g. 3.1 == 31
  unsigned PTXVersion;

  // SM version x.y is represented as 10*x+y, e.g. 3.1 == 31
  unsigned int SmVersion;

  const NVPTXTargetMachine &TM;
  NVPTXInstrInfo InstrInfo;
  NVPTXTargetLowering TLInfo;
  SelectionDAGTargetInfo TSInfo;

  // NVPTX does not have any call stack frame, but need a NVPTX specific
  // FrameLowering class because TargetFrameLowering is abstract.
  NVPTXFrameLowering FrameLowering;

public:
  /// This constructor initializes the data members to match that
  /// of the specified module.
  ///
  NVPTXSubtarget(const Triple &TT, const std::string &CPU,
                 const std::string &FS, const NVPTXTargetMachine &TM);

  const TargetFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }
  const NVPTXInstrInfo *getInstrInfo() const override { return &InstrInfo; }
  const NVPTXRegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }
  const NVPTXTargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }
  const SelectionDAGTargetInfo *getSelectionDAGInfo() const override {
    return &TSInfo;
  }

  bool hasBrkPt() const { return SmVersion >= 11; }
  bool hasAtomRedG32() const { return SmVersion >= 11; }
  bool hasAtomRedS32() const { return SmVersion >= 12; }
  bool hasAtomRedG64() const { return SmVersion >= 12; }
  bool hasAtomRedS64() const { return SmVersion >= 20; }
  bool hasAtomRedGen32() const { return SmVersion >= 20; }
  bool hasAtomRedGen64() const { return SmVersion >= 20; }
  bool hasAtomAddF32() const { return SmVersion >= 20; }
  bool hasVote() const { return SmVersion >= 12; }
  bool hasDouble() const { return SmVersion >= 13; }
  bool reqPTX20() const { return SmVersion >= 20; }
  bool hasF32FTZ() const { return SmVersion >= 20; }
  bool hasFMAF32() const { return SmVersion >= 20; }
  bool hasFMAF64() const { return SmVersion >= 13; }
  bool hasLDG() const { return SmVersion >= 32; }
  bool hasLDU() const { return ((SmVersion >= 20) && (SmVersion < 30)); }
  bool hasGenericLdSt() const { return SmVersion >= 20; }
  inline bool hasHWROT32() const { return SmVersion >= 32; }
  inline bool hasSWROT32() const {
    return ((SmVersion >= 20) && (SmVersion < 32));
  }
  inline bool hasROT32() const { return hasHWROT32() || hasSWROT32(); }
  inline bool hasROT64() const { return SmVersion >= 20; }
  bool hasImageHandles() const;

  unsigned int getSmVersion() const { return SmVersion; }
  std::string getTargetName() const { return TargetName; }

  unsigned getPTXVersion() const { return PTXVersion; }

  NVPTXSubtarget &initializeSubtargetDependencies(StringRef CPU, StringRef FS);
  void ParseSubtargetFeatures(StringRef CPU, StringRef FS);
};

} // End llvm namespace

#endif

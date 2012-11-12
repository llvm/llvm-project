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

#ifndef NVPTXSUBTARGET_H
#define NVPTXSUBTARGET_H

#include "llvm/Target/TargetSubtargetInfo.h"
#include "NVPTX.h"

#define GET_SUBTARGETINFO_HEADER
#include "NVPTXGenSubtargetInfo.inc"

#include <string>

namespace llvm {

class NVPTXSubtarget : public NVPTXGenSubtargetInfo {
  
  std::string TargetName;
  NVPTX::DrvInterface drvInterface;
  bool dummy; // For the 'dummy' feature, see NVPTX.td
  bool Is64Bit;

  // PTX version x.y is represented as 10*x+y, e.g. 3.1 == 31
  unsigned PTXVersion;

  // SM version x.y is represented as 10*x+y, e.g. 3.1 == 31
  unsigned int SmVersion;

public:
  /// This constructor initializes the data members to match that
  /// of the specified module.
  ///
  NVPTXSubtarget(const std::string &TT, const std::string &CPU,
                 const std::string &FS, bool is64Bit);

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
  bool hasLDU() const { return SmVersion >= 20; }
  bool hasGenericLdSt() const { return SmVersion >= 20; }
  inline bool hasHWROT32() const { return false; }
  inline bool hasSWROT32() const {
    return true;
  }
  inline bool hasROT32() const { return hasHWROT32() || hasSWROT32() ; }
  inline bool hasROT64() const { return SmVersion >= 20; }


  bool is64Bit() const { return Is64Bit; }

  unsigned int getSmVersion() const { return SmVersion; }
  NVPTX::DrvInterface getDrvInterface() const { return drvInterface; }
  std::string getTargetName() const { return TargetName; }

  unsigned getPTXVersion() const { return PTXVersion; }

  void ParseSubtargetFeatures(StringRef CPU, StringRef FS);

  std::string getDataLayout() const {
    const char *p;
    if (is64Bit())
      p = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"
          "f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-"
          "n16:32:64";
    else
      p = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"
          "f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-"
          "n16:32:64";

    return std::string(p);
  }

};

} // End llvm namespace

#endif  // NVPTXSUBTARGET_H

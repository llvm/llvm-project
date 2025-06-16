//===-- ARMMCAsmInfo.h - ARM asm properties --------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the ARMMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_MCTARGETDESC_ARMMCASMINFO_H
#define LLVM_LIB_TARGET_ARM_MCTARGETDESC_ARMMCASMINFO_H

#include "MCTargetDesc/ARMMCExpr.h"
#include "llvm/MC/MCAsmInfoCOFF.h"
#include "llvm/MC/MCAsmInfoDarwin.h"
#include "llvm/MC/MCAsmInfoELF.h"
#include "llvm/MC/MCExpr.h"

namespace llvm {
class Triple;

class ARMMCAsmInfoDarwin : public MCAsmInfoDarwin {
  virtual void anchor();

public:
  explicit ARMMCAsmInfoDarwin(const Triple &TheTriple);
};

class ARMELFMCAsmInfo : public MCAsmInfoELF {
  void anchor() override;

public:
  explicit ARMELFMCAsmInfo(const Triple &TT);

  void setUseIntegratedAssembler(bool Value) override;
};

class ARMCOFFMCAsmInfoMicrosoft : public MCAsmInfoMicrosoft {
  void anchor() override;

public:
  explicit ARMCOFFMCAsmInfoMicrosoft();
};

class ARMCOFFMCAsmInfoGNU : public MCAsmInfoGNUCOFF {
  void anchor() override;

public:
  explicit ARMCOFFMCAsmInfoGNU();
};

namespace ARM {
enum {
  S_None,
  S_HI16 =
      MCSymbolRefExpr::FirstTargetSpecifier, // The R_ARM_MOVT_ABS relocation
                                             // (:upper16: in the .s file)
  S_LO16, // The R_ARM_MOVW_ABS_NC relocation (:lower16: in the .s file)

  S_HI_8_15, // The R_ARM_THM_ALU_ABS_G3    relocation (:upper8_15: in
             // the .s file)
  S_HI_0_7,  // The R_ARM_THM_ALU_ABS_G2_NC relocation (:upper0_8: in the
             // .s file)
  S_LO_8_15, // The R_ARM_THM_ALU_ABS_G1_NC relocation (:lower8_15: in
             // the .s file)
  S_LO_0_7,  // The R_ARM_THM_ALU_ABS_G0_NC relocation (:lower0_7: in the
             // .s file)

  S_ARM_NONE,
  S_FUNCDESC,
  S_GOT,
  S_GOTFUNCDESC,
  S_GOTOFF,
  S_GOTOFFFUNCDESC,
  S_GOTTPOFF,
  S_GOTTPOFF_FDPIC,
  S_GOT_PREL,
  S_PLT,
  S_PREL31,
  S_SBREL,
  S_TARGET1,
  S_TARGET2,
  S_TLSCALL,
  S_TLSDESC,
  S_TLSDESCSEQ,
  S_TLSGD,
  S_TLSGD_FDPIC,
  S_TLSLDM,
  S_TLSLDM_FDPIC,
  S_TLSLDO,
  S_TPOFF,
};
}

} // namespace llvm

#endif

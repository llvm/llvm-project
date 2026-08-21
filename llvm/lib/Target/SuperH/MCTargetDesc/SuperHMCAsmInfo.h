//===-- SuperHMCAsmInfo.h - SuperH Asm Info -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides the SuperHAsmInfo class.
///
//===----------------------------------------------------------------------===//


#ifndef LLVM_LIB_TARGET_SUPERH_MCTARGETDESC_SUPERHMCASMINFO_H
#define LLVM_LIB_TARGET_SUPERH_MCTARGETDESC_SUPERHMCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"
#include "llvm/MC/MCExpr.h"

namespace llvm {
class Triple;

//===----------------------------------------------------------------------===//
//
// Class which provides the information needed to emit a SuperH ELF file.
//
//===----------------------------------------------------------------------===//
class SuperHMCAsmInfo : public MCAsmInfoELF {
private:
	void anchor() override;

public:
	explicit SuperHMCAsmInfo(const Triple &TheTriple,
                             const MCTargetOptions &Options);
};

namespace SH {
using Specifier = uint16_t;
enum {
  S_None,

  S_SH_NONE = MCSymbolRefExpr::FirstTargetSpecifier,
  S_GOT,
  S_GOT_OFF,
  S_GOT_PCREL,
  S_PCREL,
  S_DIR,
};
} // namespace SH

} // namespace llvm

#endif
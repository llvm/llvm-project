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

} // end namespace llvm

#endif
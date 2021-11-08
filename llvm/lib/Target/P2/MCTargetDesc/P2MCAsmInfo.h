//===-- P2MCAsmInfo.h - P2 Asm Info ------------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the P2MCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_P2_P2MCASMINFO_H
#define LLVM_LIB_TARGET_P2_P2MCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {
    class Triple;

    class P2MCAsmInfo : public MCAsmInfoELF {

        void anchor() override;
    public:
        explicit P2MCAsmInfo(const Triple &TheTriple, const MCTargetOptions &Options);
    };

} // namespace llvm

#endif
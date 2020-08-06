//===-- llvm/Target/P2TargetObjectFile.h - P2 Object Info ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_P2_P2TARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_P2_P2TARGETOBJECTFILE_H

#include "P2TargetMachine.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {
class P2TargetMachine;
    typedef TargetLoweringObjectFileELF Base;

    class P2TargetObjectFile : public TargetLoweringObjectFileELF {
        // MCSection *SmallDataSection;
        // MCSection *SmallBSSSection;
        // const P2TargetMachine *TM;
        MCSection *ProgmemDataSection;
    public:
        //void Initialize(MCContext &Ctx) override;
        void Initialize(MCContext &Ctx, const TargetMachine &TM) override;

    };
} // end namespace llvm

#endif

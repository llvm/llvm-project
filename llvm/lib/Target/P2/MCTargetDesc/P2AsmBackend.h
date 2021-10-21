//===-- P2AsmBackend.h - P2 Asm Backend  ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the P2AsmBackend class.
//
//===----------------------------------------------------------------------===//
//

#ifndef LLVM_LIB_TARGET_P2_MCTARGETDESC_P2ASMBACKEND_H
#define LLVM_LIB_TARGET_P2_MCTARGETDESC_P2ASMBACKEND_H

//#include "MCTargetDesc/P2FixupKinds.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCAsmBackend.h"

namespace llvm {

    class MCAssembler;
    struct MCFixupKindInfo;
    class Target;
    class MCObjectWriter;
    class MCTargetOptions;

    class P2AsmBackend : public MCAsmBackend {
        Triple::OSType OSType;

    public:
        P2AsmBackend(Triple::OSType OSType) : MCAsmBackend(support::little), OSType(OSType) {}

        std::unique_ptr<MCObjectTargetWriter> createObjectTargetWriter() const override;

        void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                        const MCValue &Target, MutableArrayRef<char> Data,
                        uint64_t Value, bool IsResolved,
                        const MCSubtargetInfo *STI) const override;

        const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override;

        unsigned getNumFixupKinds() const override {
            return P2::NumTargetFixupKinds;
        }

        bool mayNeedRelaxation(const MCInst &Inst, const MCSubtargetInfo &STI) const override {
            return false;
        }

        bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                                    const MCRelaxableFragment *DF,
                                    const MCAsmLayout &Layout) const override {
            // FIXME.
            llvm_unreachable("RelaxInstruction() unimplemented");
            return false;
        }
        //bool shouldForceRelocation(const MCAssembler &Asm, const MCFixup &Fixup, const MCValue &Target) override;

        //void relaxInstruction(const MCInst &Inst, const MCSubtargetInfo &STI, MCInst &Res) const override {}

        bool writeNopData(raw_ostream &OS, uint64_t Count, const MCSubtargetInfo *STI) const override;
    };
}

#endif
//===-- DPUAsmBackend.h - DPU Assembler Backend ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the DPUAsmBackend class.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_LIB_TARGET_DPU_MCTARGETDESC_DPUASMBACKEND_H
#define LLVM_LIB_TARGET_DPU_MCTARGETDESC_DPUASMBACKEND_H

#include "llvm/MC/MCAsmBackend.h"
#include "DPUMCTargetDesc.h"

namespace llvm {
class DPUAsmBackend : public MCAsmBackend {
    const Target& T;
    const MCSubtargetInfo& STI;
    const MCRegisterInfo& MRI;
    const MCTargetOptions& Options;

public:
    DPUAsmBackend(const Target &T, const MCSubtargetInfo &STI, const MCRegisterInfo &MRI,
            const MCTargetOptions &Options): MCAsmBackend(support::little), T(T), STI(STI), MRI(MRI), Options(Options) {}

    std::unique_ptr<MCObjectTargetWriter> createObjectTargetWriter() const override;

    unsigned int getNumFixupKinds() const override;

    void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup, const MCValue &Target,
                    MutableArrayRef<char> Data, uint64_t Value, bool IsResolved, const MCSubtargetInfo * STI) const override;

    bool mayNeedRelaxation(const MCInst&, const MCSubtargetInfo&) const override;

    bool fixupNeedsRelaxation(const MCFixup&, uint64_t, const MCRelaxableFragment*, const MCAsmLayout&) const override;

    void relaxInstruction(const MCInst&, const MCSubtargetInfo&, MCInst&) const override;

    bool writeNopData(raw_ostream&, uint64_t) const override;

    const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override;
};
}

#endif // LLVM_LIB_TARGET_DPU_MCTARGETDESC_DPUASMBACKEND_H

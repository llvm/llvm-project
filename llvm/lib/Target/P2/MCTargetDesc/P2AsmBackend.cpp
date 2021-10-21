//===-- P2AsmBackend.cpp - P2 Asm Backend  ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the P2AsmBackend class.
//
//===----------------------------------------------------------------------===//
//

#include "MCTargetDesc/P2FixupKinds.h"
#include "MCTargetDesc/P2AsmBackend.h"
#include "MCTargetDesc/P2MCTargetDesc.h"

#include "P2Subtarget.h"

#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "p2-asm-backend"

using namespace llvm;
static unsigned adjustFixupValue(const MCFixup &Fixup, uint64_t Value, bool cogex=false, MCContext *Ctx = nullptr) {

    unsigned Kind = Fixup.getKind();

    // Add/subtract and shift
    switch (Kind) {
        case P2::fixup_P2_32:
        case P2::fixup_P2_PC32:
        case P2::fixup_P2_20:
        case P2::fixup_P2_AUG20:
        case P2::fixup_P2_COG9:
            break;
        case P2::fixup_P2_PC20:
            Value -= 4; // a relative jump automatically includes the next instruction, so reduce the jump by 1 instruction (4 bytes)
            Value &= 0xfffff; // mask the 20 bits in case the relative jump is negative

            break;
        case P2::fixup_P2_PCCOG9:
            Value -= 4;
            Value /= 4;
            Value &= 0x1ff;
            break;
        default:
            return 0;
    }

    return Value;
}


std::unique_ptr<MCObjectTargetWriter> P2AsmBackend::createObjectTargetWriter() const {
    return createP2ELFObjectWriter(MCELFObjectTargetWriter::getOSABI(OSType));
}

void P2AsmBackend::applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                        const MCValue &Target, MutableArrayRef<char> Data,
                        uint64_t Value, bool IsResolved,
                        const MCSubtargetInfo *STI) const {

    // const P2Subtarget *subtarget = static_cast<const P2Subtarget*>(STI);

    MCFixupKind Kind = Fixup.getKind();
    // turns out--you don't need to know if this is hub or cog because of how the JMP instruction works,
    // but going to leave it in for future potential future use. Don't check if it's cogex for now, since STI seems 
    // to come in null so need to figure out what to do about that...
    Value = adjustFixupValue(Fixup, Value);
    uint64_t Mask = ((uint64_t)(-1) >> (64 - getFixupKindInfo(Kind).TargetSize));

    if (!Value)
        return; // Doesn't change encoding.

    LLVM_DEBUG(errs() << "-- applying fixup for ");
    LLVM_DEBUG(Target.dump(); errs() << "\n");
    // if (subtarget->isCogex())
    //     LLVM_DEBUG(errs() << "fixup is for a cogex function\n");

    LLVM_DEBUG(errs() << "new value is " << Value << "\n");

    // Where do we start in the object
    unsigned Offset = Fixup.getOffset();
    LLVM_DEBUG(errs() << "offset is: " << Offset << "\n");
    // Number of bytes we need to fixup
    unsigned NumBytes = (getFixupKindInfo(Kind).TargetSize + 7) / 8;
    LLVM_DEBUG(errs() << "num bytes is: " << NumBytes << "\n");
    // Grab current value, if any, from bits.
    uint64_t CurVal = 0;

    for (unsigned i = 0; i != NumBytes; ++i) {
        CurVal |= (uint64_t)((uint8_t)Data[Offset + i]) << (i*8);
    }

    LLVM_DEBUG(errs() << "current value is: " << CurVal << "\n");

    CurVal |= Value & Mask;

    LLVM_DEBUG(errs() << "masked value is: " << CurVal << "\n");

    // Write out the fixed up bytes back to the code/data bits.
    LLVM_DEBUG(errs() << "Fixing up " << NumBytes << " bytes\n");

    for (unsigned i = 0; i < NumBytes; i++) {
        Data[Offset + i] = (uint8_t)((CurVal >> (i*8)) & 0xff);
        LLVM_DEBUG(errs() << "Byte " << i << ": set data offset by " << Offset + i << " to " << ((CurVal >> (i*8)) & 0xff) << "\n");
    }

    LLVM_DEBUG(errs() << "-- done with fixup\n");
}

const MCFixupKindInfo &P2AsmBackend::getFixupKindInfo(MCFixupKind Kind) const {
    const static MCFixupKindInfo Infos[P2::NumTargetFixupKinds] = {
        // This table *must* be in same the order of fixup_* kinds in
        // P2FixupKinds.h.
        //
        // name                 offset  bits  flags
        { "fixup_P2_32",        0,      32,   0},
        { "fixup_P2_PC32",      0,      32,   MCFixupKindInfo::FKF_IsPCRel},
        { "fixup_P2_20",        0,      20,   0},
        { "fixup_P2_PC20",      0,      20,   MCFixupKindInfo::FKF_IsPCRel},
        { "fixup_P2_AUG20",     0,      20,   0},
        { "fixup_P2_COG9",      0,      9,    0},
        { "fixup_P2_PCCOG9",    0,      9,    MCFixupKindInfo::FKF_IsPCRel}
    };

    if (Kind < FirstTargetFixupKind)
        return MCAsmBackend::getFixupKindInfo(Kind);

    assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() && "Invalid kind!");
    return Infos[Kind - FirstTargetFixupKind];
}

bool P2AsmBackend::writeNopData(raw_ostream &OS, uint64_t Count, const MCSubtargetInfo *STI) const {
    if (Count % 4 != 0) return false;

    OS.write_zeros(Count);
    return true;
}

MCAsmBackend *llvm::createP2AsmBackend(const Target &T, const MCSubtargetInfo &STI,
                                  const MCRegisterInfo &MRI,
                                  const llvm::MCTargetOptions &TO) {
  return new P2AsmBackend(STI.getTargetTriple().getOS());
}

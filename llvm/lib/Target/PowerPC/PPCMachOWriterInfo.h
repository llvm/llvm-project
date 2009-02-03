//===-- PPCMachOWriterInfo.h - Mach-O Writer Info for PowerPC ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Mach-O writer information for the PowerPC backend.
//
//===----------------------------------------------------------------------===//

#ifndef PPC_MACHO_WRITER_INFO_H
#define PPC_MACHO_WRITER_INFO_H

#include "llvm/Target/TargetMachOWriterInfo.h"

namespace llvm {

  // Forward declarations
  class MachineRelocation;
  class OutputBuffer;
  class PPCTargetMachine;

  class PPCMachOWriterInfo : public TargetMachOWriterInfo {
  public:
    PPCMachOWriterInfo(const PPCTargetMachine &TM);
    virtual ~PPCMachOWriterInfo();

    virtual unsigned GetTargetRelocation(MachineRelocation &MR,
                                         unsigned FromIdx,
                                         unsigned ToAddr,
                                         unsigned ToIdx,
                                         OutputBuffer &RelocOut,
                                         OutputBuffer &SecOut,
                                         bool Scattered, bool Extern) const;

    // Constants for the relocation r_type field.
    // See <mach-o/ppc/reloc.h>
    enum {
      PPC_RELOC_VANILLA, // generic relocation
      PPC_RELOC_PAIR,    // the second relocation entry of a pair
      PPC_RELOC_BR14,    // 14 bit branch displacement to word address
      PPC_RELOC_BR24,    // 24 bit branch displacement to word address
      PPC_RELOC_HI16,    // a PAIR follows with the low 16 bits
      PPC_RELOC_LO16,    // a PAIR follows with the high 16 bits
      PPC_RELOC_HA16,    // a PAIR follows, which is sign extended to 32b
      PPC_RELOC_LO14     // LO16 with low 2 bits implicitly zero
    };
  };

} // end llvm namespace

#endif // PPC_MACHO_WRITER_INFO_H

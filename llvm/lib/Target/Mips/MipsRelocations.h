//===-- MipsRelocations.h - Mips Code Relocations ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Mips target-specific relocation types
// (for relocation-model=static).
//
//===----------------------------------------------------------------------===//

#ifndef MIPSRELOCATIONS_H_
#define MIPSRELOCATIONS_H_

#include "llvm/CodeGen/MachineRelocation.h"

namespace llvm {
  namespace Mips{
    enum RelocationType {
      // reloc_mips_pc16 - pc relative relocation for branches. The lower 18
      // bits of the difference between the branch target and the branch
      // instruction, shifted right by 2.
      reloc_mips_pc16 = 1,

      // reloc_mips_hi - upper 16 bits of the address (modified by +1 if the
      // lower 16 bits of the address is negative).
      reloc_mips_hi = 2,

      // reloc_mips_lo - lower 16 bits of the address.
      reloc_mips_lo = 3,

      // reloc_mips_26 - lower 28 bits of the address, shifted right by 2.
      reloc_mips_26 = 4
    };
  }
}

#endif /* MIPSRELOCATIONS_H_ */

//===-- PPCRelocations.h - PPC Code Relocations -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PowerPC 32-bit target-specific relocation types.
//
//===----------------------------------------------------------------------===//

#ifndef PPCRELOCATIONS_H
#define PPCRELOCATIONS_H

#include "llvm/CodeGen/MachineRelocation.h"

// Hack to rid us of a PPC pre-processor symbol which is erroneously
// defined in a PowerPC header file (bug in Linux/PPC)
#ifdef PPC
#undef PPC
#endif

namespace llvm {
  namespace PPC {
    enum RelocationType {
      // reloc_vanilla - A standard relocation, where the address of the
      // relocated object completely overwrites the address of the relocation.
      reloc_vanilla,
    
      // reloc_pcrel_bx - PC relative relocation, for the b or bl instructions.
      reloc_pcrel_bx,

      // reloc_pcrel_bcx - PC relative relocation, for BLT,BLE,BEQ,BGE,BGT,BNE,
      // and other bcx instructions.
      reloc_pcrel_bcx,

      // reloc_absolute_high - Absolute relocation, for the loadhi instruction
      // (which is really addis).  Add the high 16-bits of the specified global
      // address into the low 16-bits of the instruction.
      reloc_absolute_high,

      // reloc_absolute_low - Absolute relocation, for the la instruction (which
      // is really an addi).  Add the low 16-bits of the specified global
      // address into the low 16-bits of the instruction.
      reloc_absolute_low,
      
      // reloc_absolute_low_ix - Absolute relocation for the 64-bit load/store
      // instruction which have two implicit zero bits.
      reloc_absolute_low_ix
    };
  }
}

#endif

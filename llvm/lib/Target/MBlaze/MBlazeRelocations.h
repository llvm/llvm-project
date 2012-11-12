//===-- MBlazeRelocations.h - MBlaze Code Relocations -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MBlaze target-specific relocation types.
//
//===----------------------------------------------------------------------===//

#ifndef MBLAZERELOCATIONS_H
#define MBLAZERELOCATIONS_H

#include "llvm/CodeGen/MachineRelocation.h"

namespace llvm {
  namespace MBlaze {
    enum RelocationType {
      /// reloc_pcrel_word - PC relative relocation, add the relocated value to
      /// the value already in memory, after we adjust it for where the PC is.
      reloc_pcrel_word = 0,

      /// reloc_picrel_word - PIC base relative relocation, add the relocated
      /// value to the value already in memory, after we adjust it for where the
      /// PIC base is.
      reloc_picrel_word = 1,

      /// reloc_absolute_word - absolute relocation, just add the relocated
      /// value to the value already in memory.
      reloc_absolute_word = 2,

      /// reloc_absolute_word_sext - absolute relocation, just add the relocated
      /// value to the value already in memory. In object files, it represents a
      /// value which must be sign-extended when resolving the relocation.
      reloc_absolute_word_sext = 3,

      /// reloc_absolute_dword - absolute relocation, just add the relocated
      /// value to the value already in memory.
      reloc_absolute_dword = 4
    };
  }
}

#endif

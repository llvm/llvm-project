//===- X86Relocations.h - X86 Code Relocations ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the X86 target-specific relocation types.
//
//===----------------------------------------------------------------------===//

#ifndef X86RELOCATIONS_H
#define X86RELOCATIONS_H

#include "llvm/CodeGen/MachineRelocation.h"

namespace llvm {
  namespace X86 {
    /// RelocationType - An enum for the x86 relocation codes. Note that
    /// the terminology here doesn't follow x86 convention - word means
    /// 32-bit and dword means 64-bit.
    enum RelocationType {
      // reloc_pcrel_word - PC relative relocation, add the relocated value to
      // the value already in memory, after we adjust it for where the PC is.
      reloc_pcrel_word = 0,

      // reloc_picrel_word - PIC base relative relocation, add the relocated
      // value to the value already in memory, after we adjust it for where the
      // PIC base is.
      reloc_picrel_word = 1,
      
      // reloc_absolute_word, reloc_absolute_dword - Absolute relocation, just
      // add the relocated value to the value already in memory.
      reloc_absolute_word = 2,
      reloc_absolute_dword = 3
    };
  }
}

#endif

//===-- ARMRelocations.h - ARM Code Relocations -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ARM target-specific relocation types.
//
//===----------------------------------------------------------------------===//

#ifndef ARMRELOCATIONS_H
#define ARMRELOCATIONS_H

#include "llvm/CodeGen/MachineRelocation.h"

namespace llvm {
  namespace ARM {
    enum RelocationType {
      // reloc_arm_absolute - Absolute relocation, just add the relocated value
      // to the value already in memory.
      reloc_arm_absolute,

      // reloc_arm_relative - PC relative relocation, add the relocated value to
      // the value already in memory, after we adjust it for where the PC is.
      reloc_arm_relative,

      // reloc_arm_cp_entry - PC relative relocation for constpool_entry's whose
      // addresses are kept locally in a map.
      reloc_arm_cp_entry,

      // reloc_arm_vfp_cp_entry - Same as reloc_arm_cp_entry except the offset
      // should be divided by 4.
      reloc_arm_vfp_cp_entry,

      // reloc_arm_machine_cp_entry - Relocation of a ARM machine constantpool
      // entry.
      reloc_arm_machine_cp_entry,

      // reloc_arm_jt_base - PC relative relocation for jump tables whose
      // addresses are kept locally in a map.
      reloc_arm_jt_base,

      // reloc_arm_pic_jt - PIC jump table entry relocation: dest bb - jt base.
      reloc_arm_pic_jt,

      // reloc_arm_branch - Branch address relocation.
      reloc_arm_branch,

      // reloc_arm_movt  - MOVT immediate relocation.
      reloc_arm_movt,

      // reloc_arm_movw  - MOVW immediate relocation.
      reloc_arm_movw
    };
  }
}

#endif


//===-- Next32FixupKinds.h - Next32 Specific Fixup Entries ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NEXT32_MCTARGETDESC_NEXT32FIXUPKINDS_H
#define LLVM_LIB_TARGET_NEXT32_MCTARGETDESC_NEXT32FIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace Next32 {
enum Fixups {
  reloc_4byte_mem_high = FirstTargetFixupKind,
  reloc_4byte_mem_low,
  reloc_4byte_sym_bb_imm,
  reloc_4byte_sym_function,
  reloc_4byte_func_high,
  reloc_4byte_func_low,
  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
}
} // namespace llvm

#endif

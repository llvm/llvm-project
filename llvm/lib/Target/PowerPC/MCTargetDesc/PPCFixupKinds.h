//===-- PPCFixupKinds.h - PPC Specific Fixup Entries ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PPC_PPCFIXUPKINDS_H
#define LLVM_PPC_PPCFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace PPC {
enum Fixups {
  // fixup_ppc_br24 - 24-bit PC relative relocation for direct branches like 'b'
  // and 'bl'.
  fixup_ppc_br24 = FirstTargetFixupKind,
  
  /// fixup_ppc_brcond14 - 14-bit PC relative relocation for conditional
  /// branches.
  fixup_ppc_brcond14,
  
  /// fixup_ppc_lo16 - A 16-bit fixup corresponding to lo16(_foo) for instrs
  /// like 'li'.
  fixup_ppc_lo16,
  
  /// fixup_ppc_ha16 - A 16-bit fixup corresponding to ha16(_foo) for instrs
  /// like 'lis'.
  fixup_ppc_ha16,
  
  /// fixup_ppc_lo14 - A 14-bit fixup corresponding to lo16(_foo) for instrs
  /// like 'std'.
  fixup_ppc_lo14,

  /// fixup_ppc_toc - Insert value of TOC base (.TOC.).
  fixup_ppc_toc,

  /// fixup_ppc_toc16 - A 16-bit signed fixup relative to the TOC base.
  fixup_ppc_toc16,

  /// fixup_ppc_toc16_ds - A 14-bit signed fixup relative to the TOC base with
  /// implied 2 zero bits
  fixup_ppc_toc16_ds,
  
  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
}
}

#endif

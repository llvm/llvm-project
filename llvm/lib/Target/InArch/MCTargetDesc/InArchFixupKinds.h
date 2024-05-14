#ifndef LLVM_LIB_TARGET_INARCH_MCTARGETDESC_INARCHFIXUPKINDS_H
#define LLVM_LIB_TARGET_INARCH_MCTARGETDESC_INARCHFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace InArch {
  // Although most of the current fixup types reflect a unique relocation
  // one can have multiple fixup types for a given relocation and thus need
  // to be uniquely named.
  //
  // This table *must* be in the same order of
  // MCFixupKindInfo Infos[InArch::NumTargetFixupKinds]
  // in InArchAsmBackend.cpp.
  //
  enum Fixups {
    fixup_InArch_PC16 = FirstTargetFixupKind,
    // Marker
    LastTargetFixupKind,
    NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
  };
} // namespace InArch
} // namespace llvm


#endif
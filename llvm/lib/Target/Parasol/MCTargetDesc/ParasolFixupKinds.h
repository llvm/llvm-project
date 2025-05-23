#ifndef LLVM_LIB_TARGET_PARASOL_MCTARGETDESC_PARASOLFIXUPKINDS_H
#define LLVM_LIB_TARGET_PARASOL_MCTARGETDESC_PARASOLFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm::Parasol {
enum Fixups {
  fixup_br_one_reg_imm = FirstTargetFixupKind,
  fixup_br_imm,

  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
} // end namespace llvm::Parasol

#endif // LLVM_LIB_TARGET_PARASOL_MCTARGETDESC_PARASOLFIXUPKINDS_H

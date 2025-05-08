//===-- ParasolMCAsmInfo.h - Parasol Asm Info -----------------------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the ParasolMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PARASOL_MCTARGETDESC_PARASOLMCASMINFO_H
#define LLVM_LIB_TARGET_PARASOL_MCTARGETDESC_PARASOLMCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {
class Triple;

class ParasolMCAsmInfo : public MCAsmInfoELF {
  void anchor() override;

public:
  explicit ParasolMCAsmInfo(const Triple &TheTriple);
};

} // namespace llvm

#endif // end LLVM_LIB_TARGET_PARASOL_MCTARGETDESC_PARASOLMCASMINFO_H

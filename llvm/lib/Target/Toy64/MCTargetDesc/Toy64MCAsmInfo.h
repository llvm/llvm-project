//===-- Toy64MCAsmInfo.h - Toy64 Assembly Properties ----------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_TOY64_MCTARGETDESC_TOY64MCASMINFO_H
#define LLVM_LIB_TARGET_TOY64_MCTARGETDESC_TOY64MCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {

class Triple;

class Toy64MCAsmInfo : public MCAsmInfoELF {
  void anchor() override;

public:
  explicit Toy64MCAsmInfo(const Triple &TT,
                          const MCTargetOptions &Options);
};

} // namespace llvm

#endif
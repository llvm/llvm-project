#ifndef LLVM_LIB_TARGET_SC32_MCTARGETDESC_SC32MCASMINFO_H
#define LLVM_LIB_TARGET_SC32_MCTARGETDESC_SC32MCASMINFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {

class SC32MCAsmInfo : public MCAsmInfo {
public:
  SC32MCAsmInfo();

  void printSwitchToSection(const MCSection &Section, uint32_t Subsection,
                            const Triple &T, raw_ostream &OS) const override;
};

} // namespace llvm

#endif

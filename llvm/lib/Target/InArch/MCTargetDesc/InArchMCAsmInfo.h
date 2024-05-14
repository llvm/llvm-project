#ifndef LLVM_LIB_TARGET_INARCH_MCTARGETDESC_INARCHMCASMINFO_H
#define LLVM_LIB_TARGET_INARCH_MCTARGETDESC_INARCHMCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {

class Triple;

class InArchELFMCAsmInfo : public MCAsmInfoELF {
public:
  explicit InArchELFMCAsmInfo(const Triple &TheTriple);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_SIM_MCTARGETDESC_SIMMCASMINFO_H
#ifndef LLVM_LIB_TARGET_GENPAY_GENPAYINSTRINFO_H
#define LLVM_LIB_TARGET_GENPAY_GENPAYINSTRINFO_H

#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "GenpayGenInstrInfo.inc"

namespace llvm {

class GenpayInstrInfo : public GenpayGenInstrInfo {
public:
  explicit GenpayInstrInfo();
};

} // namespace llvm

#endif

#ifndef LLVM_LIB_TARGET_GENPAY_MCTARGETDESC_GENPAYMCTARGETDESC_H
#define LLVM_LIB_TARGET_GENPAY_MCTARGETDESC_GENPAYMCTARGETDESC_H

#define GET_REGINFO_ENUM
#include "GenpayGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#include "GenpayGenInstrInfo.inc"

namespace llvm {

class Target;

Target &getTheGenpayTarget();

} // namespace llvm

#endif

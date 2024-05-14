#ifndef LLVM_LIB_TARGET_INARCH_INARCHTARGETSTREAMER_H
#define LLVM_LIB_TARGET_INARCH_INARCHTARGETSTREAMER_H

#include "llvm/MC/MCStreamer.h"

namespace llvm {

class InArchTargetStreamer : public MCTargetStreamer {
public:
  InArchTargetStreamer(MCStreamer &S);
  ~InArchTargetStreamer() override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_INARCH_INARCHTARGETSTREAMER_H
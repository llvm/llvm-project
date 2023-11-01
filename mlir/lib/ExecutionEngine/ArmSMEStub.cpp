
#include "llvm/Support/Compiler.h"

extern "C" {

bool LLVM_ATTRIBUTE_WEAK __aarch64_sme_accessible() {
  // The ArmSME tests are run within an emulator so we assume SME is available.
  return true;
}
}

#include "llvm/Support/MitigationMarker.h"

namespace llvm {

///
/// Map a MitigationKey to a string
///
const llvm::DenseMap<MitigationKey, StringRef> &GetMitigationMetadataMapping() {
  static const llvm::DenseMap<MitigationKey, StringRef> MitigationToString = {
      {MitigationKey::AUTO_VAR_INIT, "security_mitigation_auto-var-init"},
      {MitigationKey::STACK_CLASH_PROTECTION,
       "security_mitigation_stack-clash-protection"},
      {MitigationKey::STACK_PROTECTOR, "security_mitigation_stack-protector"},
      {MitigationKey::STACK_PROTECTOR_STRONG,
       "security_mitigation_stack-protector-strong"},
      {MitigationKey::STACK_PROTECTOR_ALL,
       "security_mitigation_stack-protector-all"},
      {MitigationKey::CFI_ICALL, "security_mitigation_cfi-icall"},
      {MitigationKey::CFI_VCALL, "security_mitigation_cfi-vcall"},
      {MitigationKey::CFI_NVCALL, "security_mitigation_cfi-nvcall"},
  };
  return MitigationToString;
}

} // namespace llvm

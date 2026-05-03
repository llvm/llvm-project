#ifndef LLVM_SUPPORT_UNDEFPOISONKIND_H
#define LLVM_SUPPORT_UNDEFPOISONKIND_H

namespace llvm {

/// Enumeration of the different types of "undefined" values in LLVM.
enum class UndefPoisonKind {
  PoisonOnly = (1 << 0),
  UndefOnly = (1 << 1),
  UndefOrPoison = PoisonOnly | UndefOnly,
  LLVM_MARK_AS_BITMASK_ENUM(UndefOrPoison)
};

} // end namespace llvm

#endif // LLVM_SUPPORT_UNDEFPOISONKIND_H
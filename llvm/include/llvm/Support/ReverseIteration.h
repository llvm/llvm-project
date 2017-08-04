#ifndef LLVM_SUPPORT_REVERSEITERATION_H
#define LLVM_SUPPORT_REVERSEITERATION_H

#include "llvm/Config/abi-breaking.h"

namespace llvm {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
template <class T = void> struct ReverseIterate { static bool value; };
#if LLVM_ENABLE_REVERSE_ITERATION
template <class T> bool ReverseIterate<T>::value = true;
#else
template <class T> bool ReverseIterate<T>::value = false;
#endif
#endif
}

#endif

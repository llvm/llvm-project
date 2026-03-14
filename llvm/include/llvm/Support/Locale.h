#ifndef LLVM_SUPPORT_LOCALE_H
#define LLVM_SUPPORT_LOCALE_H

#include "llvm/Support/Compiler.h"

namespace llvm {
class StringRef;

namespace sys {
namespace locale {

LLVM_ABI int columnWidth(StringRef s);
LLVM_ABI bool isPrint(int c);
}
}
}

#endif // LLVM_SUPPORT_LOCALE_H

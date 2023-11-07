#ifndef LLVM_SUPPORT_LOCALE_H
#define LLVM_SUPPORT_LOCALE_H

#include "llvm/Support/Compiler.h"

namespace llvm {
class StringRef;

namespace sys {
namespace locale {

LLVM_FUNC_ABI int columnWidth(StringRef s);
LLVM_FUNC_ABI bool isPrint(int c);

}
}
}

#endif // LLVM_SUPPORT_LOCALE_H

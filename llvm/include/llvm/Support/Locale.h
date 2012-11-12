#ifndef LLVM_SUPPORT_LOCALE
#define LLVM_SUPPORT_LOCALE

#include "llvm/ADT/StringRef.h"

namespace llvm {
namespace sys {
namespace locale {

int columnWidth(StringRef s);
bool isPrint(int c);

}
}
}

#endif // LLVM_SUPPORT_LOCALE

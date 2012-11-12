#include "llvm/Support/Locale.h"
#include "llvm/Config/config.h"

#ifdef __APPLE__
#include "LocaleXlocale.inc"
#elif LLVM_ON_WIN32
#include "LocaleWindows.inc"
#else
#include "LocaleGeneric.inc"
#endif

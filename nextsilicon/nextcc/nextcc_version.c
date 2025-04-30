#include "nextcc/version.h"

#ifdef __MACH__
#define NEXTCC_SECTION "__const," NEXTCC_SECTION_NAME
#else
#define NEXTCC_SECTION NEXTCC_SECTION_NAME
#endif

__attribute__((used)) __attribute__((section(NEXTCC_SECTION)))
__attribute__((visibility("hidden"))) const char nextcc_version[] =
    NEXTCC_VERSION_STRING;

#undef NEXTCC_SECTION

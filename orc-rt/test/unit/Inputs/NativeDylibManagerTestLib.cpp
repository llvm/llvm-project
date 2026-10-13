// A minimal shared library for NativeDylibManager tests.

#include "TestVisibility.h"

extern "C" TEST_EXPORT int NativeDylibManagerTestFunc() { return 42; }
extern "C" TEST_EXPORT int NativeDylibManagerTestFunc2() { return 7; }

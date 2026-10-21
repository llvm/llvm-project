#include "dylib.h"

int present_weak_function() { return 10; }

#if defined(HAS_THEM)
int absent_weak_function() { return 15; }
#endif

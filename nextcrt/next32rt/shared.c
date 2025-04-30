#include "atexit.h"
#include <stdlib.h>

__attribute__((visibility("hidden"))) const void *const __dso_handle = &__dso_handle;

#ifndef atexit
__attribute__ ((visibility ("hidden")))
#endif
int atexit(void (*fn)(void))
{
    return __cxa_atexit((void (*)(void *))fn, NULL, (void *)&__dso_handle);
}

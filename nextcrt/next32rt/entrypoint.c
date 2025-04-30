#include "atexit.h"
#include "nextcrt/global_tables.h"
#include "nextcrt/runtime.h"
#include <stdlib.h>

const void *const __dso_handle = &__dso_handle;

/* These magic symbols are provided by the linker.  */
extern void (*__init_array_start[])(int argc, char **argv);
extern void (*__init_array_end[])(int argc, char **argv);

/* In order to allow calls with variad number of variables, call main() using
 * a function pointer
 */
extern int main(int argc, char **argv);

static void invoke_init_array(int argc, char **argv)
{
    const size_t count = __init_array_end - __init_array_start;
    unsigned int i;

    for (i = 0; i < count; i++)
        (*__init_array_start[i])(argc, argv);
}

#ifndef atexit
__attribute__ ((visibility ("hidden")))
#endif
int atexit(void (*fn)(void))
{
    return __cxa_atexit((void (*)(void *))fn, NULL, NULL);
}

int _start()
{
    int (*volatile main_ptr)(int argc, char **argv) = &main;
    const struct __next32_process_info *info;

    info = &__next32_process_table[__next32_process_index()];

    /* Call contents of INIT_ARRAY */
    invoke_init_array(info->argc, info->argv);

    /* Call main() with the command-line arguments */
    return (*main_ptr)(info->argc, info->argv);
}

/**
 * Workaround for issue https://nextsilicon.atlassian.net/browse/SOF-832
 *
 * LLVM6's linker does not keep the INIT_ARRAY if there is no construction
 * function in the code. This makes codegraph fail when loading the _start()
 * function as it needs the __init_array_start and __init_array_end symbols that
 * are relocated to the start and end of the INIT_ARRAY.
 *
 * As a workaround, a constructor function is declared here.
 */
volatile int __llvm6_missing_symbol_workaround_i;
__attribute__((constructor)) void __llvm6_missing_symbol_workaround(void)
{
    __llvm6_missing_symbol_workaround_i++;
}

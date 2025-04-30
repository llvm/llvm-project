#ifndef __RUNTIME_H__
#define __RUNTIME_H__

#include "system_variables.h"

#define BITMASK(bits) ((1 << bits) - 1)

#define __next32_thread_index()                                                                    \
    ((__builtin_next32_threadid() >> __next32_tid_shift) & BITMASK(__next32_tid_bits))

#define __next32_process_index()                                                                   \
    ((__builtin_next32_threadid() >> __next32_pid_shift) & BITMASK(__next32_pid_bits))

#endif /* __RUNTIME_H__ */

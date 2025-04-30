#pragma once

#include <stdint.h>

/* Prevent C++ name mangling */
#ifdef __cplusplus
extern "C" {
#endif

int nsapi_multiprocess_set(uint32_t proc_index, uint32_t total_processes);

#ifdef __cplusplus
}
#endif

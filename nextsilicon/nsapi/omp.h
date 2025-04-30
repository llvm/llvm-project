#pragma once

#include <stdint.h>

/* Prevent C++ name mangling */
#ifdef __cplusplus
extern "C" {
#endif

int nsapi_omp_set_num_threads(uint32_t num_threads);

int nsapi_omp_set_dynamic(uint32_t is_dynamic);

#ifdef __cplusplus
}
#endif

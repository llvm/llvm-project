#ifndef __NEXT32_BUILTIN_GLOBAL_TABLES_STRUCTURE_H__
#define __NEXT32_BUILTIN_GLOBAL_TABLES_STRUCTURE_H__

#include "next_memory_structure.h"
#include <stdint.h>

struct __next32_thread_start {
    uint64_t start_func;
    void *thread_lib_data;
};

struct __next32_module_base {
    void *base;
};

/**
 * Process control block
 *
 * @module_base: Pointer to the module base table
 * @argc: Number of command-line arguments
 * @argv: Array of command-line arguments
 * @memory_ctx: Process-wide context of the nextmalloc memory allocator
 * @process_index: Index of the process in a multi-processing team
 * @total_processes: Multi-processing team size
 */
struct __next32_process_info {
    union {
        struct {
            int argc;
            char **argv;
            struct next_memory_context memory_ctx;
            int team_size;
            int dynamic;
            uint16_t process_index;
            uint16_t total_processes;
        };
        struct {
            /* Padding to ensure size is a power of 2 */
            char pad[32];
        };
    };
};

/**
 * Thread control block
 *
 * @stack_pointer: Pointer to the thread's stack
 * @start_data: Data to use when thread starts
 * @affinity_idx: Index for this thread in the affinity vector (-1 not set)
 * @romp_runtime: Information set only for RISC OpenMP worker threads
 */
struct __next32_thread_info {
    union {
        struct {
            /* the thread stack pointer must be dereferenceable both
             * on the device and on each individual process handled
             * by next-silicon
             * Note: Each stack ptr only has to be dereferenceable
             *       solely on *its* respective host process and
             *       not necessarily on all processes...
             */
            void *stack_pointer;
            struct __next32_thread_start *thread_start_data;
            void *stack_base;
            int affinity_idx;
        };
        struct {
            /* Padding to ensure size is a power of 2 */
            char pad[64];
        };
    };
    /* Todo: For debug-stack validations, add base and size */
    /* Todo: For exception handling, add frame-stack (ut_stack?) */
};

#endif /* __NEXT32_BUILTIN_GLOBAL_TABLES_STRUCTURE_H__ */

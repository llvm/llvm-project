#ifndef __NEXT32_BUILTIN_GLOBAL_TABLES_H__
#define __NEXT32_BUILTIN_GLOBAL_TABLES_H__

#include "global_tables_structure.h"

/**
 * A global (cross-process) table containing the process control blocks
 */
extern struct __next32_process_info *const __next32_process_table;

/**
 * 2D array of module bases.
 * Each row contains the bases of one module for each process.
 */
extern const struct __next32_module_base *const *const __next32_module_table;

/**
 * A global (cross-process) table containing the thread control blocks
 */
extern struct __next32_thread_info *const __next32_thread_table;

#endif /* __NEXT32_BUILTIN_GLOBAL_TABLES_H__ */

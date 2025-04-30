/**
 *  NSAPI intrinsic declarations. The definitions are stubs - the calls will be
 * replaced in the codegraph to the appropriate values
 *
 * NOTE: DO NOT MODIFY THE FUNCTIONS IN THIS HEADER IN ANY WAY!
 * Those functions have special names that are checked at runtime stage,
 * and changed to appropriate immediate values. Changing function names,
 * implementations, attributes or signatures may result in unexpected behaviour.
 *
 * Note about attributes: all intrinsic functions are marked by two attributes,
 * `optnone` and `weak`.
 * The `optnone` attribute prevents inlining or otherwise altering the function
 * in any way by the compiler, so the intrinsic symbols will be present and the
 * name, signature and implementation will be exactly as described here,
 * regardless of compilation flags (e.g. -O3). The 'weak' attribute prevents
 * linker errors of duplicate definition of the functions, when included in
 * different translation units. It essentially mimics the behaviour of C++
 * function definitions marked `inline` in a header file. This makes it possible
 * to have the definitions in the header file, without errors or duplications
 * when compiling to C or C++, so using intrinsics does not require libnsapi.so.
 */

#pragma once

/* Prevent C++ name mangling */
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <nextcrt/global_tables_structure.h>

#define BITMASK(bits) ((1u << bits) - 1)

/**
 * Checks if the caller is running on the codegraph.
 *
 * Returns 0 if the caller is NOT on codegraph (i.e. native/host), a non-zero
 * value if the caller is codegraph (emu/lifter/grid etc.). This function can be
 * used in an `if` statement to single out NextSilicon device-specific code
 * (e.g. code that uses the intrinsics below).
 */
__attribute__((noinline)) __attribute__((optnone)) __attribute__((weak)) int
__nsapi_is_on_cg(void) {
  return 0;
}

/* The intrinsics below will ABORT if called from the host
 * This is done to prevent any misuse of the intrinsics, as they MUST be called
 * from a CG context to have valid values. It is recommended to only call NSAPI
 * intrinsics inside a conditional if (__nsapi_is_on_cg()) block
 */

/**
 * Returns the unique NextSilicon process/thread identifier (merged into a
 * single 32-bit word)
 * NOTE: It is recommended to use the __nsapi_process_index/__nsapi_thread_index
 * functions below to get just the TID or PID parts
 */
__attribute__((optnone)) __attribute__((weak)) uint32_t
__nsapi_get_ns_raw_tid(void) {
  abort();
}

/**
 * Returns the address of the process table
 *
 * The process table is an array of __next32_process_info structs. The array
 * indices are the NS process IDs.
 */
__attribute__((optnone)) __attribute__((weak)) struct __next32_process_info *
__nsapi_get_process_table(void) {
  abort();
}

/**
 * Returns the address of the thread table
 *
 * The thread table is an array of __next32_thread_info structs. The array
 * indices are the NS thread IDs.
 */
__attribute__((optnone)) __attribute__((weak)) struct __next32_thread_info *
__nsapi_get_thread_table(void) {
  abort();
}

/**
 * Returns the address of the module table
 *
 * The module table is an array of __next32_module_base structs.
 */
__attribute__((optnone)) __attribute__((weak)) struct __next32_module_base *
__nsapi_get_module_table(void) {
  abort();
}

/* Builtin value intrinsics - helpers for NS tid/pid calculation */

__attribute__((optnone)) __attribute__((weak)) uint32_t
__nsapi_get_tid_bits(void) {
  abort();
}

__attribute__((optnone)) __attribute__((weak)) uint32_t
__nsapi_get_tid_shift(void) {
  abort();
}

__attribute__((optnone)) __attribute__((weak)) uint32_t
__nsapi_get_pid_bits(void) {
  abort();
}

__attribute__((optnone)) __attribute__((weak)) uint32_t
__nsapi_get_pid_shift(void) {
  abort();
}

/* Helper functions to single out NextSilicon TID or PID from the raw identifier
 */

/**
 * Extract the process ID from the full NS-PTID of the current thread
 */
inline __attribute__((always_inline)) uint32_t __nsapi_process_index(void) {
  return (__nsapi_get_ns_raw_tid() >> __nsapi_get_pid_shift()) &
         BITMASK(__nsapi_get_pid_bits());
}

/**
 * Extract the thread ID from the full NS-PTID of the current thread
 */
inline __attribute__((always_inline)) uint32_t __nsapi_thread_index(void) {
  return (__nsapi_get_ns_raw_tid() >> __nsapi_get_tid_shift()) &
         BITMASK(__nsapi_get_tid_bits());
}

/**
 * Extract the process ID from the full NS-PTID of an arbitrary NS thread
 */
inline __attribute__((always_inline)) uint32_t
__nsapi_process_index_of(uint32_t raw_ptid) {
  return (raw_ptid >> __nsapi_get_pid_shift()) &
         BITMASK(__nsapi_get_pid_bits());
}

/**
 * Extract the thread ID from the full NS-PTID of an arbitrary NS thread
 */
inline __attribute__((always_inline)) uint32_t
__nsapi_thread_index_of(uint32_t raw_ptid) {
  return (raw_ptid >> __nsapi_get_tid_shift()) &
         BITMASK(__nsapi_get_tid_bits());
}

/* NextSilicon device thread getters */

/**
 * Returns the index of the thread in its team.
 * NOTE: Only relevant for threads spawned on the device as part of a team.
 */
uint32_t __nsapi_team_get_thread_index(void);

/**
 * Returns the size of the current thread's team.
 * NOTE: Only relevant for threads spawned on the device as part of a team.
 */
uint32_t __nsapi_team_get_team_size(void);

#ifdef __cplusplus
} // extern "C"
#endif

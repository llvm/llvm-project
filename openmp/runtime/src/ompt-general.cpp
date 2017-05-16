/*****************************************************************************
 * system include files
 ****************************************************************************/

#include <assert.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*****************************************************************************
 * ompt include files
 ****************************************************************************/

#include "ompt-specific.cpp"

/*****************************************************************************
 * macros
 ****************************************************************************/

#define ompt_get_callback_success 1
#define ompt_get_callback_failure 0

#define no_tool_present 0

#define OMPT_API_ROUTINE static

#ifndef OMPT_STR_MATCH
#define OMPT_STR_MATCH(haystack, needle) (!strcasecmp(haystack, needle))
#endif

/*****************************************************************************
 * types
 ****************************************************************************/

typedef struct {
  const char *state_name;
  ompt_state_t state_id;
} ompt_state_info_t;

enum tool_setting_e {
  omp_tool_error,
  omp_tool_unset,
  omp_tool_disabled,
  omp_tool_enabled
};

typedef void (*ompt_initialize_t)(ompt_function_lookup_t ompt_fn_lookup,
                                  const char *version,
                                  unsigned int ompt_version);

/*****************************************************************************
 * global variables
 ****************************************************************************/

int ompt_enabled = 0;

ompt_state_info_t ompt_state_info[] = {
#define ompt_state_macro(state, code) {#state, state},
    FOREACH_OMPT_STATE(ompt_state_macro)
#undef ompt_state_macro
};

ompt_callbacks_t ompt_callbacks;

static ompt_initialize_t ompt_initialize_fn = NULL;

/*****************************************************************************
 * forward declarations
 ****************************************************************************/

static ompt_interface_fn_t ompt_fn_lookup(const char *s);

OMPT_API_ROUTINE ompt_thread_id_t ompt_get_thread_id(void);

/*****************************************************************************
 * initialization and finalization (private operations)
 ****************************************************************************/

/* On Unix-like systems that support weak symbols the following implementation
 * of ompt_tool() will be used in case no tool-supplied implementation of
 * this function is present in the address space of a process.
 *
 * On Windows, the ompt_tool_windows function is used to find the
 * ompt_tool symbol across all modules loaded by a process. If ompt_tool is
 * found, ompt_tool's return value is used to initialize the tool. Otherwise,
 * NULL is returned and OMPT won't be enabled */
#if OMPT_HAVE_WEAK_ATTRIBUTE
_OMP_EXTERN
__attribute__((weak)) ompt_initialize_t ompt_tool() {
#if OMPT_DEBUG
  printf("ompt_tool() is called from the RTL\n");
#endif
  return NULL;
}

#elif OMPT_HAVE_PSAPI

#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#define ompt_tool ompt_tool_windows

// The number of loaded modules to start enumeration with EnumProcessModules()
#define NUM_MODULES 128

static ompt_initialize_t ompt_tool_windows() {
  int i;
  DWORD needed, new_size;
  HMODULE *modules;
  HANDLE process = GetCurrentProcess();
  modules = (HMODULE *)malloc(NUM_MODULES * sizeof(HMODULE));
  ompt_initialize_t (*ompt_tool_p)() = NULL;

#if OMPT_DEBUG
  printf("ompt_tool_windows(): looking for ompt_tool\n");
#endif
  if (!EnumProcessModules(process, modules, NUM_MODULES * sizeof(HMODULE),
                          &needed)) {
    // Regardless of the error reason use the stub initialization function
    free(modules);
    return NULL;
  }
  // Check if NUM_MODULES is enough to list all modules
  new_size = needed / sizeof(HMODULE);
  if (new_size > NUM_MODULES) {
#if OMPT_DEBUG
    printf("ompt_tool_windows(): resize buffer to %d bytes\n", needed);
#endif
    modules = (HMODULE *)realloc(modules, needed);
    // If resizing failed use the stub function.
    if (!EnumProcessModules(process, modules, needed, &needed)) {
      free(modules);
      return NULL;
    }
  }
  for (i = 0; i < new_size; ++i) {
    (FARPROC &)ompt_tool_p = GetProcAddress(modules[i], "ompt_tool");
    if (ompt_tool_p) {
#if OMPT_DEBUG
      TCHAR modName[MAX_PATH];
      if (GetModuleFileName(modules[i], modName, MAX_PATH))
        printf("ompt_tool_windows(): ompt_tool found in module %s\n", modName);
#endif
      free(modules);
      return ompt_tool_p();
    }
#if OMPT_DEBUG
    else {
      TCHAR modName[MAX_PATH];
      if (GetModuleFileName(modules[i], modName, MAX_PATH))
        printf("ompt_tool_windows(): ompt_tool not found in module %s\n",
               modName);
    }
#endif
  }
  free(modules);
  return NULL;
}
#else
#error Either __attribute__((weak)) or psapi.dll are required for OMPT support
#endif // OMPT_HAVE_WEAK_ATTRIBUTE

void ompt_pre_init() {
  //--------------------------------------------------
  // Execute the pre-initialization logic only once.
  //--------------------------------------------------
  static int ompt_pre_initialized = 0;

  if (ompt_pre_initialized)
    return;

  ompt_pre_initialized = 1;

  //--------------------------------------------------
  // Use a tool iff a tool is enabled and available.
  //--------------------------------------------------
  const char *ompt_env_var = getenv("OMP_TOOL");
  tool_setting_e tool_setting = omp_tool_error;

  if (!ompt_env_var || !strcmp(ompt_env_var, ""))
    tool_setting = omp_tool_unset;
  else if (OMPT_STR_MATCH(ompt_env_var, "disabled"))
    tool_setting = omp_tool_disabled;
  else if (OMPT_STR_MATCH(ompt_env_var, "enabled"))
    tool_setting = omp_tool_enabled;

#if OMPT_DEBUG
  printf("ompt_pre_init(): tool_setting = %d\n", tool_setting);
#endif
  switch (tool_setting) {
  case omp_tool_disabled:
    break;

  case omp_tool_unset:
  case omp_tool_enabled:
    ompt_initialize_fn = ompt_tool();
    if (ompt_initialize_fn) {
      ompt_enabled = 1;
    }
    break;

  case omp_tool_error:
    fprintf(stderr, "Warning: OMP_TOOL has invalid value \"%s\".\n"
                    "  legal values are (NULL,\"\",\"disabled\","
                    "\"enabled\").\n",
            ompt_env_var);
    break;
  }
#if OMPT_DEBUG
  printf("ompt_pre_init(): ompt_enabled = %d\n", ompt_enabled);
#endif
}

void ompt_post_init() {
  //--------------------------------------------------
  // Execute the post-initialization logic only once.
  //--------------------------------------------------
  static int ompt_post_initialized = 0;

  if (ompt_post_initialized)
    return;

  ompt_post_initialized = 1;

  //--------------------------------------------------
  // Initialize the tool if so indicated.
  //--------------------------------------------------
  if (ompt_enabled) {
    ompt_initialize_fn(ompt_fn_lookup, ompt_get_runtime_version(),
                       OMPT_VERSION);

    ompt_thread_t *root_thread = ompt_get_thread();

    ompt_set_thread_state(root_thread, ompt_state_overhead);

    if (ompt_callbacks.ompt_callback(ompt_event_thread_begin)) {
      ompt_callbacks.ompt_callback(ompt_event_thread_begin)(
          ompt_thread_initial, ompt_get_thread_id());
    }

    ompt_set_thread_state(root_thread, ompt_state_work_serial);
  }
}

void ompt_fini() {
  if (ompt_enabled) {
    if (ompt_callbacks.ompt_callback(ompt_event_runtime_shutdown)) {
      ompt_callbacks.ompt_callback(ompt_event_runtime_shutdown)();
    }
  }

  ompt_enabled = 0;
}

/*****************************************************************************
 * interface operations
 ****************************************************************************/

/*****************************************************************************
 * state
 ****************************************************************************/

OMPT_API_ROUTINE int ompt_enumerate_state(int current_state, int *next_state,
                                          const char **next_state_name) {
  const static int len = sizeof(ompt_state_info) / sizeof(ompt_state_info_t);
  int i = 0;

  for (i = 0; i < len - 1; i++) {
    if (ompt_state_info[i].state_id == current_state) {
      *next_state = ompt_state_info[i + 1].state_id;
      *next_state_name = ompt_state_info[i + 1].state_name;
      return 1;
    }
  }

  return 0;
}

/*****************************************************************************
 * callbacks
 ****************************************************************************/

OMPT_API_ROUTINE int ompt_set_callback(ompt_event_t evid, ompt_callback_t cb) {
  switch (evid) {

#define ompt_event_macro(event_name, callback_type, event_id)                  \
  case event_name:                                                             \
    if (ompt_event_implementation_status(event_name)) {                        \
      ompt_callbacks.ompt_callback(event_name) = (callback_type)cb;            \
    }                                                                          \
    return ompt_event_implementation_status(event_name);

    FOREACH_OMPT_EVENT(ompt_event_macro)

#undef ompt_event_macro

  default:
    return ompt_set_result_registration_error;
  }
}

OMPT_API_ROUTINE int ompt_get_callback(ompt_event_t evid, ompt_callback_t *cb) {
  switch (evid) {

#define ompt_event_macro(event_name, callback_type, event_id)                  \
  case event_name:                                                             \
    if (ompt_event_implementation_status(event_name)) {                        \
      ompt_callback_t mycb =                                                   \
          (ompt_callback_t)ompt_callbacks.ompt_callback(event_name);           \
      if (mycb) {                                                              \
        *cb = mycb;                                                            \
        return ompt_get_callback_success;                                      \
      }                                                                        \
    }                                                                          \
    return ompt_get_callback_failure;

    FOREACH_OMPT_EVENT(ompt_event_macro)

#undef ompt_event_macro

  default:
    return ompt_get_callback_failure;
  }
}

/*****************************************************************************
 * parallel regions
 ****************************************************************************/

OMPT_API_ROUTINE ompt_parallel_id_t ompt_get_parallel_id(int ancestor_level) {
  return __ompt_get_parallel_id_internal(ancestor_level);
}

OMPT_API_ROUTINE int ompt_get_parallel_team_size(int ancestor_level) {
  return __ompt_get_parallel_team_size_internal(ancestor_level);
}

OMPT_API_ROUTINE void *ompt_get_parallel_function(int ancestor_level) {
  return __ompt_get_parallel_function_internal(ancestor_level);
}

OMPT_API_ROUTINE ompt_state_t ompt_get_state(ompt_wait_id_t *ompt_wait_id) {
  ompt_state_t thread_state = __ompt_get_state_internal(ompt_wait_id);

  if (thread_state == ompt_state_undefined) {
    thread_state = ompt_state_work_serial;
  }

  return thread_state;
}

/*****************************************************************************
 * threads
 ****************************************************************************/

OMPT_API_ROUTINE void *ompt_get_idle_frame() {
  return __ompt_get_idle_frame_internal();
}

/*****************************************************************************
 * tasks
 ****************************************************************************/

OMPT_API_ROUTINE ompt_thread_id_t ompt_get_thread_id(void) {
  return __ompt_get_thread_id_internal();
}

OMPT_API_ROUTINE ompt_task_id_t ompt_get_task_id(int depth) {
  return __ompt_get_task_id_internal(depth);
}

OMPT_API_ROUTINE ompt_frame_t *ompt_get_task_frame(int depth) {
  return __ompt_get_task_frame_internal(depth);
}

OMPT_API_ROUTINE void *ompt_get_task_function(int depth) {
  return __ompt_get_task_function_internal(depth);
}

/*****************************************************************************
 * placeholders
 ****************************************************************************/

// Don't define this as static. The loader may choose to eliminate the symbol
// even though it is needed by tools.
#define OMPT_API_PLACEHOLDER

// Ensure that placeholders don't have mangled names in the symbol table.
#ifdef __cplusplus
extern "C" {
#endif

OMPT_API_PLACEHOLDER void ompt_idle(void) {
  // This function is a placeholder used to represent the calling context of
  // idle OpenMP worker threads. It is not meant to be invoked.
  assert(0);
}

OMPT_API_PLACEHOLDER void ompt_overhead(void) {
  // This function is a placeholder used to represent the OpenMP context of
  // threads working in the OpenMP runtime.  It is not meant to be invoked.
  assert(0);
}

OMPT_API_PLACEHOLDER void ompt_barrier_wait(void) {
  // This function is a placeholder used to represent the OpenMP context of
  // threads waiting for a barrier in the OpenMP runtime. It is not meant
  // to be invoked.
  assert(0);
}

OMPT_API_PLACEHOLDER void ompt_task_wait(void) {
  // This function is a placeholder used to represent the OpenMP context of
  // threads waiting for a task in the OpenMP runtime. It is not meant
  // to be invoked.
  assert(0);
}

OMPT_API_PLACEHOLDER void ompt_mutex_wait(void) {
  // This function is a placeholder used to represent the OpenMP context of
  // threads waiting for a mutex in the OpenMP runtime. It is not meant
  // to be invoked.
  assert(0);
}

#ifdef __cplusplus
};
#endif

/*****************************************************************************
 * compatability
 ****************************************************************************/

OMPT_API_ROUTINE int ompt_get_ompt_version() { return OMPT_VERSION; }

/*****************************************************************************
 * application-facing API
 ****************************************************************************/

/*----------------------------------------------------------------------------
 | control
 ---------------------------------------------------------------------------*/

_OMP_EXTERN void ompt_control(uint64_t command, uint64_t modifier) {
  if (ompt_enabled && ompt_callbacks.ompt_callback(ompt_event_control)) {
    ompt_callbacks.ompt_callback(ompt_event_control)(command, modifier);
  }
}

/*****************************************************************************
 * API inquiry for tool
 ****************************************************************************/

static ompt_interface_fn_t ompt_fn_lookup(const char *s) {

#define ompt_interface_fn(fn)                                                  \
  if (strcmp(s, #fn) == 0)                                                     \
    return (ompt_interface_fn_t)fn;

  FOREACH_OMPT_INQUIRY_FN(ompt_interface_fn)

  FOREACH_OMPT_PLACEHOLDER_FN(ompt_interface_fn)

  return (ompt_interface_fn_t)0;
}

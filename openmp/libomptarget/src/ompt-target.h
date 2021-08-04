#ifndef LIBOMPTARGET_OMPT_TARGET_H
#define LIBOMPTARGET_OMPT_TARGET_H

#include "omp-tools.h"

#define _OMP_EXTERN extern "C"

#define OMPT_WEAK_ATTRIBUTE __attribute__((weak))

// The following structs are used to pass target-related OMPT callbacks to
// libomptarget. The structs' definitions should be in sync with the definitions
// in libomptarget/src/ompt_internal.h

/* Bitmap to mark OpenMP 5.1 target events as registered*/
typedef struct ompt_target_callbacks_active_s {
  unsigned int enabled : 1;
#define ompt_event_macro(event, callback, eventid) unsigned int event : 1;

  FOREACH_OMPT_51_TARGET_EVENT(ompt_event_macro)

#undef ompt_event_macro
} ompt_target_callbacks_active_t;

extern ompt_target_callbacks_active_t ompt_target_enabled;

_OMP_EXTERN OMPT_WEAK_ATTRIBUTE bool
libomp_start_tool(ompt_target_callbacks_active_t *libomptarget_ompt_enabled);

#endif // LIBOMPTARGET_OMPT_TARGET_H

#include "ompd-private.h"
#include "omp-debug.h"
#include <cstring>

void __ompd_init_states(const ompd_callbacks_t *table) {
  callbacks = table;
}

static const char *get_ompd_state_name(ompd_word_t state) {
  switch (state) {
#define ompd_state_macro(state, code) \
    case code: return #state ;
  FOREACH_OMPD_STATE(ompd_state_macro)
#undef ompd_state_macro
    default: return NULL;
  }
}

static const char *get_ompd_cuda_state_name(ompd_word_t state) {
  switch (state) {
    case ompt_state_work_serial:
      return "ompt_state_work_serial";
    case ompt_state_work_parallel:
      return "ompt_state_work_parallel";
    case ompt_state_work_reduction:
      return "ompt_state_work_reduction";
    default:
      return NULL;
  }
}

ompd_rc_t ompd_enumerate_states(
    ompd_address_space_handle_t *address_space_handle,
    ompd_word_t current_state, ompd_word_t *next_state,
    const char **next_state_name, ompd_word_t *more_enums) {
  ompd_rc_t ret;
  if (address_space_handle->kind == OMPD_DEVICE_KIND_CUDA) {
    // We only support a small number of states for cuda devices
    *more_enums = 1;
    switch (current_state) {
      case ompt_state_undefined:
        *next_state = ompt_state_work_serial;
        break;
      case ompt_state_work_serial:
        *next_state = ompt_state_work_parallel;
        break;
      case ompt_state_work_parallel:
        *next_state = ompt_state_work_reduction;
        *more_enums = 0;
        break;
      default:
        return ompd_rc_bad_input;
    }
    const char *find_next_state_name = get_ompd_cuda_state_name(*next_state);
    char *next_state_name_cpy;
    ret = callbacks->alloc_memory(
        strlen(find_next_state_name) + 1, (void **)&next_state_name_cpy);
    if (ret != ompd_rc_ok) {
      return ret;
    }
    strcpy(next_state_name_cpy, get_ompd_cuda_state_name(*next_state));
    *next_state_name = next_state_name_cpy;
  }  else {
    if (current_state > ompt_state_undefined &&
        current_state >= OMPD_LAST_OMP_STATE) {
      return ompd_rc_bad_input;
    }
    const char *find_next_state_name;
    *next_state = (current_state == ompt_state_undefined
                  ? ompt_state_work_serial
                  : current_state + 1);
    while (!(find_next_state_name = get_ompd_state_name(*next_state))) {
      ++(*next_state);
    }

    char *next_state_name_cpy;
    ret = callbacks->alloc_memory(strlen(find_next_state_name) + 1, (void **)&next_state_name_cpy);
    if (ret != ompd_rc_ok) {
      return ret;
    }
    strcpy(next_state_name_cpy, find_next_state_name);

    *next_state_name = next_state_name_cpy;

    if (*next_state == OMPD_LAST_OMP_STATE) {
      *more_enums = 0;
    } else {
      *more_enums = 1;
    }
  }
  return ompd_rc_ok;
}

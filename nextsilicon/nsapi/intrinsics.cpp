#include "intrinsics.h"

#pragma ns mark import_single
uint32_t __nsapi_team_get_thread_index(void) { return 0; }

#pragma ns mark import_single
uint32_t __nsapi_team_get_team_size(void) { return 1; }

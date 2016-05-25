#ifndef OMPT_SPECIFIC_H
#define OMPT_SPECIFIC_H

#include "kmp.h"

/*****************************************************************************
 * types
 ****************************************************************************/

typedef kmp_info_t ompt_thread_t;



/*****************************************************************************
 * forward declarations
 ****************************************************************************/

void __ompt_team_assign_id(kmp_team_t *team, ompt_parallel_id_t ompt_pid);
void __ompt_thread_assign_wait_id(void *variable);

void __ompt_lw_taskteam_init(ompt_lw_taskteam_t *lwt, ompt_thread_t *thr,
                             int gtid, void *microtask,
                             ompt_parallel_id_t ompt_pid);

void __ompt_lw_taskteam_link(ompt_lw_taskteam_t *lwt,  ompt_thread_t *thr);

ompt_lw_taskteam_t * __ompt_lw_taskteam_unlink(ompt_thread_t *thr);

ompt_parallel_id_t __ompt_parallel_id_new(int gtid);
ompt_task_id_t __ompt_task_id_new(int gtid);

ompt_team_info_t *__ompt_get_teaminfo(int depth, int *size);

ompt_task_info_t *__ompt_get_taskinfo(int depth);

void __ompt_thread_begin(ompt_thread_type_t thread_type, int gtid);

void __ompt_thread_end(ompt_thread_type_t thread_type, int gtid);

int __ompt_get_parallel_team_size_internal(int ancestor_level);

ompt_task_id_t __ompt_get_task_id_internal(int depth);

ompt_frame_t *__ompt_get_task_frame_internal(int depth);



/*****************************************************************************
 * macros
 ****************************************************************************/

#define OMPT_HAVE_WEAK_ATTRIBUTE KMP_HAVE_WEAK_ATTRIBUTE
#define OMPT_HAVE_PSAPI KMP_HAVE_PSAPI
#define OMPT_STR_MATCH(haystack, needle) __kmp_str_match(haystack, 0, needle)



//******************************************************************************
// inline functions
//******************************************************************************

inline ompt_thread_t *
ompt_get_thread_gtid(int gtid)
{
    return (gtid >= 0) ? __kmp_thread_from_gtid(gtid) : NULL;
}


inline ompt_thread_t *
ompt_get_thread()
{
    int gtid = __kmp_get_gtid();
    return ompt_get_thread_gtid(gtid);
}


inline void
ompt_set_thread_state(ompt_thread_t *thread, ompt_state_t state)
{
    thread->th.ompt_thread_info.state = state;
}


inline const char *
ompt_get_runtime_version()
{
    return &__kmp_version_lib_ver[KMP_VERSION_MAGIC_LEN];
}

#endif

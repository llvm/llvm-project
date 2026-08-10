// RUN: %libomp-compile-and-run

/*
 GCC failed this test because __kmp_get_gtid() instead of __kmp_entry_gtid()
 was called in xexpand(KMP_API_NAME_GOMP_TASKGROUP_START)(void).
 __kmp_entry_gtid() will initialize the runtime if not yet done which does not
 happen with __kmp_get_gtid().
 */

int main()
{
    #pragma omp taskgroup
    { }

    return 0;
}

#include "kmp.h"

#if !(KMP_ARCH_X86 || KMP_ARCH_X86_64 || KMP_MIC || KMP_ARCH_AARCH64 ||        \
      KMP_ARCH_PPC64 || KMP_ARCH_RISCV64 || KMP_ARCH_LOONGARCH64 ||            \
      KMP_ARCH_ARM || KMP_ARCH_VE || KMP_ARCH_S390X || KMP_ARCH_PPC_XCOFF ||   \
      KMP_ARCH_AARCH64_32)

// Because WebAssembly will use `call_indirect` to invoke the microtask and
// WebAssembly indirect calls check that the called signature is a precise
// match, we need to cast each microtask function pointer back from `void *` to
// its original type.
typedef void (*microtask_t0)(int *, int *);
typedef void (*microtask_t1)(int *, int *, void *);
typedef void (*microtask_t2)(int *, int *, void *, void *);
typedef void (*microtask_t3)(int *, int *, void *, void *, void *);
typedef void (*microtask_t4)(int *, int *, void *, void *, void *, void *);
typedef void (*microtask_t5)(int *, int *, void *, void *, void *, void *,
                             void *);
typedef void (*microtask_t6)(int *, int *, void *, void *, void *, void *,
                             void *, void *);
typedef void (*microtask_t7)(int *, int *, void *, void *, void *, void *,
                             void *, void *, void *);
typedef void (*microtask_t8)(int *, int *, void *, void *, void *, void *,
                             void *, void *, void *, void *);
typedef void (*microtask_t9)(int *, int *, void *, void *, void *, void *,
                             void *, void *, void *, void *, void *);
typedef void (*microtask_t10)(int *, int *, void *, void *, void *, void *,
                              void *, void *, void *, void *, void *, void *);
typedef void (*microtask_t11)(int *, int *, void *, void *, void *, void *,
                              void *, void *, void *, void *, void *, void *,
                              void *);
typedef void (*microtask_t12)(int *, int *, void *, void *, void *, void *,
                              void *, void *, void *, void *, void *, void *,
                              void *, void *);
typedef void (*microtask_t13)(int *, int *, void *, void *, void *, void *,
                              void *, void *, void *, void *, void *, void *,
                              void *, void *, void *);
typedef void (*microtask_t14)(int *, int *, void *, void *, void *, void *,
                              void *, void *, void *, void *, void *, void *,
                              void *, void *, void *, void *);
typedef void (*microtask_t15)(int *, int *, void *, void *, void *, void *,
                              void *, void *, void *, void *, void *, void *,
                              void *, void *, void *, void *, void *);

// we really only need the case with 1 argument, because CLANG always build
// a struct of pointers to shared variables referenced in the outlined function
int __kmp_invoke_microtask(microtask_t pkfn, int gtid, int tid, int argc,
                           void *p_argv[]
#if OMPT_SUPPORT
                           ,
                           void **exit_frame_ptr
#endif
) {
#if OMPT_SUPPORT
  *exit_frame_ptr = OMPT_GET_FRAME_ADDRESS(0);
#endif

  switch (argc) {
  default:
    fprintf(stderr, "Too many args to microtask: %d!\n", argc);
    fflush(stderr);
    exit(-1);
  case 0:
    (*(microtask_t0)pkfn)(&gtid, &tid);
    break;
  case 1:
    (*(microtask_t1)pkfn)(&gtid, &tid, p_argv[0]);
    break;
  case 2:
    (*(microtask_t2)pkfn)(&gtid, &tid, p_argv[0], p_argv[1]);
    break;
  case 3:
    (*(microtask_t3)pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2]);
    break;
  case 4:
    (*(microtask_t4)pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2],
                          p_argv[3]);
    break;
  case 5:
    (*(microtask_t5)pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2],
                          p_argv[3], p_argv[4]);
    break;
  case 6:
    (*(microtask_t6)pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2],
                          p_argv[3], p_argv[4], p_argv[5]);
    break;
  case 7:
    (*(microtask_t7)pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2],
                          p_argv[3], p_argv[4], p_argv[5], p_argv[6]);
    break;
  case 8:
    (*(microtask_t8)pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2],
                          p_argv[3], p_argv[4], p_argv[5], p_argv[6],
                          p_argv[7]);
    break;
  case 9:
    (*(microtask_t9)pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2],
                          p_argv[3], p_argv[4], p_argv[5], p_argv[6], p_argv[7],
                          p_argv[8]);
    break;
  case 10:
    (*(microtask_t10)pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2],
                           p_argv[3], p_argv[4], p_argv[5], p_argv[6],
                           p_argv[7], p_argv[8], p_argv[9]);
    break;
  case 11:
    (*(microtask_t11)pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2],
                           p_argv[3], p_argv[4], p_argv[5], p_argv[6],
                           p_argv[7], p_argv[8], p_argv[9], p_argv[10]);
    break;
  case 12:
    (*(microtask_t12)pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2],
                           p_argv[3], p_argv[4], p_argv[5], p_argv[6],
                           p_argv[7], p_argv[8], p_argv[9], p_argv[10],
                           p_argv[11]);
    break;
  case 13:
    (*(microtask_t13)pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2],
                           p_argv[3], p_argv[4], p_argv[5], p_argv[6],
                           p_argv[7], p_argv[8], p_argv[9], p_argv[10],
                           p_argv[11], p_argv[12]);
    break;
  case 14:
    (*(microtask_t14)pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2],
                           p_argv[3], p_argv[4], p_argv[5], p_argv[6],
                           p_argv[7], p_argv[8], p_argv[9], p_argv[10],
                           p_argv[11], p_argv[12], p_argv[13]);
    break;
  case 15:
    (*(microtask_t15)pkfn)(&gtid, &tid, p_argv[0], p_argv[1], p_argv[2],
                           p_argv[3], p_argv[4], p_argv[5], p_argv[6],
                           p_argv[7], p_argv[8], p_argv[9], p_argv[10],
                           p_argv[11], p_argv[12], p_argv[13], p_argv[14]);
    break;
  }

  return 1;
}

#endif

#include "kmp.h"

#include <utility>

#if !(KMP_ARCH_X86 || KMP_ARCH_X86_64 || KMP_MIC || KMP_ARCH_AARCH64 ||        \
      KMP_ARCH_PPC64 || KMP_ARCH_RISCV64 || KMP_ARCH_LOONGARCH64 ||            \
      KMP_ARCH_ARM || KMP_ARCH_VE || KMP_ARCH_S390X || KMP_ARCH_PPC_XCOFF ||   \
      KMP_ARCH_AARCH64_32)

template <size_t> using microtask_argument_t = void *;

template <size_t... Indices>
static void invokeMicrotask(microtask_t pkfn, int *gtid, int *tid,
                            void *p_argv[], std::index_sequence<Indices...>) {
  // WebAssembly's `call_indirect` requires the callee type to exactly match the
  // call site. Cast the variadic microtask_t to the fixed-arity signature that
  // matches argc before invoking it.
  using typed_microtask_t =
      void (*)(int *, int *, microtask_argument_t<Indices>...);
  (*(typed_microtask_t)pkfn)(gtid, tid, p_argv[Indices]...);
}

// Keep a bounded set of exact signatures for targets that cannot dynamically
// construct a variable-argument microtask call.
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

#define KMP_INVOKE_MICROTASK_CASE(N)                                           \
  case N:                                                                      \
    invokeMicrotask(pkfn, &gtid, &tid, p_argv, std::make_index_sequence<N>{}); \
    break

  switch (argc) {
    KMP_INVOKE_MICROTASK_CASE(0);
    KMP_INVOKE_MICROTASK_CASE(1);
    KMP_INVOKE_MICROTASK_CASE(2);
    KMP_INVOKE_MICROTASK_CASE(3);
    KMP_INVOKE_MICROTASK_CASE(4);
    KMP_INVOKE_MICROTASK_CASE(5);
    KMP_INVOKE_MICROTASK_CASE(6);
    KMP_INVOKE_MICROTASK_CASE(7);
    KMP_INVOKE_MICROTASK_CASE(8);
    KMP_INVOKE_MICROTASK_CASE(9);
    KMP_INVOKE_MICROTASK_CASE(10);
    KMP_INVOKE_MICROTASK_CASE(11);
    KMP_INVOKE_MICROTASK_CASE(12);
    KMP_INVOKE_MICROTASK_CASE(13);
    KMP_INVOKE_MICROTASK_CASE(14);
    KMP_INVOKE_MICROTASK_CASE(15);
  default:
    fprintf(stderr, "Too many args to microtask: %d!\n", argc);
    fflush(stderr);
    exit(-1);
  }

#undef KMP_INVOKE_MICROTASK_CASE

  return 1;
}

#endif

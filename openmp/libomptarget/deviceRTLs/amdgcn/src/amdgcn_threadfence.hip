#include "target_impl.h"

EXTERN void __kmpc_impl_threadfence() {
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
}

EXTERN void __kmpc_impl_threadfence_block() {
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
}

EXTERN void __kmpc_impl_threadfence_system() {
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
}

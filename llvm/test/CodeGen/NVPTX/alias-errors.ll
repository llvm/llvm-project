; RUN: not --crash llc < %s -mtriple=nvptx64 -mcpu=sm_30 -mattr=+ptx43 2>&1 | FileCheck %s --check-prefix=ATTR
; RUN: not --crash llc < %s -mtriple=nvptx64 -mcpu=sm_20 -mattr=+ptx63 2>&1 | FileCheck %s --check-prefix=ATTR
; RUN: not llc < %s -mtriple=nvptx64 -mcpu=sm_30 -mattr=+ptx63 2>&1 | FileCheck %s --check-prefix=ALIAS

; ATTR: .alias requires PTX version >= 6.3 and sm_30

; PTX's .alias directive accepts "non-entry function symbols" only, and requires
; the aliasee to be defined in the same module, so none of the aliases below can
; be lowered. Report them instead of aborting: a frontend that emits an alias
; for every export would otherwise lose the whole compilation to a crash.

; ALIAS-DAG: error: unsupported alias 'not_a_function': PTX can only alias a function
; ALIAS-DAG: error: {{.*}}unsupported alias 'kernel_alias': PTX .alias requires non-entry functions, so a kernel cannot be aliased
; ALIAS-DAG: error: {{.*}}unsupported alias 'weak_alias': PTX forbids a .weak aliasee

@a = global i32 42, align 8
@not_a_function = internal alias i32, ptr @a

define ptx_kernel void @the_kernel(ptr %p) {
  store i32 0, ptr %p
  ret void
}
@kernel_alias = alias void (ptr), ptr @the_kernel

define void @plain_func(ptr %p) {
  store i32 1, ptr %p
  ret void
}
@weak_alias = weak alias void (ptr), ptr @plain_func

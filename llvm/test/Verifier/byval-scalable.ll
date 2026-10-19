; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; A byval argument is copied into the caller's frame, so no target can pass one
; whose size is not known at compile time (see issue #182759).

; CHECK: scalable 'byval' arguments are unsupported
declare void @vector(ptr byval(<vscale x 4 x i32>))

; CHECK: scalable 'byval' arguments are unsupported
declare void @struct(ptr byval({ <vscale x 4 x i32> }))

; CHECK: scalable 'byval' arguments are unsupported
define void @definition(ptr byval(<vscale x 4 x i32>) %p) {
  ret void
}

; The attribute is checked at a call site too, which is a separate path.

declare void @callee(ptr)

; CHECK: scalable 'byval' arguments are unsupported
define void @call(ptr %p) {
  call void @callee(ptr byval(<vscale x 4 x i32>) %p)
  ret void
}

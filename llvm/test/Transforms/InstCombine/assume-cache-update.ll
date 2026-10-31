; RUN: opt -passes='print<assumptions>,instcombine,print<assumptions>' -disable-output %s 2>&1 | FileCheck %s

declare void @llvm.assume(i1)

; CHECK: Cached assumptions for function: drop_dead_bundle
; CHECK-NEXT: [ "ignore"(ptr %q), "align"(ptr %p, i64 8) ]
; CHECK: Cached assumptions for function: drop_dead_bundle
; CHECK-NEXT: [ "align"(ptr %p, i64 8) ]
define i8 @drop_dead_bundle(ptr %p, ptr %q) {
  %v = load i8, ptr %p
  call void @llvm.assume(i1 true) [ "ignore"(ptr %q), "align"(ptr %p, i64 8) ]
  ret i8 %v
}

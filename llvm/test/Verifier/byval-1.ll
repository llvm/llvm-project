; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Wrong types for attribute: nest noalias nocapture nonnull readnone readonly byref(i32) byval(i32) inalloca(i32) preallocated(i32) sret(i32) align 1 dereferenceable(1) dereferenceable_or_null(1)
; CHECK-NEXT: void (i32)* @h
declare void @h(i32 byval(i32) %num)

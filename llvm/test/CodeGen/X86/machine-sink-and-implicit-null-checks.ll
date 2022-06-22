; RUN: llc -mtriple=x86_64-apple-macosx -O3 -enable-implicit-null-checks -o - < %s 2>&1 | FileCheck %s

declare void @throw0()
declare void @throw1()

define i1 @f(ptr %p0, ptr %p1) {
 entry:
  %c0 = icmp eq ptr %p0, null
  br i1 %c0, label %throw0, label %continue0, !make.implicit !0

 continue0:
  %v0 = load i8, ptr %p0
  %c1 = icmp eq ptr %p1, null
  br i1 %c1, label %throw1, label %continue1, !make.implicit !0

 continue1:
  %v1 = load i8, ptr %p1
  %v = icmp eq i8 %v0, %v1
  ret i1 %v

 throw0:
  call void @throw0()
  unreachable

 throw1:
  call void @throw1()
  unreachable
}

declare void @foo()

declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)
declare ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token, i32, i32) nounwind readonly

; Check for a crash.  The crash is not specific to statepoints, but
; gc.statpeoint is an easy way to generate a fill instruction in
; %continue0 (which causes the llc crash).
define i1 @g(ptr addrspace(1) %p0, ptr %p1) gc "statepoint-example" {
 entry:
  %c0 = icmp eq ptr addrspace(1) %p0, null
  %tok = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @foo, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %p0)]
  %p0.relocated = call coldcc ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %tok, i32 0, i32 0) ; (%p0, %p0)
  br i1 %c0, label %throw0, label %continue0, !make.implicit !0

 continue0:
  %c1 = icmp eq ptr %p1, null
  br i1 %c1, label %throw1, label %continue1, !make.implicit !0

 continue1:
  %v0 = load i8, ptr addrspace(1) %p0.relocated
  %v1 = load i8, ptr %p1
  %v = icmp eq i8 %v0, %v1
  ret i1 %v

 throw0:
  call void @throw0()
  unreachable

 throw1:
  call void @throw1()
  unreachable
}

; Check that we have two implicit null checks in @f

; CHECK: __LLVM_FaultMaps:
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .short  0
; CHECK-NEXT:        .long   1

; FunctionInfo[0] =

; FunctionAddress =
; CHECK-NEXT:        .quad   _f

; NumFaultingPCs =
; CHECK-NEXT:        .long   2

; Reserved =
; CHECK-NEXT:        .long   0

!0 = !{}

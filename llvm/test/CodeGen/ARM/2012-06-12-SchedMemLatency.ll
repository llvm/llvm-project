; RUN: llc < %s -o /dev/null "-mtriple=thumbv7-apple-ios" -debug-only=post-RA-sched 2> %t
; RUN: FileCheck %s < %t
; REQUIRES: asserts
; Make sure that mayalias store-load dependencies have one cycle
; latency regardless of whether they are barriers or not.

; CHECK: ** List Scheduling
; CHECK: SU(2){{.*}}STR{{.*}}Volatile
; CHECK-NOT: SU({{.*}}): Ord
; CHECK: SU(3): Ord Latency=1
; CHECK-NOT: SU({{.*}}): Ord
; CHECK: SU(3){{.*}}LDR{{.*}}Volatile
; CHECK-NOT: SU({{.*}}): Ord
; CHECK: SU(2): Ord Latency=1
; CHECK-NOT: SU({{.*}}): Ord
; CHECK: Successors:
; CHECK: ** List Scheduling
; CHECK: SU(2){{.*}}STR{{.*}}
; CHECK-NOT: SU({{.*}}): Ord
; CHECK: SU(3): Ord Latency=1
; CHECK-NOT: SU({{.*}}): Ord
; CHECK: SU(3){{.*}}LDR{{.*}}
; CHECK-NOT: SU({{.*}}): Ord
; CHECK: SU(2): Ord Latency=1
; CHECK-NOT: SU({{.*}}): Ord
; CHECK: Successors:
define i32 @f1(i32* nocapture %p1, i32* nocapture %p2) nounwind {
entry:
  store volatile i32 65540, i32* %p1, align 4
  %0 = load volatile i32, i32* %p2, align 4
  ret i32 %0
}

define i32 @f2(i32* nocapture %p1, i32* nocapture %p2) nounwind {
entry:
  store i32 65540, i32* %p1, align 4
  %0 = load i32, i32* %p2, align 4
  ret i32 %0
}

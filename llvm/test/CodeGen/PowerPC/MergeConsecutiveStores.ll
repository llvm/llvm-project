; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu -mattr=+altivec < %s | FileCheck %s

;; This test ensures that MergeConsecutiveStores does not attempt to
;; merge stores or loads when doing so would result in unaligned
;; memory operations (unless the target supports those, e.g. X86).

;; This issue happen in other situations for other targets, but PPC
;; with Altivec extensions was chosen for the test because it does not
;; support unaligned access with AltiVec instructions. If the 4
;; load/stores get merged to an v4i32 vector type severely bad code
;; gets generated: it painstakingly copies the values to a temporary
;; location on the stack, with vector ops, in order to then use
;; integer ops to load from the temporary stack location and store to
;; the final location. Yuck!

%struct.X = type { i32, i32, i32, i32 }

@fx = common global %struct.X zeroinitializer, align 4
@fy = common global %struct.X zeroinitializer, align 4

;; In this test case, lvx and stvx instructions should NOT be
;; generated, as the alignment is not sufficient for it to be
;; worthwhile.

;; CHECK-LABEL: f:
;; CHECK-DAG:      lwzu
;; CHECK-DAG:      stwu
;; CHECK-DAG: lwz
;; CHECK-DAG: lwz
;; CHECK-DAG: lwz
;; CHECK-DAG: stw
;; CHECK-DAG: stw
;; CHECK-DAG: stw
;; CHECK-NEXT: blr
define void @f() {
entry:
  %0 = load i32, ptr @fx, align 4
  %1 = load i32, ptr getelementptr inbounds (%struct.X, ptr @fx, i32 0, i32 1), align 4
  %2 = load i32, ptr getelementptr inbounds (%struct.X, ptr @fx, i32 0, i32 2), align 4
  %3 = load i32, ptr getelementptr inbounds (%struct.X, ptr @fx, i32 0, i32 3), align 4
  store i32 %0, ptr @fy, align 4
  store i32 %1, ptr getelementptr inbounds (%struct.X, ptr @fy, i32 0, i32 1), align 4
  store i32 %2, ptr getelementptr inbounds (%struct.X, ptr @fy, i32 0, i32 2), align 4
  store i32 %3, ptr getelementptr inbounds (%struct.X, ptr @fy, i32 0, i32 3), align 4
  ret void
}

@gx = common global %struct.X zeroinitializer, align 16
@gy = common global %struct.X zeroinitializer, align 16

;; In this test, lvx and stvx instructions SHOULD be generated, as
;; the 16-byte alignment of the new load/store is acceptable.
;; CHECK-LABEL: g:
;; CHECK: lvx
;; CHECK: stvx
;; CHECK: blr
define void @g() {
entry:
  %0 = load i32, ptr @fx, align 16
  %1 = load i32, ptr getelementptr inbounds (%struct.X, ptr @fx, i32 0, i32 1), align 4
  %2 = load i32, ptr getelementptr inbounds (%struct.X, ptr @fx, i32 0, i32 2), align 4
  %3 = load i32, ptr getelementptr inbounds (%struct.X, ptr @fx, i32 0, i32 3), align 4
  store i32 %0, ptr @fy, align 16
  store i32 %1, ptr getelementptr inbounds (%struct.X, ptr @fy, i32 0, i32 1), align 4
  store i32 %2, ptr getelementptr inbounds (%struct.X, ptr @fy, i32 0, i32 2), align 4
  store i32 %3, ptr getelementptr inbounds (%struct.X, ptr @fy, i32 0, i32 3), align 4
  ret void
}

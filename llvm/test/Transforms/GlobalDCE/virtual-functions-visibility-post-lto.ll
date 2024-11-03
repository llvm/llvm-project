; RUN: opt < %s -passes='globaldce<vfe-linkage-unit-visibility>' -S | FileCheck %s
; RUN: opt < %s -passes='lto<O2>' -S | FileCheck %s

; structs A, B and C have vcall_visibility of public, linkage-unit and
; translation-unit respectively. This test is run after LTO linking (the
; pass parameter simulates how GlobalDCE is invoked from the regular LTO
; pipeline), so B and C can be VFE'd.

;; Try again without being in the LTO post link, we can only eliminate C.
; RUN: opt < %s -passes='globaldce' -S | FileCheck %s --check-prefix=NO-LTO
; RUN: opt < %s -passes='default<O2>' -S | FileCheck %s --check-prefix=NO-LTO

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

%struct.A = type { ptr }

@_ZTV1A = hidden unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1A3fooEv] }, align 8, !type !0, !type !1, !vcall_visibility !2

define internal void @_ZN1AC2Ev(ptr %this) {
entry:
  store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV1A, i64 0, inrange i32 0, i64 2), ptr %this, align 8
  ret void
}

; CHECK: define {{.*}} @_ZN1A3fooEv(
; NO-LTO: define {{.*}} @_ZN1A3fooEv(
define internal void @_ZN1A3fooEv(ptr nocapture %this) {
entry:
  ret void
}

define dso_local ptr @_Z6make_Av() {
entry:
  %call = tail call ptr @_Znwm(i64 8)
  tail call void @_ZN1AC2Ev(ptr %call)
  ret ptr %call
}


%struct.B = type { ptr }

@_ZTV1B = hidden unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1B3fooEv] }, align 8, !type !0, !type !1, !vcall_visibility !3

define internal void @_ZN1BC2Ev(ptr %this) {
entry:
  store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV1B, i64 0, inrange i32 0, i64 2), ptr %this, align 8
  ret void
}

; CHECK-NOT: define {{.*}} @_ZN1B3fooEv(
; NO-LTO: define {{.*}} @_ZN1B3fooEv(
define internal void @_ZN1B3fooEv(ptr nocapture %this) {
entry:
  ret void
}

define dso_local ptr @_Z6make_Bv() {
entry:
  %call = tail call ptr @_Znwm(i64 8)
  tail call void @_ZN1BC2Ev(ptr %call)
  ret ptr %call
}


%struct.C = type { ptr }

@_ZTV1C = hidden unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1C3fooEv] }, align 8, !type !0, !type !1, !vcall_visibility !4

define internal void @_ZN1CC2Ev(ptr %this) {
entry:
  store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV1C, i64 0, inrange i32 0, i64 2), ptr %this, align 8
  ret void
}

; CHECK-NOT: define {{.*}} @_ZN1C3fooEv(
; NO-LTO-NOT: define {{.*}} @_ZN1C3fooEv(
define internal void @_ZN1C3fooEv(ptr nocapture %this) {
entry:
  ret void
}

define dso_local ptr @_Z6make_Cv() {
entry:
  %call = tail call ptr @_Znwm(i64 8)
  tail call void @_ZN1CC2Ev(ptr %call)
  ret ptr %call
}

declare dso_local noalias nonnull ptr @_Znwm(i64)

!llvm.module.flags = !{!6}

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AFvvE.virtual"}
!2 = !{i64 0} ; public vcall visibility
!3 = !{i64 1} ; linkage-unit vcall visibility
!4 = !{i64 2} ; translation-unit vcall visibility
!6 = !{i32 1, !"Virtual Function Elim", i32 1}

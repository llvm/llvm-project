; RUN: opt < %s -globaldce -S | FileCheck %s

; structs A, B and C have vcall_visibility of public, linkage-unit and
; translation-unit respectively. This test is run after LTO linking (the
; LTOPostLink metadata is present), so B and C can be VFE'd.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

%struct.A = type { i32 (...)** }

@_ZTV1A = hidden unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* null, i8* bitcast (void (%struct.A*)* @_ZN1A3fooEv to i8*)] }, align 8, !type !0, !type !1, !vcall_visibility !2

define internal void @_ZN1AC2Ev(%struct.A* %this) {
entry:
  %0 = getelementptr inbounds %struct.A, %struct.A* %this, i64 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV1A, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  ret void
}

; CHECK: define {{.*}} @_ZN1A3fooEv(
define internal void @_ZN1A3fooEv(%struct.A* nocapture %this) {
entry:
  ret void
}

define dso_local i8* @_Z6make_Av() {
entry:
  %call = tail call i8* @_Znwm(i64 8)
  %0 = bitcast i8* %call to %struct.A*
  tail call void @_ZN1AC2Ev(%struct.A* %0)
  ret i8* %call
}


%struct.B = type { i32 (...)** }

@_ZTV1B = hidden unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* null, i8* bitcast (void (%struct.B*)* @_ZN1B3fooEv to i8*)] }, align 8, !type !0, !type !1, !vcall_visibility !3

define internal void @_ZN1BC2Ev(%struct.B* %this) {
entry:
  %0 = getelementptr inbounds %struct.B, %struct.B* %this, i64 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV1B, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  ret void
}

; CHECK-NOT: define {{.*}} @_ZN1B3fooEv(
define internal void @_ZN1B3fooEv(%struct.B* nocapture %this) {
entry:
  ret void
}

define dso_local i8* @_Z6make_Bv() {
entry:
  %call = tail call i8* @_Znwm(i64 8)
  %0 = bitcast i8* %call to %struct.B*
  tail call void @_ZN1BC2Ev(%struct.B* %0)
  ret i8* %call
}


%struct.C = type { i32 (...)** }

@_ZTV1C = hidden unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* null, i8* bitcast (void (%struct.C*)* @_ZN1C3fooEv to i8*)] }, align 8, !type !0, !type !1, !vcall_visibility !4

define internal void @_ZN1CC2Ev(%struct.C* %this) {
entry:
  %0 = getelementptr inbounds %struct.C, %struct.C* %this, i64 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV1C, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  ret void
}

; CHECK-NOT: define {{.*}} @_ZN1C3fooEv(
define internal void @_ZN1C3fooEv(%struct.C* nocapture %this) {
entry:
  ret void
}

define dso_local i8* @_Z6make_Cv() {
entry:
  %call = tail call i8* @_Znwm(i64 8)
  %0 = bitcast i8* %call to %struct.C*
  tail call void @_ZN1CC2Ev(%struct.C* %0)
  ret i8* %call
}

declare dso_local noalias nonnull i8* @_Znwm(i64)

!llvm.module.flags = !{!5}

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AFvvE.virtual"}
!2 = !{i64 0} ; public vcall visibility
!3 = !{i64 1} ; linkage-unit vcall visibility
!4 = !{i64 2} ; translation-unit vcall visibility
!5 = !{i32 1, !"LTOPostLink", i32 1}

; RUN: opt < %s -globaldce -S | FileCheck %s

; We currently only use llvm.type.checked.load for virtual function pointers,
; not any other part of the vtable, so we can't remove the RTTI pointer even if
; it's never going to be loaded from.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

%struct.A = type { i32 (...)** }

; CHECK: @_ZTV1A = hidden unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*), i8* null] }, align 8, !type !0, !type !1, !vcall_visibility !2

@_ZTV1A = hidden unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*), i8* bitcast (void (%struct.A*)* @_ZN1A3fooEv to i8*)] }, align 8, !type !0, !type !1, !vcall_visibility !2
@_ZTS1A = hidden constant [3 x i8] c"1A\00", align 1
@_ZTI1A = hidden constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1A, i32 0, i32 0) }, align 8

define internal void @_ZN1AC2Ev(%struct.A* %this) {
entry:
  %0 = getelementptr inbounds %struct.A, %struct.A* %this, i64 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV1A, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  ret void
}

; CHECK-NOT: define {{.*}} @_ZN1A3fooEv(
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


declare dso_local noalias nonnull i8* @_Znwm(i64)
@_ZTVN10__cxxabiv117__class_type_infoE = external dso_local global i8*

!llvm.module.flags = !{!3, !4}

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AFvvE.virtual"}
!2 = !{i64 2} ; translation-unit vcall visibility
!3 = !{i32 1, !"LTOPostLink", i32 1}
!4 = !{i32 1, !"Virtual Function Elim", i32 1}

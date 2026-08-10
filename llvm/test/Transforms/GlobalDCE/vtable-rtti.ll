; RUN: opt < %s -passes=globaldce -S | FileCheck %s

; We currently only use llvm.type.checked.load for virtual function pointers,
; not any other part of the vtable, so we can't remove the RTTI pointer even if
; it's never going to be loaded from.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

%struct.A = type { ptr }

; CHECK: @_ZTV1A = hidden unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1A, ptr null] }, align 8, !type !0, !type !1, !vcall_visibility !2

@_ZTV1A = hidden unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN1A3fooEv] }, align 8, !type !0, !type !1, !vcall_visibility !2
@_ZTS1A = hidden constant [3 x i8] c"1A\00", align 1
@_ZTI1A = hidden constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A }, align 8

define internal void @_ZN1AC2Ev(ptr %this) {
entry:
  store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV1A, i64 0, i32 0, i64 2), ptr %this, align 8
  ret void
}

; CHECK-NOT: define {{.*}} @_ZN1A3fooEv(
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


declare dso_local noalias nonnull ptr @_Znwm(i64)
@_ZTVN10__cxxabiv117__class_type_infoE = external dso_local global ptr

!llvm.module.flags = !{!4}

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AFvvE.virtual"}
!2 = !{i64 2} ; translation-unit vcall visibility
!4 = !{i32 1, !"Virtual Function Elim", i32 1}

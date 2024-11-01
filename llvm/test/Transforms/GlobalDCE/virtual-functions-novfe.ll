; Tests that VFE is not performed when the Virtual Function Elim metadata set
; to 0. This is the same as virtual-functions.ll otherwise.
; RUN: opt < %s -passes=globaldce -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare dso_local noalias nonnull ptr @_Znwm(i64)
declare { ptr, i1 } @llvm.type.checked.load(ptr, i32, metadata)

; %struct.A is a C++ struct with two virtual functions, A::foo and A::bar. The
; !vcall_visibility metadata is set on the vtable, so we know that all virtual
; calls through this vtable are visible and use the @llvm.type.checked.load
; intrinsic. Function test_A makes a call to A::foo, but there is no call to
; A::bar anywhere, so A::bar can be deleted, and its vtable slot replaced with
; null.
; However, with the metadata set to 0 we should not perform this VFE.

%struct.A = type { ptr }

; We should retain @_ZN1A3barEv in the vtable.
; CHECK: @_ZTV1A = internal unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZN1A3fooEv, ptr @_ZN1A3barEv] }
@_ZTV1A = internal unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZN1A3fooEv, ptr @_ZN1A3barEv] }, align 8, !type !0, !type !1, !type !2, !vcall_visibility !3

; A::foo is called, so must be retained.
; CHECK: define internal i32 @_ZN1A3fooEv(
define internal i32 @_ZN1A3fooEv(ptr nocapture readnone %this) {
entry:
  ret i32 42
}

; A::bar is not used, so can be deleted with VFE, however, we should not be
; performing that elimination here.
; CHECK: define internal i32 @_ZN1A3barEv(
define internal i32 @_ZN1A3barEv(ptr nocapture readnone %this) {
entry:
  ret i32 1337
}

define dso_local i32 @test_A() {
entry:
  %call = tail call ptr @_Znwm(i64 8)
  store ptr getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV1A, i64 0, inrange i32 0, i64 2), ptr %call, align 8
  %0 = tail call { ptr, i1 } @llvm.type.checked.load(ptr getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV1A, i64 0, inrange i32 0, i64 2), i32 0, metadata !"_ZTS1A"), !nosanitize !9
  %1 = extractvalue { ptr, i1 } %0, 0, !nosanitize !9
  %call1 = tail call i32 %1(ptr nonnull %call)
  ret i32 %call1
}

!llvm.module.flags = !{!4}

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AFivE.virtual"}
!2 = !{i64 24, !"_ZTSM1AFivE.virtual"}
!3 = !{i64 2}
!4 = !{i32 1, !"Virtual Function Elim", i32 0}
!9 = !{}

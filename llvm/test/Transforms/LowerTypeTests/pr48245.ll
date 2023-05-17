; Test to ensure type tests that are only used in assumes are ignored by
; LowerTypeTests (in the normal pass sequence they will be stripped out
; by a subsequent special LTT invocation).

; RUN: opt -S -passes=lowertypetests < %s | FileCheck %s

; ModuleID = 'pr48245.o'
source_filename = "pr48245.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Foo = type { ptr }

; Check that the vtable was not turned into an alias to a rewritten private
; global.
; CHECK: @_ZTV3Foo = dso_local unnamed_addr constant
@_ZTV3Foo = dso_local unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI3Foo, ptr @_ZN3Foo2f1Ev, ptr @_ZN3Foo2f2Ev] }, align 8, !type !0, !type !1, !type !2

@_ZTVN10__cxxabiv117__class_type_infoE = external dso_local global ptr
@_ZTS3Foo = dso_local constant [5 x i8] c"3Foo\00", align 1
@_ZTI3Foo = dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS3Foo }, align 8
@b = dso_local local_unnamed_addr global ptr null, align 8

define dso_local i32 @main() local_unnamed_addr {
entry:
  %0 = load ptr, ptr @b, align 8
  %vtable.i = load ptr, ptr %0, align 8

  ; Check that the type test was not lowered.
  ; CHECK: tail call i1 @llvm.type.test
  %1 = tail call i1 @llvm.type.test(ptr %vtable.i, metadata !"_ZTS3Foo")

  tail call void @llvm.assume(i1 %1)
  %2 = load ptr, ptr %vtable.i, align 8
  %call.i = tail call i32 %2(ptr nonnull dereferenceable(8) %0)
  ret i32 %call.i
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1 noundef)
declare dso_local i32 @_ZN3Foo2f1Ev(ptr nocapture nonnull readnone dereferenceable(8) %this) unnamed_addr
declare dso_local i32 @_ZN3Foo2f2Ev(ptr nocapture nonnull readnone dereferenceable(8) %this) unnamed_addr

!0 = !{i64 16, !"_ZTS3Foo"}
!1 = !{i64 16, !"_ZTSM3FooFivE.virtual"}
!2 = !{i64 24, !"_ZTSM3FooFivE.virtual"}

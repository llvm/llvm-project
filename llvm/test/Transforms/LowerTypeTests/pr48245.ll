; Test to ensure type tests that are only used in assumes are ignored by
; LowerTypeTests (in the normal pass sequence they will be stripped out
; by a subsequent special LTT invocation).

; RUN: opt -S -passes=lowertypetests < %s | FileCheck %s

; ModuleID = 'pr48245.o'
source_filename = "pr48245.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Foo = type { i32 (...)** }

; Check that the vtable was not turned into an alias to a rewritten private
; global.
; CHECK: @_ZTV3Foo = dso_local unnamed_addr constant
@_ZTV3Foo = dso_local unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI3Foo to i8*), i8* bitcast (i32 (%struct.Foo*)* @_ZN3Foo2f1Ev to i8*), i8* bitcast (i32 (%struct.Foo*)* @_ZN3Foo2f2Ev to i8*)] }, align 8, !type !0, !type !1, !type !2

@_ZTVN10__cxxabiv117__class_type_infoE = external dso_local global i8*
@_ZTS3Foo = dso_local constant [5 x i8] c"3Foo\00", align 1
@_ZTI3Foo = dso_local constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @_ZTS3Foo, i32 0, i32 0) }, align 8
@b = dso_local local_unnamed_addr global %struct.Foo* null, align 8

define dso_local i32 @main() local_unnamed_addr {
entry:
  %0 = load %struct.Foo*, %struct.Foo** @b, align 8
  %1 = bitcast %struct.Foo* %0 to i32 (%struct.Foo*)***
  %vtable.i = load i32 (%struct.Foo*)**, i32 (%struct.Foo*)*** %1, align 8
  %2 = bitcast i32 (%struct.Foo*)** %vtable.i to i8*

  ; Check that the type test was not lowered.
  ; CHECK: tail call i1 @llvm.type.test
  %3 = tail call i1 @llvm.type.test(i8* %2, metadata !"_ZTS3Foo")

  tail call void @llvm.assume(i1 %3)
  %4 = load i32 (%struct.Foo*)*, i32 (%struct.Foo*)** %vtable.i, align 8
  %call.i = tail call i32 %4(%struct.Foo* nonnull dereferenceable(8) %0)
  ret i32 %call.i
}

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1 noundef)
declare dso_local i32 @_ZN3Foo2f1Ev(%struct.Foo* nocapture nonnull readnone dereferenceable(8) %this) unnamed_addr
declare dso_local i32 @_ZN3Foo2f2Ev(%struct.Foo* nocapture nonnull readnone dereferenceable(8) %this) unnamed_addr

!0 = !{i64 16, !"_ZTS3Foo"}
!1 = !{i64 16, !"_ZTSM3FooFivE.virtual"}
!2 = !{i64 24, !"_ZTSM3FooFivE.virtual"}

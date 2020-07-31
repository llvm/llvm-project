; Test handling of llvm.lifetime intrinsics with C++ exceptions.
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-use-after-scope -asan-use-after-return=0 -S | FileCheck %s
; RUN: opt < %s -passes='asan-pipeline' -asan-use-after-scope -asan-use-after-return=0 -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ABC = type { i32 }

$_ZN3ABCD2Ev = comdat any
$_ZTS3ABC = comdat any
$_ZTI3ABC = comdat any

@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS3ABC = linkonce_odr constant [5 x i8] c"3ABC\00", comdat
@_ZTI3ABC = linkonce_odr constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @_ZTS3ABC, i32 0, i32 0) }, comdat

define void @Throw() sanitize_address personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: define void @Throw()
entry:
  %x = alloca %struct.ABC, align 4
  %0 = bitcast %struct.ABC* %x to i8*

  ; Poison memory in prologue: F1F1F1F1F8F3F3F3
  ; CHECK: store i64 -868082052615769615, i64* %{{[0-9]+}}

  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0)
  ; CHECK: store i8 4, i8* %{{[0-9]+}}
  ; CHECK-NEXT: @llvm.lifetime.start

  %exception = call i8* @__cxa_allocate_exception(i64 4)
  invoke void @__cxa_throw(i8* %exception, i8* bitcast ({ i8*, i8* }* @_ZTI3ABC to i8*), i8* bitcast (void (%struct.ABC*)* @_ZN3ABCD2Ev to i8*)) noreturn
          to label %unreachable unwind label %lpad
  ; CHECK: call void @__asan_handle_no_return
  ; CHECK-NEXT: @__cxa_throw

lpad:
  %1 = landingpad { i8*, i32 }
          cleanup
  call void @_ZN3ABCD2Ev(%struct.ABC* nonnull %x)
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %0)
  ; CHECK: store i8 -8, i8* %{{[0-9]+}}
  ; CHECK-NEXT: @llvm.lifetime.end

  resume { i8*, i32 } %1
  ; CHECK: store i64 0, i64* %{{[0-9]+}}
  ; CHECK-NEXT: resume

unreachable:
  unreachable
}

%rtti.TypeDescriptor9 = type { i8**, i8*, [10 x i8] }
%eh.CatchableType = type { i32, i32, i32, i32, i32, i32, i32 }
%eh.CatchableTypeArray.1 = type { i32, [1 x i32] }
%eh.ThrowInfo = type { i32, i32, i32, i32 }

$"\01??1ABC@@QEAA@XZ" = comdat any
$"\01??_R0?AUABC@@@8" = comdat any
$"_CT??_R0?AUABC@@@84" = comdat any
$"_CTA1?AUABC@@" = comdat any
$"_TI1?AUABC@@" = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0?AUABC@@@8" = linkonce_odr global %rtti.TypeDescriptor9 { i8** @"\01??_7type_info@@6B@", i8* null, [10 x i8] c".?AUABC@@\00" }, comdat
@__ImageBase = external constant i8
@"_CT??_R0?AUABC@@@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor9* @"\01??_R0?AUABC@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 4, i32 0 }, section ".xdata", comdat
@"_CTA1?AUABC@@" = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableType* @"_CT??_R0?AUABC@@@84" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@"_TI1?AUABC@@" = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (void (%struct.ABC*)* @"\01??1ABC@@QEAA@XZ" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%eh.CatchableTypeArray.1* @"_CTA1?AUABC@@" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, section ".xdata", comdat

define void @ThrowWin() sanitize_address personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
; CHECK-LABEL: define void @ThrowWin()
entry:
  %x = alloca %struct.ABC, align 4
  %tmp = alloca %struct.ABC, align 4
  %0 = bitcast %struct.ABC* %x to i8*

  ; Poison memory in prologue: F1F1F1F1F8F304F2
  ; CHECK: store i64 -935355671561244175, i64* %{{[0-9]+}}

  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0)
  ; CHECK: store i8 4, i8* %{{[0-9]+}}
  ; CHECK-NEXT: @llvm.lifetime.start

  %1 = bitcast %struct.ABC* %tmp to i8*
  invoke void @_CxxThrowException(i8* %1, %eh.ThrowInfo* nonnull @"_TI1?AUABC@@") noreturn
          to label %unreachable unwind label %ehcleanup
  ; CHECK: call void @__asan_handle_no_return
  ; CHECK-NEXT: @_CxxThrowException

ehcleanup:
  %2 = cleanuppad within none []
  call void @"\01??1ABC@@QEAA@XZ"(%struct.ABC* nonnull %x) [ "funclet"(token %2) ]
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %0)
  ; CHECK: store i8 -8, i8* %{{[0-9]+}}
  ; CHECK-NEXT: @llvm.lifetime.end

  cleanupret from %2 unwind to caller
  ; CHECK: store i64 0, i64* %{{[0-9]+}}
  ; CHECK-NEXT: cleanupret

unreachable:
  unreachable
}


declare i32 @__gxx_personality_v0(...)
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
declare void @__cxa_throw(i8*, i8*, i8*) local_unnamed_addr
declare i8* @__cxa_allocate_exception(i64) local_unnamed_addr
declare void @_ZN3ABCD2Ev(%struct.ABC* %this) unnamed_addr
declare void @"\01??1ABC@@QEAA@XZ"(%struct.ABC* %this)
declare void @_CxxThrowException(i8*, %eh.ThrowInfo*)
declare i32 @__CxxFrameHandler3(...)

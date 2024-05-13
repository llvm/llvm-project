; Test handling of llvm.lifetime intrinsics with C++ exceptions.
; RUN: opt < %s -passes=asan -asan-use-after-scope -asan-use-after-return=never -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ABC = type { i32 }

$_ZN3ABCD2Ev = comdat any
$_ZTS3ABC = comdat any
$_ZTI3ABC = comdat any

@_ZTVN10__cxxabiv117__class_type_infoE = external global ptr
@_ZTS3ABC = linkonce_odr constant [5 x i8] c"3ABC\00", comdat
@_ZTI3ABC = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS3ABC }, comdat

define void @Throw() sanitize_address personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: define void @Throw()
entry:
  %x = alloca %struct.ABC, align 4

  ; Poison memory in prologue: F1F1F1F1F8F3F3F3
  ; CHECK: store i64 -868082052615769615, ptr %{{[0-9]+}}

  call void @llvm.lifetime.start.p0(i64 4, ptr %x)
  ; CHECK: store i8 4, ptr %{{[0-9]+}}
  ; CHECK-NEXT: @llvm.lifetime.start

  %exception = call ptr @__cxa_allocate_exception(i64 4)
  invoke void @__cxa_throw(ptr %exception, ptr @_ZTI3ABC, ptr @_ZN3ABCD2Ev) noreturn
          to label %unreachable unwind label %lpad
  ; CHECK: call void @__asan_handle_no_return
  ; CHECK-NEXT: @__cxa_throw

lpad:
  %0 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN3ABCD2Ev(ptr nonnull %x)
  call void @llvm.lifetime.end.p0(i64 4, ptr %x)
  ; CHECK: store i8 -8, ptr %{{[0-9]+}}
  ; CHECK-NEXT: @llvm.lifetime.end

  resume { ptr, i32 } %0
  ; CHECK: store i64 0, ptr %{{[0-9]+}}
  ; CHECK-NEXT: resume

unreachable:
  unreachable
}

%rtti.TypeDescriptor9 = type { ptr, ptr, [10 x i8] }
%eh.CatchableType = type { i32, i32, i32, i32, i32, i32, i32 }
%eh.CatchableTypeArray.1 = type { i32, [1 x i32] }
%eh.ThrowInfo = type { i32, i32, i32, i32 }

$"\01??1ABC@@QEAA@XZ" = comdat any
$"\01??_R0?AUABC@@@8" = comdat any
$"_CT??_R0?AUABC@@@84" = comdat any
$"_CTA1?AUABC@@" = comdat any
$"_TI1?AUABC@@" = comdat any

@"\01??_7type_info@@6B@" = external constant ptr
@"\01??_R0?AUABC@@@8" = linkonce_odr global %rtti.TypeDescriptor9 { ptr @"\01??_7type_info@@6B@", ptr null, [10 x i8] c".?AUABC@@\00" }, comdat
@__ImageBase = external constant i8
@"_CT??_R0?AUABC@@@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R0?AUABC@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 -1, i32 0, i32 4, i32 0 }, section ".xdata", comdat
@"_CTA1?AUABC@@" = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CT??_R0?AUABC@@@84" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32)] }, section ".xdata", comdat
@"_TI1?AUABC@@" = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??1ABC@@QEAA@XZ" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"_CTA1?AUABC@@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, section ".xdata", comdat

define void @ThrowWin() sanitize_address personality ptr @__CxxFrameHandler3 {
; CHECK-LABEL: define void @ThrowWin()
entry:
  %x = alloca %struct.ABC, align 4
  %tmp = alloca %struct.ABC, align 4

  ; Poison memory in prologue: F1F1F1F1F8F304F2
  ; CHECK: store i64 -935355671561244175, ptr %{{[0-9]+}}

  call void @llvm.lifetime.start.p0(i64 4, ptr %x)
  ; CHECK: store i8 4, ptr %{{[0-9]+}}
  ; CHECK-NEXT: @llvm.lifetime.start

  invoke void @_CxxThrowException(ptr %tmp, ptr nonnull @"_TI1?AUABC@@") noreturn
          to label %unreachable unwind label %ehcleanup
  ; CHECK: call void @__asan_handle_no_return
  ; CHECK-NEXT: @_CxxThrowException

ehcleanup:
  %0 = cleanuppad within none []
  call void @"\01??1ABC@@QEAA@XZ"(ptr nonnull %x) [ "funclet"(token %0) ]
  call void @llvm.lifetime.end.p0(i64 4, ptr %x)
  ; CHECK: store i8 -8, ptr %{{[0-9]+}}
  ; CHECK-NEXT: @llvm.lifetime.end

  cleanupret from %0 unwind to caller
  ; CHECK: store i64 0, ptr %{{[0-9]+}}
  ; CHECK-NEXT: cleanupret

unreachable:
  unreachable
}


declare i32 @__gxx_personality_v0(...)
declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)
declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr
declare ptr @__cxa_allocate_exception(i64) local_unnamed_addr
declare void @_ZN3ABCD2Ev(ptr %this) unnamed_addr
declare void @"\01??1ABC@@QEAA@XZ"(ptr %this)
declare void @_CxxThrowException(ptr, ptr)
declare i32 @__CxxFrameHandler3(...)

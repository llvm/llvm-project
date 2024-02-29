; RUN: opt < %s -passes='debugify,function(loop-mssa(licm))'  -S -o /dev/null
; RUN: opt < %s -passes='debugify,function(loop-mssa(licm))'  -S -o /dev/null --try-experimental-debuginfo-iterators
;
; The following test is from https://bugs.llvm.org/show_bug.cgi?id=36238
; This test should pass (not assert or fault). The error that originally
; provoked this test was regarding the LCSSA pass trying to insert a dbg.value
; intrinsic into a catchswitch block.

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.0"

%struct.e = type { i32 }
%struct.d = type { i8 }
%class.f = type { %class.b }
%class.b = type { i8 }
%struct.k = type opaque

@"\01?l@@3HA" = local_unnamed_addr global i32 0, align 4

define i32 @"\01?m@@YAJXZ"() personality ptr @__C_specific_handler {
entry:
  %n = alloca %struct.e, align 4
  %db = alloca i32, align 4
  %o = alloca %struct.d, align 1
  %q = alloca ptr, align 8
  %r = alloca i32, align 4
  %u = alloca i64, align 8
  %s = alloca %class.f, align 1
  %offset = alloca i64, align 8
  %t = alloca i64, align 8
  %status = alloca i32, align 4
  call void (...) @llvm.localescape(ptr nonnull %s, ptr nonnull %status)
  %0 = load i32, ptr @"\01?l@@3HA", align 4, !tbaa !3
  %call = call ptr @"\01??0f@@QEAA@H@Z"(ptr nonnull %s, i32 %0)
  br label %for.cond

for.cond:                                         ; preds = %cleanup.cont, %entry
  %p.0 = phi i32 [ undef, %entry ], [ %call2, %cleanup.cont ]
  invoke void @"\01?h@@YAXPEAH0HPEAIPEAPEAEPEA_K33PEAUd@@4@Z"(ptr nonnull %db, ptr nonnull %n, i32 undef, ptr nonnull %r, ptr nonnull %q, ptr nonnull %u, ptr nonnull %offset, ptr nonnull %t, ptr nonnull %s, ptr nonnull %o)
          to label %__try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %for.cond
  %1 = catchswitch within none [label %__except.ret] unwind label %ehcleanup

__except.ret:                                     ; preds = %catch.dispatch
  %2 = catchpad within %1 [ptr @"\01?filt$0@0@m@@"]
  catchret from %2 to label %cleanup7

__try.cont:                                       ; preds = %for.cond
  %tobool = icmp eq i32 %p.0, 0
  br i1 %tobool, label %if.end, label %cleanup7

if.end:                                           ; preds = %__try.cont
  %call2 = invoke i32 @"\01?a@@YAJXZ"()
          to label %cleanup.cont unwind label %ehcleanup

cleanup.cont:                                     ; preds = %if.end
  br label %for.cond

ehcleanup:                                        ; preds = %if.end, %catch.dispatch
  %3 = cleanuppad within none []
  call void @"\01??1b@@QEAA@XZ"(ptr nonnull %s) [ "funclet"(token %3) ]
  cleanupret from %3 unwind to caller

cleanup7:                                         ; preds = %__try.cont, %__except.ret
  %p.2.ph = phi i32 [ 7, %__except.ret ], [ %p.0, %__try.cont ]
  call void @"\01??1b@@QEAA@XZ"(ptr nonnull %s)
  ret i32 %p.2.ph
}

declare ptr @"\01??0f@@QEAA@H@Z"(ptr returned, i32) unnamed_addr

define internal i32 @"\01?filt$0@0@m@@"(ptr %exception_pointers, ptr %frame_pointer) personality ptr @__C_specific_handler {
entry:
  %0 = tail call ptr @llvm.eh.recoverfp(ptr @"\01?m@@YAJXZ", ptr %frame_pointer)
  %1 = tail call ptr @llvm.localrecover(ptr @"\01?m@@YAJXZ", ptr %0, i32 0)
  %2 = tail call ptr @llvm.localrecover(ptr @"\01?m@@YAJXZ", ptr %0, i32 1)
  %agg.tmp = alloca %class.f, align 1
  %3 = load ptr, ptr %exception_pointers, align 8
  %4 = load i32, ptr %3, align 4
  %5 = load i8, ptr %1, align 1
  store i8 %5, ptr %agg.tmp, align 1
  %call = invoke i32 @"\01?j@@YAJVf@@JPEAUk@@PEAH@Z"(i8 %5, i32 %4, ptr %exception_pointers, ptr %2)
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  call void @"\01??1b@@QEAA@XZ"(ptr nonnull %agg.tmp)
  ret i32 %call

ehcleanup:                                        ; preds = %entry
  %6 = cleanuppad within none []
  call void @"\01??1b@@QEAA@XZ"(ptr nonnull %agg.tmp) [ "funclet"(token %6) ]
  cleanupret from %6 unwind to caller
}

declare ptr @llvm.eh.recoverfp(ptr, ptr)
declare ptr @llvm.localrecover(ptr, ptr, i32)
declare i32 @"\01?j@@YAJVf@@JPEAUk@@PEAH@Z"(i8, i32, ptr, ptr) local_unnamed_addr
declare i32 @__C_specific_handler(...)
declare void @"\01?h@@YAXPEAH0HPEAIPEAPEAEPEA_K33PEAUd@@4@Z"(ptr, ptr, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr) local_unnamed_addr
declare i32 @"\01?a@@YAJXZ"() local_unnamed_addr
declare void @llvm.localescape(...)
declare void @"\01??1b@@QEAA@XZ"(ptr) unnamed_addr

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}

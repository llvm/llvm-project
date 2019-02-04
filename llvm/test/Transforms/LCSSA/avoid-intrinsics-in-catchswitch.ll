; RUN: opt < %s -debugify -licm -S -o /dev/null
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

define i32 @"\01?m@@YAJXZ"() personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  %n = alloca %struct.e, align 4
  %db = alloca i32, align 4
  %o = alloca %struct.d, align 1
  %q = alloca i8*, align 8
  %r = alloca i32, align 4
  %u = alloca i64, align 8
  %s = alloca %class.f, align 1
  %offset = alloca i64, align 8
  %t = alloca i64, align 8
  %status = alloca i32, align 4
  call void (...) @llvm.localescape(%class.f* nonnull %s, i32* nonnull %status)
  %0 = bitcast %struct.e* %n to i8*
  %1 = bitcast i32* %db to i8*
  %2 = getelementptr inbounds %struct.d, %struct.d* %o, i64 0, i32 0
  %3 = bitcast i8** %q to i8*
  %4 = bitcast i32* %r to i8*
  %5 = bitcast i64* %u to i8*
  %6 = getelementptr inbounds %class.f, %class.f* %s, i64 0, i32 0, i32 0
  %7 = load i32, i32* @"\01?l@@3HA", align 4, !tbaa !3
  %call = call %class.f* @"\01??0f@@QEAA@H@Z"(%class.f* nonnull %s, i32 %7)
  %8 = bitcast i64* %offset to i8*
  %9 = bitcast i64* %t to i8*
  %10 = bitcast i32* %status to i8*
  %11 = bitcast %class.f* %s to %struct.d*
  %c = getelementptr inbounds %struct.e, %struct.e* %n, i64 0, i32 0
  br label %for.cond

for.cond:                                         ; preds = %cleanup.cont, %entry
  %p.0 = phi i32 [ undef, %entry ], [ %call2, %cleanup.cont ]
  invoke void @"\01?h@@YAXPEAH0HPEAIPEAPEAEPEA_K33PEAUd@@4@Z"(i32* nonnull %db, i32* nonnull %c, i32 undef, i32* nonnull %r, i8** nonnull %q, i64* nonnull %u, i64* nonnull %offset, i64* nonnull %t, %struct.d* nonnull %11, %struct.d* nonnull %o)
          to label %__try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %for.cond
  %12 = catchswitch within none [label %__except.ret] unwind label %ehcleanup

__except.ret:                                     ; preds = %catch.dispatch
  %13 = catchpad within %12 [i8* bitcast (i32 (i8*, i8*)* @"\01?filt$0@0@m@@" to i8*)]
  catchret from %13 to label %cleanup7

__try.cont:                                       ; preds = %for.cond
  %tobool = icmp eq i32 %p.0, 0
  br i1 %tobool, label %if.end, label %cleanup7

if.end:                                           ; preds = %__try.cont
  %call2 = invoke i32 @"\01?a@@YAJXZ"()
          to label %cleanup.cont unwind label %ehcleanup

cleanup.cont:                                     ; preds = %if.end
  br label %for.cond

ehcleanup:                                        ; preds = %if.end, %catch.dispatch
  %14 = cleanuppad within none []
  %g.i = getelementptr inbounds %class.f, %class.f* %s, i64 0, i32 0
  call void @"\01??1b@@QEAA@XZ"(%class.b* nonnull %g.i) [ "funclet"(token %14) ]
  cleanupret from %14 unwind to caller

cleanup7:                                         ; preds = %__try.cont, %__except.ret
  %p.2.ph = phi i32 [ 7, %__except.ret ], [ %p.0, %__try.cont ]
  %g.i32 = getelementptr inbounds %class.f, %class.f* %s, i64 0, i32 0
  call void @"\01??1b@@QEAA@XZ"(%class.b* nonnull %g.i32)
  ret i32 %p.2.ph
}

declare %class.f* @"\01??0f@@QEAA@H@Z"(%class.f* returned, i32) unnamed_addr

define internal i32 @"\01?filt$0@0@m@@"(i8* %exception_pointers, i8* %frame_pointer) personality i8* bitcast (i32 (...)* @__C_specific_handler to i8*) {
entry:
  %0 = tail call i8* @llvm.eh.recoverfp(i8* bitcast (i32 ()* @"\01?m@@YAJXZ" to i8*), i8* %frame_pointer)
  %1 = tail call i8* @llvm.localrecover(i8* bitcast (i32 ()* @"\01?m@@YAJXZ" to i8*), i8* %0, i32 0)
  %2 = tail call i8* @llvm.localrecover(i8* bitcast (i32 ()* @"\01?m@@YAJXZ" to i8*), i8* %0, i32 1)
  %status = bitcast i8* %2 to i32*
  %agg.tmp = alloca %class.f, align 1
  %3 = bitcast i8* %exception_pointers to i32**
  %4 = load i32*, i32** %3, align 8
  %5 = load i32, i32* %4, align 4
  %6 = bitcast i8* %exception_pointers to %struct.k*
  %7 = getelementptr inbounds %class.f, %class.f* %agg.tmp, i64 0, i32 0, i32 0
  %8 = load i8, i8* %1, align 1
  store i8 %8, i8* %7, align 1
  %call = invoke i32 @"\01?j@@YAJVf@@JPEAUk@@PEAH@Z"(i8 %8, i32 %5, %struct.k* %6, i32* %status)
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %g.i = getelementptr inbounds %class.f, %class.f* %agg.tmp, i64 0, i32 0
  call void @"\01??1b@@QEAA@XZ"(%class.b* nonnull %g.i)
  ret i32 %call

ehcleanup:                                        ; preds = %entry
  %9 = cleanuppad within none []
  %g.i2 = getelementptr inbounds %class.f, %class.f* %agg.tmp, i64 0, i32 0
  call void @"\01??1b@@QEAA@XZ"(%class.b* nonnull %g.i2) [ "funclet"(token %9) ]
  cleanupret from %9 unwind to caller
}

declare i8* @llvm.eh.recoverfp(i8*, i8*)
declare i8* @llvm.localrecover(i8*, i8*, i32)
declare i32 @"\01?j@@YAJVf@@JPEAUk@@PEAH@Z"(i8, i32, %struct.k*, i32*) local_unnamed_addr
declare i32 @__C_specific_handler(...)
declare void @"\01?h@@YAXPEAH0HPEAIPEAPEAEPEA_K33PEAUd@@4@Z"(i32*, i32*, i32, i32*, i8**, i64*, i64*, i64*, %struct.d*, %struct.d*) local_unnamed_addr
declare i32 @"\01?a@@YAJXZ"() local_unnamed_addr
declare void @llvm.localescape(...)
declare void @"\01??1b@@QEAA@XZ"(%class.b*) unnamed_addr

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}

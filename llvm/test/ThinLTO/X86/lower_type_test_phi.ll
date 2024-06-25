; Test to ensure that type tests feeding assumes via a phi are lowered
; correctly after ThinLTO index-only WPD, which needs to set up the necessary
; type id summaries.

; RUN: rm -rf %t && split-file %s %t && cd %t

; RUN: opt -thinlto-bc y.ll -o y.bc
; RUN: opt -thinlto-bc z.ll -o z.bc

; RUN: llvm-lto2 run y.bc z.bc -o out \
; RUN:	-r y.bc,main,plx \
; RUN:	-r y.bc,_Z2b1v, \
; RUN:	-r y.bc,_Z2b2v, \
; RUN:	-r z.bc,_Z2b1v,pl \
; RUN:	-r z.bc,_Znwm, \
; RUN:	-r z.bc,_Z2b2v,pl \
; RUN:	-r z.bc,_ZN2D11fEv,pl \
; RUN:	-r z.bc,_ZN1B1fEv,pl \
; RUN:	-r z.bc,_ZN2D21fEv,pl \
; RUN:	-r z.bc,_ZTV2D1,pl \
; RUN:	-r z.bc,_ZTV1B,pl \
; RUN:	-r z.bc,_ZTV2D2,pl \
; RUN:	-print-after=lowertypetests -filter-print-funcs=main 2>&1 | FileCheck %s

; The first LTT should leave the type tests as is (instead of lowering
; them to false incorrectly).
; CHECK: *** IR Dump After LowerTypeTestsPass on [module] ***
; CHECK: 4:
; CHECK:   %7 = tail call i1 @llvm.type.test(ptr %6, metadata !"_ZTS2D1")
; CHECK:   br label %12
; CHECK: 8:
; CHECK:   %11 = tail call i1 @llvm.type.test(ptr %10, metadata !"_ZTS2D2")
; CHECK:   br label %12
; CHECK: 12:
; CHECK:   %13 = phi i1 [ %11, %8 ], [ %7, %4 ]

; The second LTT should lower them to true.
; CHECK: *** IR Dump After LowerTypeTestsPass on [module] ***
; CHECK-NOT: @llvm.type.test
; CHECK: 10:
; CHECK:   %11 = phi i1 [ true, %7 ], [ true, %4 ]


;--- y.ll
; ModuleID = 'y.cc'
source_filename = "y.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %tobool.not = icmp eq i32 %argc, 0
  br i1 %tobool.not, label %if.else, label %if.then

if.then:
  %call = tail call ptr @_Z2b1v()
  %vtable = load ptr, ptr %call, align 8
  %0 = tail call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS2D1")
  br label %if.end

if.else:
  %call1 = tail call ptr @_Z2b2v()
  %vtable2 = load ptr, ptr %call1, align 8
  %1 = tail call i1 @llvm.type.test(ptr %vtable2, metadata !"_ZTS2D2")
  br label %if.end

if.end:
  %.sink = phi i1 [ %1, %if.else ], [ %0, %if.then ]
  %vtable2.sink = phi ptr [ %vtable2, %if.else ], [ %vtable, %if.then ]
  %call1.sink = phi ptr [ %call1, %if.else ], [ %call, %if.then ]
  tail call void @llvm.assume(i1 %.sink)
  %2 = load ptr, ptr %vtable2.sink, align 8
  tail call void %2(ptr align 8 %call1.sink)
  ret i32 0
}

declare ptr @_Z2b1v()

declare i1 @llvm.type.test(ptr, metadata)

declare void @llvm.assume(i1)

declare ptr @_Z2b2v()

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"Virtual Function Elim", i32 0}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{!"clang version 15.0.0 (git@github.com:llvm/llvm-project.git e6f39e3f31ba88f2084a5d987f9a827aff4e17b1)"}

;--- z.ll
; ModuleID = 'z.cc'
source_filename = "z.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZN2D11fEv = comdat any

$_ZN1B1fEv = comdat any

$_ZN2D21fEv = comdat any

$_ZTV2D1 = comdat any

$_ZTV1B = comdat any

$_ZTV2D2 = comdat any

@_ZTV2D1 = linkonce_odr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN2D11fEv] }, comdat, align 8, !type !0, !type !1, !type !2, !type !3, !vcall_visibility !4
@_ZTV1B = linkonce_odr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1B1fEv] }, comdat, align 8, !type !0, !type !1, !vcall_visibility !4
@_ZTV2D2 = linkonce_odr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN2D21fEv] }, comdat, align 8, !type !0, !type !1, !type !5, !type !6, !vcall_visibility !4

define ptr @_Z2b1v() {
entry:
  %call = tail call ptr @_Znwm(i64 8)
  store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV2D1, i64 0, i32 0, i64 2), ptr %call, align 8
  ret ptr %call
}

declare ptr @_Znwm(i64)

define ptr @_Z2b2v() {
entry:
  %call = tail call ptr @_Znwm(i64 8)
  store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV2D2, i64 0, i32 0, i64 2), ptr %call, align 8
  ret ptr %call
}

define linkonce_odr void @_ZN2D11fEv(ptr %this) comdat align 2 {
entry:
  ret void
}

define linkonce_odr void @_ZN1B1fEv(ptr %this) comdat align 2 {
entry:
  ret void
}

define linkonce_odr void @_ZN2D21fEv(ptr %this) comdat align 2 {
entry:
  ret void
}

!llvm.module.flags = !{!7, !8, !9, !10, !11}
!llvm.ident = !{!12}

!0 = !{i64 16, !"_ZTS1B"}
!1 = !{i64 16, !"_ZTSM1BFvvE.virtual"}
!2 = !{i64 16, !"_ZTS2D1"}
!3 = !{i64 16, !"_ZTSM2D1FvvE.virtual"}
!4 = !{i64 1}
!5 = !{i64 16, !"_ZTS2D2"}
!6 = !{i64 16, !"_ZTSM2D2FvvE.virtual"}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{i32 1, !"Virtual Function Elim", i32 0}
!9 = !{i32 7, !"PIC Level", i32 2}
!10 = !{i32 7, !"PIE Level", i32 2}
!11 = !{i32 7, !"uwtable", i32 2}
!12 = !{!"clang version 15.0.0 (git@github.com:llvm/llvm-project.git e6f39e3f31ba88f2084a5d987f9a827aff4e17b1)"}

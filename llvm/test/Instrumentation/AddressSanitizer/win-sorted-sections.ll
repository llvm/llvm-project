; RUN: opt < %s -passes='asan-pipeline' -S | FileCheck %s

; All of these globals should pass through uninstrumented because of their
; custom section name. The .CRT section is the standard way to register custom
; initializers, and the ATL uses the ATL section to register other stuff.
; Either way, if the section has a '$' character, it is probably participating
; in section sorting, and we should probably not put a redzone around it.

; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.14.26430"

$__pobjMapEntryFirst = comdat any
$__pobjMapEntryMiddle = comdat any
$__pobjMapEntryLast = comdat any
$__crt_init_begin = comdat any
$__crt_init_callback = comdat any
$__crt_init_end = comdat any

@__pobjMapEntryFirst = weak_odr dso_local constant i8* null, section "ATL$__a", comdat, align 8
@__pobjMapEntryMiddle = weak_odr dso_local constant i8* null, section "ATL$__m", comdat, align 8
@__pobjMapEntryLast = weak_odr dso_local constant i8* null, section "ATL$__z", comdat, align 8
@__crt_init_begin = weak_odr dso_local constant i8* null, section ".CRT$XCA", comdat, align 8
@__crt_init_callback = weak_odr dso_local constant i8* null, section ".CRT$XCU", comdat, align 8
@__crt_init_end = weak_odr dso_local constant i8* null, section ".CRT$XCZ", comdat, align 8

; CHECK: @__pobjMapEntryFirst = weak_odr dso_local constant i8* null, section "ATL$__a", comdat, align 8
; CHECK: @__pobjMapEntryMiddle = weak_odr dso_local constant i8* null, section "ATL$__m", comdat, align 8
; CHECK: @__pobjMapEntryLast = weak_odr dso_local constant i8* null, section "ATL$__z", comdat, align 8
; CHECK: @__crt_init_begin = weak_odr dso_local constant i8* null, section ".CRT$XCA", comdat, align 8
; CHECK: @__crt_init_callback = weak_odr dso_local constant i8* null, section ".CRT$XCU", comdat, align 8
; CHECK: @__crt_init_end = weak_odr dso_local constant i8* null, section ".CRT$XCZ", comdat, align 8

!llvm.asan.globals = !{!0, !2, !4, !6, !8, !10}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = !{i8** @__pobjMapEntryFirst, !1, !"__pobjMapEntryFirst", i1 false, i1 false}
!1 = !{!"t.c", i32 6, i32 61}
!2 = !{i8** @__pobjMapEntryMiddle, !3, !"__pobjMapEntryMiddle", i1 false, i1 false}
!3 = !{!"t.c", i32 7, i32 61}
!4 = !{i8** @__pobjMapEntryLast, !5, !"__pobjMapEntryLast", i1 false, i1 false}
!5 = !{!"t.c", i32 8, i32 61}
!6 = !{i8** @__crt_init_begin, !7, !"__crt_init_begin", i1 false, i1 false}
!7 = !{!"t.c", i32 16, i32 62}
!8 = !{i8** @__crt_init_callback, !9, !"__crt_init_callback", i1 false, i1 false}
!9 = !{!"t.c", i32 17, i32 62}
!10 = !{i8** @__crt_init_end, !11, !"__crt_init_end", i1 false, i1 false}
!11 = !{!"t.c", i32 18, i32 62}
!12 = !{i32 1, !"wchar_size", i32 2}
!13 = !{i32 7, !"PIC Level", i32 2}
!14 = !{!"clang version 7.0.0 "}

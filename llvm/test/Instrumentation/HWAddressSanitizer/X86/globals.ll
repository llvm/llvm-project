; RUN: opt < %s -S -passes=hwasan -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

; CHECK: @__start_hwasan_globals = external hidden constant [0 x i8]
; CHECK: @__stop_hwasan_globals = external hidden constant [0 x i8]

; CHECK: @hwasan.note = private constant { i32, i32, i32, [8 x i8], i32, i32 } { i32 8, i32 8, i32 3, [8 x i8] c"LLVM\00\00\00\00", i32 trunc (i64 sub (i64 ptrtoint (ptr @__start_hwasan_globals to i64), i64 ptrtoint (ptr @hwasan.note to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @__stop_hwasan_globals to i64), i64 ptrtoint (ptr @hwasan.note to i64)) to i32) }, section ".note.hwasan.globals", comdat($hwasan.module_ctor), align 4

; CHECK: @hwasan.dummy.global = private constant [0 x i8] zeroinitializer, section "hwasan_globals", comdat($hwasan.module_ctor), !associated [[NOTE:![0-9]+]]

; CHECK: @four.hwasan = private global { i32, [12 x i8] } { i32 1, [12 x i8] c"\00\00\00\00\00\00\00\00\00\00\00\10" }, align 16
; CHECK: @four.hwasan.descriptor = private constant { i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint (ptr @four.hwasan to i64), i64 ptrtoint (ptr @four.hwasan.descriptor to i64)) to i32), i32 268435460 }, section "hwasan_globals", !associated [[FOUR:![0-9]+]]

; CHECK: @sixteen.hwasan = private global [16 x i8] zeroinitializer, align 16
; CHECK: @sixteen.hwasan.descriptor = private constant { i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint (ptr @sixteen.hwasan to i64), i64 ptrtoint (ptr @sixteen.hwasan.descriptor to i64)) to i32), i32 285212688 }, section "hwasan_globals", !associated [[SIXTEEN:![0-9]+]]

; CHECK: @huge.hwasan = private global [16777232 x i8] zeroinitializer, align 16
; CHECK: @huge.hwasan.descriptor = private constant { i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint (ptr @huge.hwasan to i64), i64 ptrtoint (ptr @huge.hwasan.descriptor to i64)) to i32), i32 318767088 }, section "hwasan_globals", !associated [[HUGE:![0-9]+]]
; CHECK: @huge.hwasan.descriptor.1 = private constant { i32, i32 } { i32 trunc (i64 add (i64 sub (i64 ptrtoint (ptr @huge.hwasan to i64), i64 ptrtoint (ptr @huge.hwasan.descriptor.1 to i64)), i64 16777200) to i32), i32 301989920 }, section "hwasan_globals", !associated [[HUGE]]

; CHECK: @four = alias i32, inttoptr (i64 add (i64 ptrtoint (ptr @four.hwasan to i64), i64 2305843009213693952) to ptr)
; CHECK: @sixteen = alias [16 x i8], inttoptr (i64 add (i64 ptrtoint (ptr @sixteen.hwasan to i64), i64 2449958197289549824) to ptr)
; CHECK: @huge = alias [16777232 x i8], inttoptr (i64 add (i64 ptrtoint (ptr @huge.hwasan to i64), i64 2594073385365405696) to ptr)

; CHECK: [[NOTE]] = !{ptr @hwasan.note}
; CHECK: [[FOUR]] = !{ptr @four.hwasan}
; CHECK: [[SIXTEEN]] = !{ptr @sixteen.hwasan}
; CHECK: [[HUGE]] = !{ptr @huge.hwasan}

source_filename = "foo"

@four = global i32 1
@sixteen = global [16 x i8] zeroinitializer
@huge = global [16777232 x i8] zeroinitializer

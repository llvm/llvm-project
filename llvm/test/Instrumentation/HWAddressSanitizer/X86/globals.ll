; RUN: opt < %s -S -passes=hwasan -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

; CHECK: @__start_hwasan_globals = external hidden constant [0 x i8]
; CHECK: @__stop_hwasan_globals = external hidden constant [0 x i8]

; CHECK: @hwasan.note = private constant { i32, i32, i32, [8 x i8], i32, i32 } { i32 8, i32 8, i32 3, [8 x i8] c"LLVM\00\00\00\00", i32 trunc (i64 sub (i64 ptrtoint ([0 x i8]* @__start_hwasan_globals to i64), i64 ptrtoint ({ i32, i32, i32, [8 x i8], i32, i32 }* @hwasan.note to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint ([0 x i8]* @__stop_hwasan_globals to i64), i64 ptrtoint ({ i32, i32, i32, [8 x i8], i32, i32 }* @hwasan.note to i64)) to i32) }, section ".note.hwasan.globals", comdat($hwasan.module_ctor), align 4

; CHECK: @hwasan.dummy.global = private constant [0 x i8] zeroinitializer, section "hwasan_globals", comdat($hwasan.module_ctor), !associated [[NOTE:![0-9]+]]

; CHECK: @four.hwasan = private global { i32, [12 x i8] } { i32 1, [12 x i8] c"\00\00\00\00\00\00\00\00\00\00\00," }, align 16
; CHECK: @four.hwasan.descriptor = private constant { i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint ({ i32, [12 x i8] }* @four.hwasan to i64), i64 ptrtoint ({ i32, i32 }* @four.hwasan.descriptor to i64)) to i32), i32 738197508 }, section "hwasan_globals", !associated [[FOUR:![0-9]+]]

; CHECK: @sixteen.hwasan = private global [16 x i8] zeroinitializer, align 16
; CHECK: @sixteen.hwasan.descriptor = private constant { i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint ([16 x i8]* @sixteen.hwasan to i64), i64 ptrtoint ({ i32, i32 }* @sixteen.hwasan.descriptor to i64)) to i32), i32 754974736 }, section "hwasan_globals", !associated [[SIXTEEN:![0-9]+]]

; CHECK: @huge.hwasan = private global [16777232 x i8] zeroinitializer, align 16
; CHECK: @huge.hwasan.descriptor = private constant { i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint ([16777232 x i8]* @huge.hwasan to i64), i64 ptrtoint ({ i32, i32 }* @huge.hwasan.descriptor to i64)) to i32), i32 788529136 }, section "hwasan_globals", !associated [[HUGE:![0-9]+]]
; CHECK: @huge.hwasan.descriptor.1 = private constant { i32, i32 } { i32 trunc (i64 add (i64 sub (i64 ptrtoint ([16777232 x i8]* @huge.hwasan to i64), i64 ptrtoint ({ i32, i32 }* @huge.hwasan.descriptor.1 to i64)), i64 16777200) to i32), i32 771751968 }, section "hwasan_globals", !associated [[HUGE]]

; CHECK: @four = alias i32, inttoptr (i64 add (i64 ptrtoint ({ i32, [12 x i8] }* @four.hwasan to i64), i64 6341068275337658368) to i32*)
; CHECK: @sixteen = alias [16 x i8], inttoptr (i64 add (i64 ptrtoint ([16 x i8]* @sixteen.hwasan to i64), i64 6485183463413514240) to [16 x i8]*)
; CHECK: @huge = alias [16777232 x i8], inttoptr (i64 add (i64 ptrtoint ([16777232 x i8]* @huge.hwasan to i64), i64 6629298651489370112) to [16777232 x i8]*)

; CHECK: [[NOTE]] = !{{{{}} i32, i32, i32, [8 x i8], i32, i32 }* @hwasan.note}
; CHECK: [[FOUR]] = !{{{{}} i32, [12 x i8] }* @four.hwasan}
; CHECK: [[SIXTEEN]] = !{[16 x i8]* @sixteen.hwasan}
; CHECK: [[HUGE]] = !{[16777232 x i8]* @huge.hwasan}

source_filename = "foo"

@four = global i32 1
@sixteen = global [16 x i8] zeroinitializer
@huge = global [16777232 x i8] zeroinitializer

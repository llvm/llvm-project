; RUN: opt < %s -S -passes=hwasan -mtriple=aarch64--linux-android29 | FileCheck --check-prefixes=CHECK,CHECK29 %s
; RUN: opt < %s -S -passes=hwasan -mtriple=aarch64--linux-android30 | FileCheck --check-prefixes=CHECK,CHECK30 %s

; CHECK29: @four = global

; CHECK: @specialcaselisted = global i16 2, no_sanitize_hwaddress

; CHECK: @__start_hwasan_globals = external hidden constant [0 x i8]
; CHECK: @__stop_hwasan_globals = external hidden constant [0 x i8]

; CHECK: @hwasan.note = private constant { i32, i32, i32, [8 x i8], i32, i32 } { i32 8, i32 8, i32 3, [8 x i8] c"LLVM\00\00\00\00", i32 trunc (i64 sub (i64 ptrtoint (ptr @__start_hwasan_globals to i64), i64 ptrtoint (ptr @hwasan.note to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @__stop_hwasan_globals to i64), i64 ptrtoint (ptr @hwasan.note to i64)) to i32) }, section ".note.hwasan.globals", comdat($hwasan.module_ctor), align 4

; CHECK: @hwasan.dummy.global = private constant [0 x i8] zeroinitializer, section "hwasan_globals", comdat($hwasan.module_ctor), !associated [[NOTE:![0-9]+]]

; CHECK30: @four.hwasan = private global { i32, [12 x i8] } { i32 1, [12 x i8] c"\00\00\00\00\00\00\00\00\00\00\00\AC" }, align 16
; CHECK30: @four.hwasan.descriptor = private constant { i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint (ptr @four.hwasan to i64), i64 ptrtoint (ptr @four.hwasan.descriptor to i64)) to i32), i32 -1409286140 }, section "hwasan_globals", !associated [[FOUR:![0-9]+]]

; CHECK30: @sixteen.hwasan = private global [16 x i8] zeroinitializer, align 16
; CHECK30: @sixteen.hwasan.descriptor = private constant { i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint (ptr @sixteen.hwasan to i64), i64 ptrtoint (ptr @sixteen.hwasan.descriptor to i64)) to i32), i32 -1392508912 }, section "hwasan_globals", !associated [[SIXTEEN:![0-9]+]]

; CHECK30: @huge.hwasan = private global [16777232 x i8] zeroinitializer, align 16
; CHECK30: @huge.hwasan.descriptor = private constant { i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint (ptr @huge.hwasan to i64), i64 ptrtoint (ptr @huge.hwasan.descriptor to i64)) to i32), i32 -1358954512 }, section "hwasan_globals", !associated [[HUGE:![0-9]+]]
; CHECK30: @huge.hwasan.descriptor.1 = private constant { i32, i32 } { i32 trunc (i64 add (i64 sub (i64 ptrtoint (ptr @huge.hwasan to i64), i64 ptrtoint (ptr @huge.hwasan.descriptor.1 to i64)), i64 16777200) to i32), i32 -1375731680 }, section "hwasan_globals", !associated [[HUGE]]

; CHECK30: @four = alias i32, inttoptr (i64 add (i64 ptrtoint (ptr @four.hwasan to i64), i64 -6052837899185946624) to ptr)
; CHECK30: @sixteen = alias [16 x i8], inttoptr (i64 add (i64 ptrtoint (ptr @sixteen.hwasan to i64), i64 -5980780305148018688) to ptr)
; CHECK30: @huge = alias [16777232 x i8], inttoptr (i64 add (i64 ptrtoint (ptr @huge.hwasan to i64), i64 -5908722711110090752) to ptr)

; CHECK: [[NOTE]] = !{ptr @hwasan.note}
; CHECK30: [[FOUR]] = !{ptr @four.hwasan}
; CHECK30: [[SIXTEEN]] = !{ptr @sixteen.hwasan}
; CHECK30: [[HUGE]] = !{ptr @huge.hwasan}

source_filename = "foo"

@four = global i32 1
@sixteen = global [16 x i8] zeroinitializer
@huge = global [16777232 x i8] zeroinitializer
@specialcaselisted = global i16 2, no_sanitize_hwaddress

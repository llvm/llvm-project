; RUN: llc < %s -mtriple=arm64-linux-gnu | FileCheck %s

@x1 = external hidden global [2 x i64]
@x2 = external hidden global [16777216 x i64]
@x3 = external hidden global { [9 x i8*], [8 x i8*] }

define i64 @f1() {
  ; CHECK: f1:
  ; CHECK: adrp x8, x1+16
  ; CHECK: ldr x0, [x8, :lo12:x1+16]
  %l = load i64, i64* getelementptr ([2 x i64], [2 x i64]* @x1, i64 0, i64 2)
  ret i64 %l
}

define i64 @f2() {
  ; CHECK: f2:
  ; CHECK: adrp x8, x1
  ; CHECK: add x8, x8, :lo12:x1
  ; CHECK: ldr x0, [x8, #24]
  %l = load i64, i64* getelementptr ([2 x i64], [2 x i64]* @x1, i64 0, i64 3)
  ret i64 %l
}

define i64 @f3() {
  ; CHECK: f3:
  ; CHECK: adrp x8, x1+1
  ; CHECK: add x8, x8, :lo12:x1+1
  ; CHECK: ldr x0, [x8]
  %l = load i64, i64* bitcast (i8* getelementptr (i8, i8* bitcast ([2 x i64]* @x1 to i8*), i64 1) to i64*)
  ret i64 %l
}

define [2 x i64] @f4() {
  ; CHECK: f4:
  ; CHECK: adrp x8, x2+8
  ; CHECK: add x8, x8, :lo12:x2+8
  ; CHECK: ldp x0, x1, [x8]
  %l = load [2 x i64], [2 x i64]* bitcast (i8* getelementptr (i8, i8* bitcast ([16777216 x i64]* @x2 to i8*), i64 8) to [2 x i64]*)
  ret [2 x i64] %l
}

define i64 @f5() {
  ; CHECK: f5:
  ; CHECK: adrp x8, x2+2097144
  ; CHECK: ldr x0, [x8, :lo12:x2+2097144]
  ; CHECK: ret
  %l = load i64, i64* getelementptr ([16777216 x i64], [16777216 x i64]* @x2, i64 0, i64 262143)
  ret i64 %l
}

define i64 @f6() {
  ; CHECK: f6:
  ; CHECK: adrp x8, x2
  ; CHECK: add x8, x8, :lo12:x2
  ; CHECK: orr w9, wzr, #0x200000
  ; CHECK: ldr x0, [x8, x9]
  ; CHECK: ret
  %l = load i64, i64* getelementptr ([16777216 x i64], [16777216 x i64]* @x2, i64 0, i64 262144)
  ret i64 %l
}

define i32 @f7() {
entry:
  ; CHECK: f7
  ; CHECK: adrp x8, x3+108
  ; CHECK: ldr w0, [x8, :lo12:x3+108]
  %l = load i32, i32* getelementptr (i32, i32* inttoptr (i64 trunc (i128 lshr (i128 bitcast (<2 x i64> <i64 undef, i64 ptrtoint (i8** getelementptr inbounds ({ [9 x i8*], [8 x i8*] }, { [9 x i8*], [8 x i8*] }* @x3, i64 0, inrange i32 1, i64 2) to i64)> to i128), i128 64) to i64) to i32*), i64 5)
  ret i32 %l
}

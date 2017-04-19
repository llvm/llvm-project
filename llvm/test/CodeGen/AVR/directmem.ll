; RUN: llc -mattr=sram,addsubiw < %s -march=avr | FileCheck %s

@char = common global i8 0
@char.array = common global [3 x i8] zeroinitializer
@char.static = internal global i8 0

@int = common global i16 0
@int.array = common global [3 x i16] zeroinitializer
@int.static = internal global i16 0

@long = common global i32 0
@long.array = common global [3 x i32] zeroinitializer
@long.static = internal global i32 0

@longlong = common global i64 0
@longlong.array = common global [3 x i64] zeroinitializer
@longlong.static = internal global i64 0

define void @global8_store() {
; CHECK-LABEL: global8_store:
; CHECK: ldi [[REG:r[0-9]+]], 6
; CHECK: sts char, [[REG]]
  store i8 6, i8* @char
  ret void
}

define i8 @global8_load() {
; CHECK-LABEL: global8_load:
; CHECK: lds r24, char
  %result = load i8, i8* @char
  ret i8 %result
}

define void @array8_store() {
; CHECK-LABEL: array8_store:
; CHECK: ldi [[REG2:r[0-9]+]], 2
; CHECK: sts char.array+1, [[REG2]]
; CHECK: ldi [[REG1:r[0-9]+]], 1
; CHECK: sts char.array, [[REG1]]
; CHECK: ldi [[REG:r[0-9]+]], 3
; CHECK: sts char.array+2, [[REG]]
  store i8 1, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @char.array, i32 0, i64 0)
  store i8 2, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @char.array, i32 0, i64 1)
  store i8 3, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @char.array, i32 0, i64 2)
  ret void
}

define i8 @array8_load() {
; CHECK-LABEL: array8_load:
; CHECK: lds r24, char.array+2
  %result = load i8, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @char.array, i32 0, i64 2)
  ret i8 %result
}

define i8 @static8_inc() {
; CHECK-LABEL: static8_inc:
; CHECK: lds r24, char.static
; CHECK: inc r24
; CHECK: sts char.static, r24
  %1 = load i8, i8* @char.static
  %inc = add nsw i8 %1, 1
  store i8 %inc, i8* @char.static
  ret i8 %inc
}

define void @global16_store() {
; CHECK-LABEL: global16_store:
; CHECK: ldi [[REG1:r[0-9]+]], 187
; CHECK: ldi [[REG2:r[0-9]+]], 170
; CHECK: sts int+1, [[REG2]]
; CHECK: sts int, [[REG1]]
  store i16 43707, i16* @int
  ret void
}

define i16 @global16_load() {
; CHECK-LABEL: global16_load:
; CHECK: lds r24, int
; CHECK: lds r25, int+1
  %result = load i16, i16* @int
  ret i16 %result
}

define void @array16_store() {
; CHECK-LABEL: array16_store:

; CHECK: ldi [[REG1:r[0-9]+]], 204
; CHECK: ldi [[REG2:r[0-9]+]], 170
; CHECK: sts int.array+3, [[REG2]]
; CHECK: sts int.array+2, [[REG1]]

; CHECK: ldi [[REG1:r[0-9]+]], 187
; CHECK: ldi [[REG2:r[0-9]+]], 170
; CHECK: sts int.array+1, [[REG2]]
; CHECK: sts int.array, [[REG1]]


; CHECK: ldi [[REG1:r[0-9]+]], 221
; CHECK: ldi [[REG2:r[0-9]+]], 170
; CHECK: sts int.array+5, [[REG2]]
; CHECK: sts int.array+4, [[REG1]]
  store i16 43707, i16* getelementptr inbounds ([3 x i16], [3 x i16]* @int.array, i32 0, i64 0)
  store i16 43724, i16* getelementptr inbounds ([3 x i16], [3 x i16]* @int.array, i32 0, i64 1)
  store i16 43741, i16* getelementptr inbounds ([3 x i16], [3 x i16]* @int.array, i32 0, i64 2)
  ret void
}

define i16 @array16_load() {
; CHECK-LABEL: array16_load:
; CHECK: lds r24, int.array+4
; CHECK: lds r25, int.array+5
  %result = load i16, i16* getelementptr inbounds ([3 x i16], [3 x i16]* @int.array, i32 0, i64 2)
  ret i16 %result
}

define i16 @static16_inc() {
; CHECK-LABEL: static16_inc:
; CHECK: lds r24, int.static
; CHECK: lds r25, int.static+1
; CHECK: adiw r24, 1
; CHECK: sts int.static+1, r25
; CHECK: sts int.static, r24
  %1 = load i16, i16* @int.static
  %inc = add nsw i16 %1, 1
  store i16 %inc, i16* @int.static
  ret i16 %inc
}

define void @global32_store() {
; CHECK-LABEL: global32_store:
; CHECK: ldi [[REG1:r[0-9]+]], 187
; CHECK: ldi [[REG2:r[0-9]+]], 170
; CHECK: sts long+3, [[REG2]]
; CHECK: sts long+2, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 221
; CHECK: ldi [[REG2:r[0-9]+]], 204
; CHECK: sts long+1, [[REG2]]
; CHECK: sts long, [[REG1]]
  store i32 2864434397, i32* @long
  ret void
}

define i32 @global32_load() {
; CHECK-LABEL: global32_load:
; CHECK: lds r22, long
; CHECK: lds r23, long+1
; CHECK: lds r24, long+2
; CHECK: lds r25, long+3
  %result = load i32, i32* @long
  ret i32 %result
}

define void @array32_store() {
; CHECK-LABEL: array32_store:
; CHECK: ldi [[REG1:r[0-9]+]], 102
; CHECK: ldi [[REG2:r[0-9]+]], 85
; CHECK: sts long.array+7, [[REG2]]
; CHECK: sts long.array+6, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 136
; CHECK: ldi [[REG2:r[0-9]+]], 119
; CHECK: sts long.array+5, [[REG2]]
; CHECK: sts long.array+4, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 27
; CHECK: ldi [[REG2:r[0-9]+]], 172
; CHECK: sts long.array+3, [[REG2]]
; CHECK: sts long.array+2, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 68
; CHECK: ldi [[REG2:r[0-9]+]], 13
; CHECK: sts long.array+1, [[REG2]]
; CHECK: sts long.array, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 170
; CHECK: ldi [[REG2:r[0-9]+]], 153
; CHECK: sts long.array+11, [[REG2]]
; CHECK: sts long.array+10, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 204
; CHECK: ldi [[REG2:r[0-9]+]], 187
; CHECK: sts long.array+9, [[REG2]]
; CHECK: sts long.array+8, [[REG1]]
  store i32 2887454020, i32* getelementptr inbounds ([3 x i32], [3 x i32]* @long.array, i32 0, i64 0)
  store i32 1432778632, i32* getelementptr inbounds ([3 x i32], [3 x i32]* @long.array, i32 0, i64 1)
  store i32 2578103244, i32* getelementptr inbounds ([3 x i32], [3 x i32]* @long.array, i32 0, i64 2)
  ret void
}

define i32 @array32_load() {
; CHECK-LABEL: array32_load:
; CHECK: lds r22, long.array+8
; CHECK: lds r23, long.array+9
; CHECK: lds r24, long.array+10
; CHECK: lds r25, long.array+11
  %result = load i32, i32* getelementptr inbounds ([3 x i32], [3 x i32]* @long.array, i32 0, i64 2)
  ret i32 %result
}

define i32 @static32_inc() {
; CHECK-LABEL: static32_inc:
; CHECK: lds r22, long.static
; CHECK: lds r23, long.static+1
; CHECK: lds r24, long.static+2
; CHECK: lds r25, long.static+3
; CHECK: subi r22, 255
; CHECK: sbci r23, 255
; CHECK: sbci r24, 255
; CHECK: sbci r25, 255
; CHECK: sts long.static+3, r25
; CHECK: sts long.static+2, r24
; CHECK: sts long.static+1, r23
; CHECK: sts long.static, r22
  %1 = load i32, i32* @long.static
  %inc = add nsw i32 %1, 1
  store i32 %inc, i32* @long.static
  ret i32 %inc
}

define void @global64_store() {
; CHECK-LABEL: global64_store:
; CHECK: ldi [[REG1:r[0-9]+]], 34
; CHECK: ldi [[REG2:r[0-9]+]], 17
; CHECK: sts longlong+7, [[REG2]]
; CHECK: sts longlong+6, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 68
; CHECK: ldi [[REG2:r[0-9]+]], 51
; CHECK: sts longlong+5, [[REG2]]
; CHECK: sts longlong+4, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 102
; CHECK: ldi [[REG2:r[0-9]+]], 85
; CHECK: sts longlong+3, [[REG2]]
; CHECK: sts longlong+2, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 136
; CHECK: ldi [[REG2:r[0-9]+]], 119
; CHECK: sts longlong+1, [[REG2]]
; CHECK: sts longlong, [[REG1]]
  store i64 1234605616436508552, i64* @longlong
  ret void
}

define i64 @global64_load() {
; CHECK-LABEL: global64_load:
; CHECK: lds r18, longlong
; CHECK: lds r19, longlong+1
; CHECK: lds r20, longlong+2
; CHECK: lds r21, longlong+3
; CHECK: lds r22, longlong+4
; CHECK: lds r23, longlong+5
; CHECK: lds r24, longlong+6
; CHECK: lds r25, longlong+7
  %result = load i64, i64* @longlong
  ret i64 %result
}

define void @array64_store() {
; CHECK-LABEL: array64_store:
; CHECK: ldi [[REG1:r[0-9]+]], 34
; CHECK: ldi [[REG2:r[0-9]+]], 17
; CHECK: sts longlong.array+7, [[REG2]]
; CHECK: sts longlong.array+6, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 68
; CHECK: ldi [[REG2:r[0-9]+]], 51
; CHECK: sts longlong.array+5, [[REG2]]
; CHECK: sts longlong.array+4, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 102
; CHECK: ldi [[REG2:r[0-9]+]], 85
; CHECK: sts longlong.array+3, [[REG2]]
; CHECK: sts longlong.array+2, [[REG1]]
; CHECK: ldi [[REG1:r[0-9]+]], 136
; CHECK: ldi [[REG2:r[0-9]+]], 119
; CHECK: sts longlong.array+1, [[REG2]]
; CHECK: sts longlong.array, [[REG1]]
  store i64 1234605616436508552, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @longlong.array, i64 0, i64 0)
  store i64 81985529216486895, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @longlong.array, i64 0, i64 1)
  store i64 1836475854449306472, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @longlong.array, i64 0, i64 2)
  ret void
}

define i64 @array64_load() {
; CHECK-LABEL: array64_load:
; CHECK: lds r18, longlong.array+16
; CHECK: lds r19, longlong.array+17
; CHECK: lds r20, longlong.array+18
; CHECK: lds r21, longlong.array+19
; CHECK: lds r22, longlong.array+20
; CHECK: lds r23, longlong.array+21
; CHECK: lds r24, longlong.array+22
; CHECK: lds r25, longlong.array+23
  %result = load i64, i64* getelementptr inbounds ([3 x i64], [3 x i64]* @longlong.array, i64 0, i64 2)
  ret i64 %result
}

define i64 @static64_inc() {
; CHECK-LABEL: static64_inc:
; CHECK: lds r18, longlong.static
; CHECK: lds r19, longlong.static+1
; CHECK: lds r20, longlong.static+2
; CHECK: lds r21, longlong.static+3
; CHECK: lds r22, longlong.static+4
; CHECK: lds r23, longlong.static+5
; CHECK: lds r24, longlong.static+6
; CHECK: lds r25, longlong.static+7
; CHECK: subi r18, 255
; CHECK: sbci r19, 255
; CHECK: sbci r20, 255
; CHECK: sbci r21, 255
; CHECK: sbci r22, 255
; CHECK: sbci r23, 255
; CHECK: sbci r24, 255
; CHECK: sbci r25, 255
; CHECK: sts longlong.static+7, r25
; CHECK: sts longlong.static+6, r24
; CHECK: sts longlong.static+5, r23
; CHECK: sts longlong.static+4, r22
; CHECK: sts longlong.static+3, r21
; CHECK: sts longlong.static+2, r20
; CHECK: sts longlong.static+1, r19
; CHECK: sts longlong.static, r18
  %1 = load i64, i64* @longlong.static
  %inc = add nsw i64 %1, 1
  store i64 %inc, i64* @longlong.static
  ret i64 %inc
}

define i8 @constantaddr_read8() {
; CHECK-LABEL: constantaddr_read8:
; CHECK: lds r24, 1234
  %1 = load i8, i8* inttoptr (i16 1234 to i8*)
  ret i8 %1
}

define i16 @constantaddr_read16() {
; CHECK-LABEL: constantaddr_read16:
; CHECK: lds r24, 1234
; CHECK: lds r25, 1235
  %1 = load i16, i16* inttoptr (i16 1234 to i16*)
  ret i16 %1
}

define void @constantaddr_write8() {
; CHECK-LABEL: constantaddr_write8:
; CHECK: sts 1234
  store i8 22, i8* inttoptr (i16 1234 to i8*)
  ret void
}

define void @constantaddr_write16() {
; CHECK-LABEL: constantaddr_write16:
; CHECK: sts 1235
; CHECK: sts 1234
  store i16 2222, i16* inttoptr (i16 1234 to i16*)
  ret void
}

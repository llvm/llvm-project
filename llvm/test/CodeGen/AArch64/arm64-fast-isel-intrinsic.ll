; RUN: llc -O0 -fast-isel -fast-isel-abort=1 -verify-machineinstrs -relocation-model=dynamic-no-pic -mtriple=arm64-apple-ios < %s | FileCheck %s --check-prefix=ARM64

@message = global [80 x i8] c"The LLVM Compiler Infrastructure\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 16
@temp = common global [80 x i8] zeroinitializer, align 16

define void @t1() {
; ARM64-LABEL: t1
; ARM64: adrp x8, _message@PAGE
; ARM64: add x0, x8, _message@PAGEOFF
; ARM64: mov [[REG:w[0-9]+]], wzr
; ARM64: mov x2, #80
; ARM64: uxtb w1, [[REG]]
; ARM64: bl _memset
  call void @llvm.memset.p0.i64(ptr align 16 @message, i8 0, i64 80, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1)

define void @t2() {
; ARM64-LABEL: t2
; ARM64: adrp x8, _temp@GOTPAGE
; ARM64: ldr x0, [x8, _temp@GOTPAGEOFF]
; ARM64: adrp x8, _message@PAGE
; ARM64: add x1, x8, _message@PAGEOFF
; ARM64: mov x2, #80
; ARM64: bl _memcpy
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 @temp, ptr align 16 @message, i64 80, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1)

define void @t3() {
; ARM64-LABEL: t3
; ARM64: adrp x8, _temp@GOTPAGE
; ARM64: ldr x0, [x8, _temp@GOTPAGEOFF]
; ARM64: adrp x8, _message@PAGE
; ARM64: add x1, x8, _message@PAGEOFF
; ARM64: mov x2, #20
; ARM64: bl _memmove
  call void @llvm.memmove.p0.p0.i64(ptr align 16 @temp, ptr align 16 @message, i64 20, i1 false)
  ret void
}

declare void @llvm.memmove.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1)

define void @t4() {
; ARM64-LABEL: t4
; ARM64: adrp x8, _temp@GOTPAGE
; ARM64: ldr [[REG0:x[0-9]+]], [x8, _temp@GOTPAGEOFF]
; ARM64: adrp [[REG1:x[0-9]+]], _message@PAGE
; ARM64: add [[REG2:x[0-9]+]], [[REG1]], _message@PAGEOFF
; ARM64: ldr x10, [[[REG2]]]
; ARM64: str x10, [[[REG0]]]
; ARM64: ldr x10, [[[REG2]], #8]
; ARM64: str x10, [[[REG0]], #8]
; ARM64: ldrb [[REG3:w[0-9]+]], [[[REG2]], #16]
; ARM64: strb [[REG3]], [[[REG0]], #16]
; ARM64: ret
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 @temp, ptr align 16 @message, i64 17, i1 false)
  ret void
}

define void @t5() {
; ARM64-LABEL: t5
; ARM64: adrp x8, _temp@GOTPAGE
; ARM64: ldr [[REG0:x[0-9]+]], [x8, _temp@GOTPAGEOFF]
; ARM64: adrp [[REG3:x[0-9]+]], _message@PAGE
; ARM64: add [[REG1:x[0-9]+]], [[REG3]], _message@PAGEOFF
; ARM64: ldr x10, [[[REG1]]]
; ARM64: str x10, [[[REG0]]]
; ARM64: ldr x10, [[[REG1]], #8]
; ARM64: str x10, [[[REG0]], #8]
; ARM64: ldrb [[REG4:w[0-9]+]], [[[REG1]], #16]
; ARM64: strb [[REG4]], [[[REG0]], #16]
; ARM64: ret
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 @temp, ptr align 8 @message, i64 17, i1 false)
  ret void
}

define void @t6() {
; ARM64-LABEL: t6
; ARM64: adrp x8, _temp@GOTPAGE
; ARM64: ldr [[REG0:x[0-9]+]], [x8, _temp@GOTPAGEOFF]
; ARM64: adrp [[REG1:x[0-9]+]], _message@PAGE
; ARM64: add [[REG2:x[0-9]+]], [[REG1]], _message@PAGEOFF
; ARM64: ldr w10, [[[REG2]]]
; ARM64: str w10, [[[REG0]]]
; ARM64: ldr w10, [[[REG2]], #4]
; ARM64: str w10, [[[REG0]], #4]
; ARM64: ldrb [[REG3:w[0-9]+]], [[[REG2]], #8]
; ARM64: strb [[REG3]], [[[REG0]], #8]
; ARM64: ret
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 @temp, ptr align 4 @message, i64 9, i1 false)
  ret void
}

define void @t7() {
; ARM64-LABEL: t7
; ARM64: adrp x8, _temp@GOTPAGE
; ARM64: ldr [[REG0:x[0-9]+]], [x8, _temp@GOTPAGEOFF]
; ARM64: adrp [[REG1:x[0-9]+]], _message@PAGE
; ARM64: add [[REG2:x[0-9]+]], [[REG1]], _message@PAGEOFF
; ARM64: ldrh w10, [[[REG2]]]
; ARM64: strh w10, [[[REG0]]]
; ARM64: ldrh w10, [[[REG2]], #2]
; ARM64: strh w10, [[[REG0]], #2]
; ARM64: ldrh w10, [[[REG2]], #4]
; ARM64: strh w10, [[[REG0]], #4]
; ARM64: ldrb [[REG3:w[0-9]+]], [[[REG2]], #6]
; ARM64: strb [[REG3]], [[[REG0]], #6]
; ARM64: ret
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 @temp, ptr align 2 @message, i64 7, i1 false)
  ret void
}

define void @t8() {
; ARM64-LABEL: t8
; ARM64: adrp x8, _temp@GOTPAGE
; ARM64: ldr [[REG0:x[0-9]+]], [x8, _temp@GOTPAGEOFF]
; ARM64: adrp [[REG1:x[0-9]+]], _message@PAGE
; ARM64: add [[REG2:x[0-9]+]], [[REG1:x[0-9]+]], _message@PAGEOFF
; ARM64: ldrb w10, [[[REG2]]]
; ARM64: strb w10, [[[REG0]]]
; ARM64: ldrb w10, [[[REG2]], #1]
; ARM64: strb w10, [[[REG0]], #1]
; ARM64: ldrb w10, [[[REG2]], #2]
; ARM64: strb w10, [[[REG0]], #2]
; ARM64: ldrb [[REG3:w[0-9]+]], [[[REG2]], #3]
; ARM64: strb [[REG3]], [[[REG0]], #3]
; ARM64: ret
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 @temp, ptr align 1 @message, i64 4, i1 false)
  ret void
}

define void @test_distant_memcpy(ptr %dst) {
; ARM64-LABEL: test_distant_memcpy:
; ARM64: mov [[ARRAY:x[0-9]+]], sp
; ARM64: mov [[OFFSET:x[0-9]+]], #8000
; ARM64: add x[[ADDR:[0-9]+]], [[ARRAY]], [[OFFSET]]
; ARM64: ldrb [[BYTE:w[0-9]+]], [x[[ADDR]]]
; ARM64: strb [[BYTE]], [x0]
  %array = alloca i8, i32 8192
  %elem = getelementptr i8, ptr %array, i32 8000
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %elem, i64 1, i1 false)
  ret void
}

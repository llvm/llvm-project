; RUN: llc -O0 -fast-isel-abort=1 -verify-machineinstrs -relocation-model=dynamic-no-pic -mtriple=arm64-apple-ios < %s | FileCheck %s --check-prefix=ARM64

@message = global [80 x i8] c"The LLVM Compiler Infrastructure\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 16
@temp = common global [80 x i8] zeroinitializer, align 16

define void @t1() {
; ARM64-LABEL: t1
; ARM64: adrp x8, _message@PAGE
; ARM64: add x0, x8, _message@PAGEOFF
; ARM64: mov w9, wzr
; ARM64: mov x2, #80
; ARM64: uxtb w1, w9
; ARM64: bl _memset
  call void @llvm.memset.p0i8.i64(i8* getelementptr inbounds ([80 x i8], [80 x i8]* @message, i32 0, i32 0), i8 0, i64 80, i32 16, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1)

define void @t2() {
; ARM64-LABEL: t2
; ARM64: adrp x8, _temp@GOTPAGE
; ARM64: ldr x0, [x8, _temp@GOTPAGEOFF]
; ARM64: adrp x8, _message@PAGE
; ARM64: add x1, x8, _message@PAGEOFF
; ARM64: mov x2, #80
; ARM64: bl _memcpy
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* getelementptr inbounds ([80 x i8], [80 x i8]* @temp, i32 0, i32 0), i8* getelementptr inbounds ([80 x i8], [80 x i8]* @message, i32 0, i32 0), i64 80, i32 16, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1)

define void @t3() {
; ARM64-LABEL: t3
; ARM64: adrp x8, _temp@GOTPAGE
; ARM64: ldr x0, [x8, _temp@GOTPAGEOFF]
; ARM64: adrp x8, _message@PAGE
; ARM64: add x1, x8, _message@PAGEOFF
; ARM64: mov x2, #20
; ARM64: bl _memmove
  call void @llvm.memmove.p0i8.p0i8.i64(i8* getelementptr inbounds ([80 x i8], [80 x i8]* @temp, i32 0, i32 0), i8* getelementptr inbounds ([80 x i8], [80 x i8]* @message, i32 0, i32 0), i64 20, i32 16, i1 false)
  ret void
}

declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1)

define void @t4() {
; ARM64-LABEL: t4
; ARM64: adrp x8, _temp@GOTPAGE
; ARM64: ldr x8, [x8, _temp@GOTPAGEOFF]
; ARM64: adrp x9, _message@PAGE
; ARM64: add x9, x9, _message@PAGEOFF
; ARM64: ldr x10, [x9]
; ARM64: str x10, [x8]
; ARM64: ldr x10, [x9, #8]
; ARM64: str x10, [x8, #8]
; ARM64: ldrb w11, [x9, #16]
; ARM64: strb w11, [x8, #16]
; ARM64: ret
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* getelementptr inbounds ([80 x i8], [80 x i8]* @temp, i32 0, i32 0), i8* getelementptr inbounds ([80 x i8], [80 x i8]* @message, i32 0, i32 0), i64 17, i32 16, i1 false)
  ret void
}

define void @t5() {
; ARM64-LABEL: t5
; ARM64: adrp x8, _temp@GOTPAGE
; ARM64: ldr x8, [x8, _temp@GOTPAGEOFF]
; ARM64: adrp x9, _message@PAGE
; ARM64: add x9, x9, _message@PAGEOFF
; ARM64: ldr x10, [x9]
; ARM64: str x10, [x8]
; ARM64: ldr x10, [x9, #8]
; ARM64: str x10, [x8, #8]
; ARM64: ldrb w11, [x9, #16]
; ARM64: strb w11, [x8, #16]
; ARM64: ret
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* getelementptr inbounds ([80 x i8], [80 x i8]* @temp, i32 0, i32 0), i8* getelementptr inbounds ([80 x i8], [80 x i8]* @message, i32 0, i32 0), i64 17, i32 8, i1 false)
  ret void
}

define void @t6() {
; ARM64-LABEL: t6
; ARM64: adrp x8, _temp@GOTPAGE
; ARM64: ldr x8, [x8, _temp@GOTPAGEOFF]
; ARM64: adrp x9, _message@PAGE
; ARM64: add x9, x9, _message@PAGEOFF
; ARM64: ldr w10, [x9]
; ARM64: str w10, [x8]
; ARM64: ldr w10, [x9, #4]
; ARM64: str w10, [x8, #4]
; ARM64: ldrb w10, [x9, #8]
; ARM64: strb w10, [x8, #8]
; ARM64: ret
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* getelementptr inbounds ([80 x i8], [80 x i8]* @temp, i32 0, i32 0), i8* getelementptr inbounds ([80 x i8], [80 x i8]* @message, i32 0, i32 0), i64 9, i32 4, i1 false)
  ret void
}

define void @t7() {
; ARM64-LABEL: t7
; ARM64: adrp x8, _temp@GOTPAGE
; ARM64: ldr x8, [x8, _temp@GOTPAGEOFF]
; ARM64: adrp x9, _message@PAGE
; ARM64: add x9, x9, _message@PAGEOFF
; ARM64: ldrh w10, [x9]
; ARM64: strh w10, [x8]
; ARM64: ldrh w10, [x9, #2]
; ARM64: strh w10, [x8, #2]
; ARM64: ldrh w10, [x9, #4]
; ARM64: strh w10, [x8, #4]
; ARM64: ldrb w10, [x9, #6]
; ARM64: strb w10, [x8, #6]
; ARM64: ret
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* getelementptr inbounds ([80 x i8], [80 x i8]* @temp, i32 0, i32 0), i8* getelementptr inbounds ([80 x i8], [80 x i8]* @message, i32 0, i32 0), i64 7, i32 2, i1 false)
  ret void
}

define void @t8() {
; ARM64-LABEL: t8
; ARM64: adrp x8, _temp@GOTPAGE
; ARM64: ldr x8, [x8, _temp@GOTPAGEOFF]
; ARM64: adrp x9, _message@PAGE
; ARM64: add x9, x9, _message@PAGEOFF
; ARM64: ldrb w10, [x9]
; ARM64: strb w10, [x8]
; ARM64: ldrb w10, [x9, #1]
; ARM64: strb w10, [x8, #1]
; ARM64: ldrb w10, [x9, #2]
; ARM64: strb w10, [x8, #2]
; ARM64: ldrb w10, [x9, #3]
; ARM64: strb w10, [x8, #3]
; ARM64: ret
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* getelementptr inbounds ([80 x i8], [80 x i8]* @temp, i32 0, i32 0), i8* getelementptr inbounds ([80 x i8], [80 x i8]* @message, i32 0, i32 0), i64 4, i32 1, i1 false)
  ret void
}

define void @test_distant_memcpy(i8* %dst) {
; ARM64-LABEL: test_distant_memcpy:
; ARM64: mov [[ARRAY:x[0-9]+]], sp
; ARM64: mov [[OFFSET:x[0-9]+]], #8000
; ARM64: add x[[ADDR:[0-9]+]], [[ARRAY]], [[OFFSET]]
; ARM64: ldrb [[BYTE:w[0-9]+]], [x[[ADDR]]]
; ARM64: strb [[BYTE]], [x0]
  %array = alloca i8, i32 8192
  %elem = getelementptr i8, i8* %array, i32 8000
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %elem, i64 1, i32 1, i1 false)
  ret void
}

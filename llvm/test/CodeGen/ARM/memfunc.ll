; RUN: llc < %s -mtriple=armv7-apple-ios -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-IOS --check-prefix=CHECK
; RUN: llc < %s -mtriple=thumbv7m-none-macho -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-DARWIN --check-prefix=CHECK
; RUN: llc < %s -mtriple=arm-none-eabi -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI --check-prefix=CHECK
; RUN: llc < %s -mtriple=arm-none-eabihf -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI --check-prefix=CHECK
; RUN: llc < %s -mtriple=arm-none-androideabi -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI --check-prefix=CHECK
; RUN: llc < %s -mtriple=arm-none-gnueabi -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI --check-prefix=CHECK
; RUN: llc < %s -mtriple=arm-none-gnueabihf -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI --check-prefix=CHECK

define void @f1(i8* %dest, i8* %src) {
entry:
  ; CHECK-LABEL: f1

  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: bl memmove
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 500, i32 0, i1 false)

  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 500, i32 0, i1 false)

  ; EABI memset swaps arguments
  ; CHECK-IOS: mov r1, #1
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: movs r1, #1
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: mov r2, #1
  ; CHECK-EABI: bl __aeabi_memset
  ; CHECK-GNUEABI: mov r1, #1
  ; CHECK-GNUEABI: bl memset
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 1, i32 500, i32 0, i1 false)

  ; EABI uses memclr if value set to 0
  ; CHECK-IOS: mov r1, #0
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: movs r1, #0
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: bl __aeabi_memclr
  ; CHECK-GNUEABI: bl memset
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 0, i32 500, i32 0, i1 false)

  ; EABI uses aligned function variants if possible

  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove4
  ; CHECK-GNUEABI: bl memmove
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 500, i32 4, i1 false)

  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy4
  ; CHECK-GNUEABI: bl memcpy
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 500, i32 4, i1 false)

  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: bl __aeabi_memset4
  ; CHECK-GNUEABI: bl memset
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 1, i32 500, i32 4, i1 false)

  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: bl __aeabi_memclr4
  ; CHECK-GNUEABI: bl memset
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 0, i32 500, i32 4, i1 false)

  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove8
  ; CHECK-GNUEABI: bl memmove
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 500, i32 8, i1 false)

  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy8
  ; CHECK-GNUEABI: bl memcpy
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 500, i32 8, i1 false)

  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: bl __aeabi_memset8
  ; CHECK-GNUEABI: bl memset
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 1, i32 500, i32 8, i1 false)

  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: bl __aeabi_memclr8
  ; CHECK-GNUEABI: bl memset
  call void @llvm.memset.p0i8.i32(i8* %dest, i8 0, i32 500, i32 8, i1 false)

  unreachable
}

; Check that alloca arguments to memory intrinsics are automatically aligned if at least 8 bytes in size
define void @f2(i8* %dest, i32 %n) {
entry:
  ; CHECK-LABEL: f2

  ; IOS (ARMv7) should 8-byte align, others should 4-byte align
  ; CHECK-IOS: add r1, sp, #32
  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: add r1, sp, #28
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: add r1, sp, #28
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: add r1, sp, #28
  ; CHECK-GNUEABI: bl memmove
  %arr0 = alloca [9 x i8], align 1
  %0 = bitcast [9 x i8]* %arr0 to i8*
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %0, i32 %n, i32 0, i1 false)

  ; CHECK: add r1, sp, #16
  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  %arr1 = alloca [9 x i8], align 1
  %1 = bitcast [9 x i8]* %arr1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %1, i32 %n, i32 0, i1 false)

  ; CHECK-IOS: mov r0, sp
  ; CHECK-IOS: mov r1, #1
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: add r0, sp, #4
  ; CHECK-DARWIN: movs r1, #1
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: add r0, sp, #4
  ; CHECK-EABI: mov r2, #1
  ; CHECK-EABI: bl __aeabi_memset
  ; CHECK-GNUEABI: add r0, sp, #4
  ; CHECK-GNUEABI: mov r1, #1
  ; CHECK-GNUEABI: bl memset
  %arr2 = alloca [9 x i8], align 1
  %2 = bitcast [9 x i8]* %arr2 to i8*
  call void @llvm.memset.p0i8.i32(i8* %2, i8 1, i32 %n, i32 0, i1 false)

  unreachable
}

; Check that alloca arguments are not aligned if less than 8 bytes in size
define void @f3(i8* %dest, i32 %n) {
entry:
  ; CHECK-LABEL: f3

  ; CHECK: {{add(.w)? r1, sp, #17|sub(.w)? r1, r7, #15}}
  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: bl memmove
  %arr0 = alloca [7 x i8], align 1
  %0 = bitcast [7 x i8]* %arr0 to i8*
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %0, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r1, sp, #10}}
  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  %arr1 = alloca [7 x i8], align 1
  %1 = bitcast [7 x i8]* %arr1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %1, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r0, sp, #3}}
  ; CHECK-IOS: mov r1, #1
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: movs r1, #1
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: mov r2, #1
  ; CHECK-EABI: bl __aeabi_memset
  ; CHECK-GNUEABI: mov r1, #1
  ; CHECK-GNUEABI: bl memset
  %arr2 = alloca [7 x i8], align 1
  %2 = bitcast [7 x i8]* %arr2 to i8*
  call void @llvm.memset.p0i8.i32(i8* %2, i8 1, i32 %n, i32 0, i1 false)

  unreachable
}

; Check that alloca arguments are not aligned if size+offset is less than 8 bytes
define void @f4(i8* %dest, i32 %n) {
entry:
  ; CHECK-LABEL: f4

  ; CHECK: {{add(.w)? r., sp, #23|sub(.w)? r., r7, #17}}
  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: bl memmove
  %arr0 = alloca [9 x i8], align 1
  %0 = getelementptr inbounds [9 x i8], [9 x i8]* %arr0, i32 0, i32 4
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %0, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(10|14)}}
  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  %arr1 = alloca [9 x i8], align 1
  %1 = getelementptr inbounds [9 x i8], [9 x i8]* %arr1, i32 0, i32 4
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %1, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(1|5)}}
  ; CHECK-IOS: mov r1, #1
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: movs r1, #1
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: mov r2, #1
  ; CHECK-EABI: bl __aeabi_memset
  ; CHECK-GNUEABI: mov r1, #1
  ; CHECK-GNUEABI: bl memset
  %arr2 = alloca [9 x i8], align 1
  %2 = getelementptr inbounds [9 x i8], [9 x i8]* %arr2, i32 0, i32 4
  call void @llvm.memset.p0i8.i32(i8* %2, i8 1, i32 %n, i32 0, i1 false)

  unreachable
}

; Check that alloca arguments are not aligned if the offset is not a multiple of 4
define void @f5(i8* %dest, i32 %n) {
entry:
  ; CHECK-LABEL: f5

  ; CHECK: {{add(.w)? r., sp, #27|sub(.w)? r., r7, #21}}
  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: bl memmove
  %arr0 = alloca [13 x i8], align 1
  %0 = getelementptr inbounds [13 x i8], [13 x i8]* %arr0, i32 0, i32 1
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %0, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(10|14)}}
  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  %arr1 = alloca [13 x i8], align 1
  %1 = getelementptr inbounds [13 x i8], [13 x i8]* %arr1, i32 0, i32 1
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %1, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(1|5)}}
  ; CHECK-IOS: mov r1, #1
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: movs r1, #1
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: mov r2, #1
  ; CHECK-EABI: bl __aeabi_memset
  ; CHECK-GNUEABI: mov r1, #1
  ; CHECK-GNUEABI: bl memset
  %arr2 = alloca [13 x i8], align 1
  %2 = getelementptr inbounds [13 x i8], [13 x i8]* %arr2, i32 0, i32 1
  call void @llvm.memset.p0i8.i32(i8* %2, i8 1, i32 %n, i32 0, i1 false)

  unreachable
}

; Check that alloca arguments are not aligned if the offset is unknown
define void @f6(i8* %dest, i32 %n, i32 %i) {
entry:
  ; CHECK-LABEL: f6

  ; CHECK: {{add(.w)? r., sp, #27|sub(.w)? r., r7, #25}}
  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: bl memmove
  %arr0 = alloca [13 x i8], align 1
  %0 = getelementptr inbounds [13 x i8], [13 x i8]* %arr0, i32 0, i32 %i
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %0, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(10|14)}}
  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  %arr1 = alloca [13 x i8], align 1
  %1 = getelementptr inbounds [13 x i8], [13 x i8]* %arr1, i32 0, i32 %i
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %1, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(1|5)}}
  ; CHECK-IOS: mov r1, #1
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: movs r1, #1
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: mov r2, #1
  ; CHECK-EABI: bl __aeabi_memset
  ; CHECK-GNUEABI: mov r1, #1
  ; CHECK-GNUEABI: bl memset
  %arr2 = alloca [13 x i8], align 1
  %2 = getelementptr inbounds [13 x i8], [13 x i8]* %arr2, i32 0, i32 %i
  call void @llvm.memset.p0i8.i32(i8* %2, i8 1, i32 %n, i32 0, i1 false)

  unreachable
}

; Check that alloca arguments are not aligned if the GEP is not inbounds
define void @f7(i8* %dest, i32 %n) {
entry:
  ; CHECK-LABEL: f7

  ; CHECK: {{add(.w)? r., sp, #27|sub(.w)? r., r7, #21}}
  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: bl memmove
  %arr0 = alloca [13 x i8], align 1
  %0 = getelementptr [13 x i8], [13 x i8]* %arr0, i32 0, i32 4
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %0, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(10|14)}}
  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  %arr1 = alloca [13 x i8], align 1
  %1 = getelementptr [13 x i8], [13 x i8]* %arr1, i32 0, i32 4
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %1, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(1|5)}}
  ; CHECK-IOS: mov r1, #1
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: movs r1, #1
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: mov r2, #1
  ; CHECK-EABI: bl __aeabi_memset
  ; CHECK-GNUEABI: mov r1, #1
  ; CHECK-GNUEABI: bl memset
  %arr2 = alloca [13 x i8], align 1
  %2 = getelementptr [13 x i8], [13 x i8]* %arr2, i32 0, i32 4
  call void @llvm.memset.p0i8.i32(i8* %2, i8 1, i32 %n, i32 0, i1 false)

  unreachable
}

; Check that alloca arguments are not aligned when the offset is past the end of the allocation
define void @f8(i8* %dest, i32 %n) {
entry:
  ; CHECK-LABEL: f8

  ; CHECK: {{add(.w)? r., sp, #27|sub(.w)? r., r7, #21}}
  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: bl memmove
  %arr0 = alloca [13 x i8], align 1
  %0 = getelementptr inbounds [13 x i8], [13 x i8]* %arr0, i32 0, i32 16
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %dest, i8* %0, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(10|14)}}
  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  %arr1 = alloca [13 x i8], align 1
  %1 = getelementptr inbounds [13 x i8], [13 x i8]* %arr1, i32 0, i32 16
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %1, i32 %n, i32 0, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(1|5)}}
  ; CHECK-IOS: mov r1, #1
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: movs r1, #1
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: mov r2, #1
  ; CHECK-EABI: bl __aeabi_memset
  ; CHECK-GNUEABI: mov r1, #1
  ; CHECK-GNUEABI: bl memset
  %arr2 = alloca [13 x i8], align 1
  %2 = getelementptr inbounds [13 x i8], [13 x i8]* %arr2, i32 0, i32 16
  call void @llvm.memset.p0i8.i32(i8* %2, i8 1, i32 %n, i32 0, i1 false)

  unreachable
}

; Check that global variables are aligned if they are large enough, but only if
; they are defined in this object and don't have an explicit section.
@arr1 = global [7 x i8] c"\01\02\03\04\05\06\07", align 1
@arr2 = global [8 x i8] c"\01\02\03\04\05\06\07\08", align 1
@arr3 = global [7 x i8] c"\01\02\03\04\05\06\07", section "foo,bar", align 1
@arr4 = global [8 x i8] c"\01\02\03\04\05\06\07\08", section "foo,bar", align 1
@arr5 = weak global [7 x i8] c"\01\02\03\04\05\06\07", align 1
@arr6 = weak_odr global [7 x i8] c"\01\02\03\04\05\06\07", align 1
@arr7 = external global [7 x i8], align 1
define void @f9(i8* %dest, i32 %n) {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @arr1, i32 0, i32 0), i32 %n, i32 1, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @arr2, i32 0, i32 0), i32 %n, i32 1, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @arr3, i32 0, i32 0), i32 %n, i32 1, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @arr4, i32 0, i32 0), i32 %n, i32 1, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @arr5, i32 0, i32 0), i32 %n, i32 1, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @arr6, i32 0, i32 0), i32 %n, i32 1, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @arr7, i32 0, i32 0), i32 %n, i32 1, i1 false)

  unreachable
}

; CHECK: {{\.data|\.section.+data}}
; CHECK-NOT: .p2align
; CHECK: arr1:
; CHECK-IOS: .p2align 3
; CHECK-DARWIN: .p2align 2
; CHECK-EABI-NOT: .p2align
; CHECK-GNUEABI-NOT: .p2align
; CHECK: arr2:
; CHECK: {{\.section.+foo,bar}}
; CHECK-NOT: .p2align
; CHECK: arr3:
; CHECK-NOT: .p2align
; CHECK: arr4:
; CHECK: {{\.data|\.section.+data}}
; CHECK-NOT: .p2align
; CHECK: arr5:
; CHECK-NOT: .p2align
; CHECK: arr6:
; CHECK-NOT: arr7:

declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind

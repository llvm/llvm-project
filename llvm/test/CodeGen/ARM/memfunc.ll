; RUN: llc < %s -mtriple=armv7-apple-ios -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-IOS --check-prefix=CHECK
; RUN: llc < %s -mtriple=thumbv7m-none-macho -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-DARWIN --check-prefix=CHECK
; RUN: llc < %s -mtriple=arm-none-eabi -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI --check-prefix=CHECK
; RUN: llc < %s -mtriple=arm-none-eabihf -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI --check-prefix=CHECK
; RUN: llc < %s -mtriple=arm-none-androideabi -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI --check-prefix=CHECK
; RUN: llc < %s -mtriple=arm-none-gnueabi -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI --check-prefix=CHECK
; RUN: llc < %s -mtriple=arm-none-gnueabihf -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI --check-prefix=CHECK
; RUN: llc < %s -mtriple=arm-none-musleabi -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI --check-prefix=CHECK
; RUN: llc < %s -mtriple=arm-none-musleabihf -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI --check-prefix=CHECK

define void @f1(ptr %dest, ptr %src) "frame-pointer"="all" {
entry:
  ; CHECK-LABEL: f1

  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: bl memmove
  call void @llvm.memmove.p0.p0.i32(ptr %dest, ptr %src, i32 500, i1 false)

  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %src, i32 500, i1 false)

  ; EABI memset swaps arguments
  ; CHECK-IOS: mov r1, #1
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: movs r1, #1
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: mov r2, #1
  ; CHECK-EABI: bl __aeabi_memset
  ; CHECK-GNUEABI: mov r1, #1
  ; CHECK-GNUEABI: bl memset
  call void @llvm.memset.p0.i32(ptr %dest, i8 1, i32 500, i1 false)

  ; EABI uses memclr if value set to 0
  ; CHECK-IOS: mov r1, #0
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: movs r1, #0
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: bl __aeabi_memclr
  ; CHECK-GNUEABI: bl memset
  call void @llvm.memset.p0.i32(ptr %dest, i8 0, i32 500, i1 false)

  ; EABI uses aligned function variants if possible

  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove4
  ; CHECK-GNUEABI: bl memmove
  call void @llvm.memmove.p0.p0.i32(ptr align 4 %dest, ptr align 4 %src, i32 500, i1 false)

  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy4
  ; CHECK-GNUEABI: bl memcpy
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %dest, ptr align 4 %src, i32 500, i1 false)

  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: bl __aeabi_memset4
  ; CHECK-GNUEABI: bl memset
  call void @llvm.memset.p0.i32(ptr align 4 %dest, i8 1, i32 500, i1 false)

  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: bl __aeabi_memclr4
  ; CHECK-GNUEABI: bl memset
  call void @llvm.memset.p0.i32(ptr align 4 %dest, i8 0, i32 500, i1 false)

  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove8
  ; CHECK-GNUEABI: bl memmove
  call void @llvm.memmove.p0.p0.i32(ptr align 8 %dest, ptr align 8 %src, i32 500, i1 false)

  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy8
  ; CHECK-GNUEABI: bl memcpy
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %dest, ptr align 8 %src, i32 500, i1 false)

  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: bl __aeabi_memset8
  ; CHECK-GNUEABI: bl memset
  call void @llvm.memset.p0.i32(ptr align 8 %dest, i8 1, i32 500, i1 false)

  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: bl __aeabi_memclr8
  ; CHECK-GNUEABI: bl memset
  call void @llvm.memset.p0.i32(ptr align 8 %dest, i8 0, i32 500, i1 false)

  ret void
}

; Check that alloca arguments to memory intrinsics are automatically aligned if at least 8 bytes in size
define void @f2(ptr %dest, i32 %n) "frame-pointer"="all" {
entry:
  ; CHECK-LABEL: f2

  ; IOS (ARMv7) should 8-byte align, others should 4-byte align
  ; CHECK-IOS: add r1, sp, #32
  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: add r1, sp, #28
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: {{add r1, sp, #28|sub r1, r(7|11), #20}}
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: {{add r1, sp, #28|sub r1, r(7|11), #20}}
  ; CHECK-GNUEABI: bl memmove
  %arr0 = alloca [9 x i8], align 1
  call void @llvm.memmove.p0.p0.i32(ptr %dest, ptr %arr0, i32 %n, i1 false)

  ; CHECK: add r1, sp, #16
  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  %arr1 = alloca [9 x i8], align 1
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %arr1, i32 %n, i1 false)

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
  call void @llvm.memset.p0.i32(ptr %arr2, i8 1, i32 %n, i1 false)

  ret void
}

; Check that alloca arguments are not aligned if less than 8 bytes in size
define void @f3(ptr %dest, i32 %n) "frame-pointer"="all" {
entry:
  ; CHECK-LABEL: f3

  ; CHECK: {{add(.w)? r1, sp, #17|sub(.w)? r1, r(7|11), #15}}
  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: bl memmove
  %arr0 = alloca [7 x i8], align 1
  call void @llvm.memmove.p0.p0.i32(ptr %dest, ptr %arr0, i32 %n, i1 false)

  ; CHECK: {{add(.w)? r1, sp, #10|sub(.w)? r1, r(7|11), #22}}
  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  %arr1 = alloca [7 x i8], align 1
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %arr1, i32 %n, i1 false)

  ; CHECK: {{add(.w)? r0, sp, #3|sub(.w)? r0, r(7|11), #29}}
  ; CHECK-IOS: mov r1, #1
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: movs r1, #1
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: mov r2, #1
  ; CHECK-EABI: bl __aeabi_memset
  ; CHECK-GNUEABI: mov r1, #1
  ; CHECK-GNUEABI: bl memset
  %arr2 = alloca [7 x i8], align 1
  call void @llvm.memset.p0.i32(ptr %arr2, i8 1, i32 %n, i1 false)

  ret void
}

; Check that alloca arguments are not aligned if size+offset is less than 8 bytes
define void @f4(ptr %dest, i32 %n) "frame-pointer"="all" {
entry:
  ; CHECK-LABEL: f4

  ; CHECK: {{add(.w)? r., sp, #23|sub(.w)? r., r(7|11), #17}}
  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: bl memmove
  %arr0 = alloca [9 x i8], align 1
  %0 = getelementptr inbounds [9 x i8], ptr %arr0, i32 0, i32 4
  call void @llvm.memmove.p0.p0.i32(ptr %dest, ptr %0, i32 %n, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(10|14)|sub(.w) r., r(7|11), #26}}
  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  %arr1 = alloca [9 x i8], align 1
  %1 = getelementptr inbounds [9 x i8], ptr %arr1, i32 0, i32 4
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %1, i32 %n, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(1|5)|sub(.w) r., r(7|11), #35}}
  ; CHECK-IOS: mov r1, #1
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: movs r1, #1
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: mov r2, #1
  ; CHECK-EABI: bl __aeabi_memset
  ; CHECK-GNUEABI: mov r1, #1
  ; CHECK-GNUEABI: bl memset
  %arr2 = alloca [9 x i8], align 1
  %2 = getelementptr inbounds [9 x i8], ptr %arr2, i32 0, i32 4
  call void @llvm.memset.p0.i32(ptr %2, i8 1, i32 %n, i1 false)

  ret void
}

; Check that alloca arguments are not aligned if the offset is not a multiple of 4
define void @f5(ptr %dest, i32 %n) "frame-pointer"="all" {
entry:
  ; CHECK-LABEL: f5

  ; CHECK: {{add(.w)? r., sp, #27|sub(.w)? r., r(7|11), #21}}
  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: bl memmove
  %arr0 = alloca [13 x i8], align 1
  %0 = getelementptr inbounds [13 x i8], ptr %arr0, i32 0, i32 1
  call void @llvm.memmove.p0.p0.i32(ptr %dest, ptr %0, i32 %n, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(10|14)|sub(.w)? r., r(7|11), #34}}
  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  %arr1 = alloca [13 x i8], align 1
  %1 = getelementptr inbounds [13 x i8], ptr %arr1, i32 0, i32 1
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %1, i32 %n, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(1|5)|sub(.w)? r., r(7|11), #47}}
  ; CHECK-IOS: mov r1, #1
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: movs r1, #1
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: mov r2, #1
  ; CHECK-EABI: bl __aeabi_memset
  ; CHECK-GNUEABI: mov r1, #1
  ; CHECK-GNUEABI: bl memset
  %arr2 = alloca [13 x i8], align 1
  %2 = getelementptr inbounds [13 x i8], ptr %arr2, i32 0, i32 1
  call void @llvm.memset.p0.i32(ptr %2, i8 1, i32 %n, i1 false)

  ret void
}

; Check that alloca arguments are not aligned if the offset is unknown
define void @f6(ptr %dest, i32 %n, i32 %i) "frame-pointer"="all" {
entry:
  ; CHECK-LABEL: f6

  ; CHECK: {{add(.w)? r., sp, #27|sub(.w)? r., r(7|11), #(25|29)}}
  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: bl memmove
  %arr0 = alloca [13 x i8], align 1
  %0 = getelementptr inbounds [13 x i8], ptr %arr0, i32 0, i32 %i
  call void @llvm.memmove.p0.p0.i32(ptr %dest, ptr %0, i32 %n, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(10|14)|sub(.w)? r., r(7|11), #42}}
  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  %arr1 = alloca [13 x i8], align 1
  %1 = getelementptr inbounds [13 x i8], ptr %arr1, i32 0, i32 %i
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %1, i32 %n, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(1|5)|sub(.w)? r., r(7|11), #55}}
  ; CHECK-IOS: mov r1, #1
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: movs r1, #1
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: mov r2, #1
  ; CHECK-EABI: bl __aeabi_memset
  ; CHECK-GNUEABI: mov r1, #1
  ; CHECK-GNUEABI: bl memset
  %arr2 = alloca [13 x i8], align 1
  %2 = getelementptr inbounds [13 x i8], ptr %arr2, i32 0, i32 %i
  call void @llvm.memset.p0.i32(ptr %2, i8 1, i32 %n, i1 false)

  ret void
}

; Check that alloca arguments are not aligned if the GEP is not inbounds
define void @f7(ptr %dest, i32 %n) "frame-pointer"="all" {
entry:
  ; CHECK-LABEL: f7

  ; CHECK: {{add(.w)? r., sp, #27|sub(.w)? r., r(7|11), #21}}
  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: bl memmove
  %arr0 = alloca [13 x i8], align 1
  %0 = getelementptr [13 x i8], ptr %arr0, i32 0, i32 4
  call void @llvm.memmove.p0.p0.i32(ptr %dest, ptr %0, i32 %n, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(10|14)|sub(.w)? r., r(7|11), #34}}
  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  %arr1 = alloca [13 x i8], align 1
  %1 = getelementptr [13 x i8], ptr %arr1, i32 0, i32 4
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %1, i32 %n, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(1|5)|sub(.w)? r., r(7|11), #47}}
  ; CHECK-IOS: mov r1, #1
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: movs r1, #1
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: mov r2, #1
  ; CHECK-EABI: bl __aeabi_memset
  ; CHECK-GNUEABI: mov r1, #1
  ; CHECK-GNUEABI: bl memset
  %arr2 = alloca [13 x i8], align 1
  %2 = getelementptr [13 x i8], ptr %arr2, i32 0, i32 4
  call void @llvm.memset.p0.i32(ptr %2, i8 1, i32 %n, i1 false)

  ret void
}

; Check that alloca arguments are not aligned when the offset is past the end of the allocation
define void @f8(ptr %dest, i32 %n) "frame-pointer"="all" {
entry:
  ; CHECK-LABEL: f8

  ; CHECK: {{add(.w)? r., sp, #27|sub(.w)? r., r(7|11), #21}}
  ; CHECK-IOS: bl _memmove
  ; CHECK-DARWIN: bl _memmove
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: bl memmove
  %arr0 = alloca [13 x i8], align 1
  %0 = getelementptr inbounds [13 x i8], ptr %arr0, i32 0, i32 16
  call void @llvm.memmove.p0.p0.i32(ptr %dest, ptr %0, i32 %n, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(10|14)|sub(.w)? r., r(7|11), #34}}
  ; CHECK-IOS: bl _memcpy
  ; CHECK-DARWIN: bl _memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  %arr1 = alloca [13 x i8], align 1
  %1 = getelementptr inbounds [13 x i8], ptr %arr1, i32 0, i32 16
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %1, i32 %n, i1 false)

  ; CHECK: {{add(.w)? r., sp, #(1|5)|sub(.w)? r., r(7|11), #47}}
  ; CHECK-IOS: mov r1, #1
  ; CHECK-IOS: bl _memset
  ; CHECK-DARWIN: movs r1, #1
  ; CHECK-DARWIN: bl _memset
  ; CHECK-EABI: mov r2, #1
  ; CHECK-EABI: bl __aeabi_memset
  ; CHECK-GNUEABI: mov r1, #1
  ; CHECK-GNUEABI: bl memset
  %arr2 = alloca [13 x i8], align 1
  %2 = getelementptr inbounds [13 x i8], ptr %arr2, i32 0, i32 16
  call void @llvm.memset.p0.i32(ptr %2, i8 1, i32 %n, i1 false)

  ret void
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
@arr8 = internal global [128 x i8] undef
@arr9 = weak_odr global [128 x i8] undef
@arr10 = dso_local global [8 x i8] c"\01\02\03\04\05\06\07\08", align 1
define void @f9(ptr %dest, i32 %n) "frame-pointer"="all" {
entry:
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr @arr1, i32 %n, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr @arr2, i32 %n, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr @arr3, i32 %n, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr @arr4, i32 %n, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr @arr5, i32 %n, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr @arr6, i32 %n, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr @arr7, i32 %n, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr @arr8, i32 %n, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr @arr9, i32 %n, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr @arr10, i32 %n, i1 false)
  ret void
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
; CHECK-IOS: arr8,128,4
; CHECK-DARWIN: arr8,128,4
; CHECK-EABI: arr8,128,16
; CHECK-GNUEABI: arr8,128,16
; CHECK: .p2align 4
; CHECK: arr9:
; CHECK-IOS: .p2align 3
; CHECK-DARWIN: .p2align 2
; CHECK-EABI: .p2align 2
; CHECK-GNUEABI: .p2align 2
; CHECK: arr10:

; CHECK-NOT: arr7:

declare void @llvm.memmove.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
declare void @llvm.memset.p0.i32(ptr nocapture, i8, i32, i1) nounwind

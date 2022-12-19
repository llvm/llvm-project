; RUN: llc < %s -mtriple=arm-none-eabi -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=arm-none-eabihf -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=arm-none-androideabi -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=arm-none-gnueabi -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI
; RUN: llc < %s -mtriple=arm-none-gnueabihf -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI
; RUN: llc < %s -mtriple=arm-none-musleabi -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI
; RUN: llc < %s -mtriple=arm-none-musleabihf -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI
; RUN: llc < %s -mtriple=arm-none-eabi -meabi=gnu -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI
; RUN: llc < %s -mtriple=arm-none-eabihf -meabi=gnu -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI
; RUN: llc < %s -mtriple=arm-none-androideabi -meabi=gnu -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI
; RUN: llc < %s -mtriple=arm-none-gnueabi -meabi=gnu -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI
; RUN: llc < %s -mtriple=arm-none-gnueabihf -meabi=gnu -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI
; RUN: llc < %s -mtriple=arm-none-musleabi -meabi=gnu -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI
; RUN: llc < %s -mtriple=arm-none-musleabihf -meabi=gnu -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-GNUEABI
; RUN: llc < %s -mtriple=arm-none-eabi -meabi=4 -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=arm-none-eabihf -meabi=4 -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=arm-none-androideabi -meabi=4 -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=arm-none-gnueabi -meabi=4 -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=arm-none-gnueabihf -meabi=4 -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=arm-none-musleabi -meabi=4 -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=arm-none-musleabihf -meabi=4 -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=arm-none-eabi -meabi=5 -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=arm-none-eabihf -meabi=5 -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=arm-none-androideabi -meabi=5 -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=arm-none-gnueabi -meabi=5 -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=arm-none-gnueabihf -meabi=5 -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=arm-none-musleabi -meabi=5 -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI
; RUN: llc < %s -mtriple=arm-none-musleabihf -meabi=5 -disable-post-ra -o - | FileCheck %s --check-prefix=CHECK-EABI

%struct.my_s = type { [18 x i32] }

define void @foo(ptr %t) {
  ; CHECK-LABEL: foo

  %1 = alloca ptr, align 4
  store ptr %t, ptr %1, align 4
  %2 = load ptr, ptr %1, align 4
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %2, ptr align 4 inttoptr (i32 1 to ptr), i32 72, i1 false)
  ret void
}

define void @f1(ptr %dest, ptr %src) {
entry:
  ; CHECK-LABEL: f1

  ; memmove
  ; CHECK-EABI: bl __aeabi_memmove
  ; CHECK-GNUEABI: bl memmove
  call void @llvm.memmove.p0.p0.i32(ptr %dest, ptr %src, i32 500, i1 false)

  ; memcpy
  ; CHECK-EABI: bl __aeabi_memcpy
  ; CHECK-GNUEABI: bl memcpy
  call void @llvm.memcpy.p0.p0.i32(ptr %dest, ptr %src, i32 500, i1 false)

  ; memset
  ; CHECK-EABI: mov r2, #1
  ; CHECK-EABI: bl __aeabi_memset
  ; CHECK-GNUEABI: mov r1, #1
  ; CHECK-GNUEABI: bl memset
  call void @llvm.memset.p0.i32(ptr %dest, i8 1, i32 500, i1 false)
  ret void
}

declare void @llvm.memmove.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
declare void @llvm.memset.p0.i32(ptr nocapture, i8, i32, i1) nounwind

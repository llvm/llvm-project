; RUN: llc -mtriple=i686-windows -mattr=+sse2 < %s | FileCheck %s

target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"
target triple = "i686-pc-windows-msvc"

; There is a conflict between lowering the X86 memory intrinsics and the "base"
; register used to address stack locals.  See X86RegisterInfo::hasBaseRegister
; for when this is necessary. Typically, we chose ESI for the base register,
; which all of the X86 string instructions use.

declare void @escape_vla_and_icmp(i8*, i1 zeroext)
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i32, i1)
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1)

define i32 @memcpy_novla_vector(<4 x i32>* %vp0, i8* %a, i8* %b, i32 %n, i1 zeroext %cond) {
  %foo = alloca <4 x i32>, align 16
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %b, i32 128, i32 4, i1 false)
  br i1 %cond, label %spill_vectors, label %no_vectors

no_vectors:
  ret i32 0

spill_vectors:
  %vp1 = getelementptr <4 x i32>, <4 x i32>* %vp0, i32 1
  %v0 = load <4 x i32>, <4 x i32>* %vp0
  %v1 = load <4 x i32>, <4 x i32>* %vp1
  %vicmp = icmp slt <4 x i32> %v0, %v1
  %icmp = extractelement <4 x i1> %vicmp, i32 0
  call void @escape_vla_and_icmp(i8* null, i1 zeroext %icmp)
  %r = extractelement <4 x i32> %v0, i32 0
  ret i32 %r
}

; CHECK-LABEL: _memcpy_novla_vector:
; CHECK: andl $-16, %esp
; CHECK-DAG: movl $32, %ecx
; CHECK-DAG: movl {{.*}}, %esi
; CHECK-DAG: movl {{.*}}, %edi
; CHECK: rep;movsl

define i32 @memcpy_vla_vector(<4 x i32>* %vp0, i8* %a, i8* %b, i32 %n, i1 zeroext %cond) {
  %foo = alloca <4 x i32>, align 16
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %b, i32 128, i32 4, i1 false)
  br i1 %cond, label %spill_vectors, label %no_vectors

no_vectors:
  ret i32 0

spill_vectors:
  %vp1 = getelementptr <4 x i32>, <4 x i32>* %vp0, i32 1
  %v0 = load <4 x i32>, <4 x i32>* %vp0
  %v1 = load <4 x i32>, <4 x i32>* %vp1
  %vicmp = icmp slt <4 x i32> %v0, %v1
  %icmp = extractelement <4 x i1> %vicmp, i32 0
  %vla = alloca i8, i32 %n
  call void @escape_vla_and_icmp(i8* %vla, i1 zeroext %icmp)
  %r = extractelement <4 x i32> %v0, i32 0
  ret i32 %r
}

; CHECK-LABEL: _memcpy_vla_vector:
; CHECK: andl $-16, %esp
; CHECK: movl %esp, %esi
; CHECK: pushl $128
; CHECK: calll _memcpy
; CHECK: calll __chkstk

; stosd doesn't clobber esi, so we can use it.

define i32 @memset_vla_vector(<4 x i32>* %vp0, i8* %a, i32 %n, i1 zeroext %cond) {
  %foo = alloca <4 x i32>, align 16
  call void @llvm.memset.p0i8.i32(i8* %a, i8 42, i32 128, i32 4, i1 false)
  br i1 %cond, label %spill_vectors, label %no_vectors

no_vectors:
  ret i32 0

spill_vectors:
  %vp1 = getelementptr <4 x i32>, <4 x i32>* %vp0, i32 1
  %v0 = load <4 x i32>, <4 x i32>* %vp0
  %v1 = load <4 x i32>, <4 x i32>* %vp1
  %vicmp = icmp slt <4 x i32> %v0, %v1
  %icmp = extractelement <4 x i1> %vicmp, i32 0
  %vla = alloca i8, i32 %n
  call void @escape_vla_and_icmp(i8* %vla, i1 zeroext %icmp)
  %r = extractelement <4 x i32> %v0, i32 0
  ret i32 %r
}

; CHECK-LABEL: _memset_vla_vector:
; CHECK: andl $-16, %esp
; CHECK: movl %esp, %esi
; CHECK-DAG: movl $707406378, %eax        # imm = 0x2A2A2A2A
; CHECK-DAG: movl $32, %ecx
; CHECK-DAG: movl {{.*}}, %edi
; CHECK-NOT: movl {{.*}}, %esi
; CHECK: rep;stosl

; Add a test for memcmp if we ever add a special lowering for it.

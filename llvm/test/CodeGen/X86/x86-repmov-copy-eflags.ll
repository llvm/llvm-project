; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

%struct.T = type { i64, [3 x i32] }

; Function Attrs: nounwind optsize
define void @f(i8* %p, i8* %q, i32* inalloca nocapture %unused) #0 {
entry:
  %g = alloca %struct.T, align 8
  %r = alloca i32, align 8
  store i32 0, i32* %r, align 4
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %p, i8* align 8 %q, i32 24, i1 false)
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %load = load i32, i32* %r, align 4
  %dec = add nsw i32 %load, -1
  store i32 %dec, i32* %r, align 4
  call void @g(%struct.T* %g)
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1) #1

declare void @g(%struct.T*)

; CHECK-LABEL: _f:
; CHECK:     pushl %ebp
; CHECK:     movl %esp, %ebp
; CHECK:     andl $-8, %esp
; CHECK-NOT: movl %esp, %esi
; CHECK:     rep;movsl
; CHECK:     leal 8(%esp), %esi

; CHECK:     decl     (%esp)
; CHECK:     setne    %[[NE_REG:.*]]
; CHECK:     pushl     %esi
; CHECK:     calll     _g
; CHECK:     addl     $4, %esp
; CHECK:     testb    %[[NE_REG]], %[[NE_REG]]
; CHECK:     jne

attributes #0 = { nounwind optsize }
attributes #1 = { argmemonly nounwind }

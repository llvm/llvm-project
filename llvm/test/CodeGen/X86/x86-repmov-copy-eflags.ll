; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

%struct.T = type { i64, [3 x i32] }

; Function Attrs: nounwind optsize
define void @f(ptr %p, ptr %q, ptr inalloca(i32) nocapture %unused) #0 {
entry:
  %g = alloca %struct.T, align 8
  %r = alloca i32, align 8
  store i32 0, ptr %r, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %p, ptr align 8 %q, i32 24, i1 false)
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %load = load i32, ptr %r, align 4
  %dec = add nsw i32 %load, -1
  store i32 %dec, ptr %r, align 4
  call void @g(ptr %g)
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body
  ret void
}

define void @f_pgso(ptr %p, ptr %q, ptr inalloca(i32) nocapture %unused) !prof !14 {
entry:
  %g = alloca %struct.T, align 8
  %r = alloca i32, align 8
  store i32 0, ptr %r, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %p, ptr align 8 %q, i32 24, i1 false)
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %load = load i32, ptr %r, align 4
  %dec = add nsw i32 %load, -1
  store i32 %dec, ptr %r, align 4
  call void @g(ptr %g)
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture readonly, i32, i1) #1

declare void @g(ptr)

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

; CHECK-LABEL: _f_pgso:
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

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 0}

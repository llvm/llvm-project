; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core2 | FileCheck %s
; rdar://7842028

; Do not delete partially dead copy instructions.
; dead %rdi = MOV64rr killed %rax, implicit-def %edi
; REP_MOVSD implicit dead %ecx, implicit dead %edi, implicit dead %esi, implicit killed %ecx, implicit killed %edi, implicit killed %esi


%struct.F = type { ptr, i32, i32, i8, i32, i32, i32 }
%struct.FC = type { [10 x i8], [32 x i32], ptr, i32 }

define void @t(ptr %this) nounwind {
entry:
; CHECK-LABEL: t:
; CHECK: addq $12, %rsi
  %BitValueArray = alloca [32 x i32], align 4
  %tmp3 = load ptr, ptr %this, align 8
  %tmp4 = getelementptr inbounds %struct.FC, ptr %tmp3, i64 0, i32 1, i64 0
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %BitValueArray, ptr align 4 %tmp4, i64 128, i1 false)
  unreachable
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind

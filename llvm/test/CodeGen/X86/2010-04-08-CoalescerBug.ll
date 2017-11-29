; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core2 | FileCheck %s
; rdar://7842028

; Do not delete partially dead copy instructions.
; %rdi<def,dead> = MOV64rr %rax<kill>, %edi<imp-def>
; REP_MOVSD %ecx<imp-def,dead>, %edi<imp-def,dead>, %esi<imp-def,dead>, %ecx<imp-use,kill>, %edi<imp-use,kill>, %esi<imp-use,kill>


%struct.F = type { %struct.FC*, i32, i32, i8, i32, i32, i32 }
%struct.FC = type { [10 x i8], [32 x i32], %struct.FC*, i32 }

define void @t(%struct.F* %this) nounwind {
entry:
; CHECK-LABEL: t:
; CHECK: addq $12, %rsi
  %BitValueArray = alloca [32 x i32], align 4
  %tmp2 = getelementptr inbounds %struct.F, %struct.F* %this, i64 0, i32 0
  %tmp3 = load %struct.FC*, %struct.FC** %tmp2, align 8
  %tmp4 = getelementptr inbounds %struct.FC, %struct.FC* %tmp3, i64 0, i32 1, i64 0
  %tmp5 = bitcast [32 x i32]* %BitValueArray to i8*
  %tmp6 = bitcast i32* %tmp4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp5, i8* %tmp6, i64 128, i32 4, i1 false)
  unreachable
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

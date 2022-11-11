; RUN: llc -verify-machineinstrs -mcpu=pwr7 -disable-ppc-instr-form-prep < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s -check-prefix=PIP
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define double @foo() #1 {
entry:
  %x = alloca [2048 x float], align 4
  %y = alloca [2048 x float], align 4
  call void @llvm.lifetime.start.p0(i64 8192, ptr %x) #2
  call void @llvm.lifetime.start.p0(i64 8192, ptr %y) #2
  br label %for.body.i

; CHECK-LABEL: @foo
; CHECK: addi [[REG1:[0-9]+]], 1,
; CHECK: addi [[REG2:[0-9]+]], 1,
; CHECK: li [[REG3:[0-9]+]], 0
; CHECK: %for.body.i
; CHECK-DAG: lfsx {{[0-9]+}}, [[REG1]], [[REG3]]
; CHECK-DAG: lfsx {{[0-9]+}}, [[REG2]], [[REG3]]
; CHECK: blr

; PIP-LABEL: @foo
; PIP: addi [[REG1:[0-9]+]], 1,
; PIP: addi [[REG2:[0-9]+]], 1,
; PIP: %for.body.i
; PIP-DAG: lfsu {{[0-9]+}}, 4([[REG1]])
; PIP-DAG: lfsu {{[0-9]+}}, 4([[REG2]])
; PIP: blr

for.body.i:                                       ; preds = %for.body.i.preheader, %for.body.i
  %accumulator.09.i = phi double [ %add.i, %for.body.i ], [ 0.000000e+00, %entry ]
  %i.08.i = phi i64 [ %inc.i, %for.body.i ], [ 0, %entry ]
  %arrayidx.i = getelementptr inbounds [2048 x float], ptr %x, i64 0, i64 %i.08.i
  %v14 = load float, ptr %arrayidx.i, align 4
  %conv.i = fpext float %v14 to double
  %arrayidx1.i = getelementptr inbounds [2048 x float], ptr %y, i64 0, i64 %i.08.i
  %v15 = load float, ptr %arrayidx1.i, align 4
  %conv2.i = fpext float %v15 to double
  %mul.i = fmul double %conv.i, %conv2.i
  %add.i = fadd double %accumulator.09.i, %mul.i
  %inc.i = add nuw nsw i64 %i.08.i, 1
  %exitcond.i = icmp eq i64 %i.08.i, 2047
  br i1 %exitcond.i, label %loop.exit, label %for.body.i

loop.exit:                                        ; preds = %for.body.i
  ret double %accumulator.09.i
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #2

declare void @bar(ptr, ptr)

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #2

attributes #0 = { nounwind readonly }
attributes #1 = { nounwind }
attributes #2 = { nounwind }



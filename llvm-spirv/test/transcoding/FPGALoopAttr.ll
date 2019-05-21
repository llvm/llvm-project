; RUN: llvm-as < %s > %t.bc
; RUN: llvm-spirv %t.bc -o - -spirv-text | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; ModuleID = 'FPGALoopAttr.cl'
source_filename = "FPGALoopAttr.cl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown-unknown"

; CHECK-SPIRV: Function
; Function Attrs: convergent noinline nounwind optnone
define spir_kernel void @test_ivdep() #0 {
entry:
  %a = alloca [10 x i32], align 4
  %i = alloca i32, align 4
  %i1 = alloca i32, align 4
  %i10 = alloca i32, align 4
  %i19 = alloca i32, align 4
  %i28 = alloca i32, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond
; CHECK-SPIRV: 4 LoopMerge {{[0-9]+}} {{[0-9]+}} 4
; CHECK-SPIRV: 4 BranchConditional {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp ne i32 %0, 10
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [10 x i32], [10 x i32]* %a, i64 0, i64 %idxprom
  store i32 0, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %2 = load i32, i32* %i, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond, !llvm.loop !3

for.end:                                          ; preds = %for.cond
  store i32 0, i32* %i1, align 4
  br label %for.cond2

; CHECK-SPIRV: 5 LoopMerge {{[0-9]+}} {{[0-9]+}} 8 2
; CHECK-SPIRV: 4 BranchConditional {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
for.cond2:                                        ; preds = %for.inc7, %for.end
  %3 = load i32, i32* %i1, align 4
  %cmp3 = icmp ne i32 %3, 10
  br i1 %cmp3, label %for.body4, label %for.end9

for.body4:                                        ; preds = %for.cond2
  %4 = load i32, i32* %i1, align 4
  %idxprom5 = sext i32 %4 to i64
  %arrayidx6 = getelementptr inbounds [10 x i32], [10 x i32]* %a, i64 0, i64 %idxprom5
  store i32 0, i32* %arrayidx6, align 4
  br label %for.inc7

for.inc7:                                         ; preds = %for.body4
  %5 = load i32, i32* %i1, align 4
  %inc8 = add nsw i32 %5, 1
  store i32 %inc8, i32* %i1, align 4
  br label %for.cond2, !llvm.loop !5

for.end9:                                         ; preds = %for.cond2
  store i32 0, i32* %i10, align 4
  br label %for.cond11

; CHECK-SPIRV: 6 LoopMerge {{[0-9]+}} {{[0-9]+}} 2147483648 5889 2
; CHECK-SPIRV: 4 BranchConditional {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
for.cond11:                                       ; preds = %for.inc16, %for.end9
  %6 = load i32, i32* %i10, align 4
  %cmp12 = icmp ne i32 %6, 10
  br i1 %cmp12, label %for.body13, label %for.end18

for.body13:                                       ; preds = %for.cond11
  %7 = load i32, i32* %i10, align 4
  %idxprom14 = sext i32 %7 to i64
  %arrayidx15 = getelementptr inbounds [10 x i32], [10 x i32]* %a, i64 0, i64 %idxprom14
  store i32 0, i32* %arrayidx15, align 4
  br label %for.inc16

for.inc16:                                        ; preds = %for.body13
  %8 = load i32, i32* %i10, align 4
  %inc17 = add nsw i32 %8, 1
  store i32 %inc17, i32* %i10, align 4
  br label %for.cond11, !llvm.loop !7

for.end18:                                        ; preds = %for.cond11
  store i32 0, i32* %i19, align 4
  br label %for.cond20

; CHECK-SPIRV: 6 LoopMerge {{[0-9]+}} {{[0-9]+}} 2147483648 5890 2
; CHECK-SPIRV: 4 BranchConditional {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
for.cond20:                                       ; preds = %for.inc25, %for.end18
  %9 = load i32, i32* %i19, align 4
  %cmp21 = icmp ne i32 %9, 10
  br i1 %cmp21, label %for.body22, label %for.end27

for.body22:                                       ; preds = %for.cond20
  %10 = load i32, i32* %i19, align 4
  %idxprom23 = sext i32 %10 to i64
  %arrayidx24 = getelementptr inbounds [10 x i32], [10 x i32]* %a, i64 0, i64 %idxprom23
  store i32 0, i32* %arrayidx24, align 4
  br label %for.inc25

for.inc25:                                        ; preds = %for.body22
  %11 = load i32, i32* %i19, align 4
  %inc26 = add nsw i32 %11, 1
  store i32 %inc26, i32* %i19, align 4
  br label %for.cond20, !llvm.loop !9

for.end27:                                        ; preds = %for.cond20
  store i32 0, i32* %i28, align 4
  br label %for.cond29

; CHECK-SPIRV: 8 LoopMerge {{[0-9]+}} {{[0-9]+}} 2147483648 5889 2 5890 2
; CHECK-SPIRV: 4 BranchConditional {{[0-9]+}} {{[0-9]+}} {{[0-9]+}}
for.cond29:                                       ; preds = %for.inc34, %for.end27
  %12 = load i32, i32* %i28, align 4
  %cmp30 = icmp ne i32 %12, 10
  br i1 %cmp30, label %for.body31, label %for.end36

for.body31:                                       ; preds = %for.cond29
  %13 = load i32, i32* %i28, align 4
  %idxprom32 = sext i32 %13 to i64
  %arrayidx33 = getelementptr inbounds [10 x i32], [10 x i32]* %a, i64 0, i64 %idxprom32
  store i32 0, i32* %arrayidx33, align 4
  br label %for.inc34

for.inc34:                                        ; preds = %for.body31
  %14 = load i32, i32* %i28, align 4
  %inc35 = add nsw i32 %14, 1
  store i32 %inc35, i32* %i28, align 4
  br label %for.cond29, !llvm.loop !11

for.end36:                                        ; preds = %for.cond29
  ret void
}

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!3 = distinct !{!3, !4}
!4 = !{!"llvm.loop.ivdep.enable"}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.ivdep.safelen", i32 2}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.ii.count", i32 2}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.max_concurrency.count", i32 2}
!11 = distinct !{!11, !8, !10}

; CHECK-LLVM: br i1 %cmp, label %for.body, label %for.end, !llvm.loop ![[MD_A:[0-9]+]]
; CHECK-LLVM: br i1 %cmp{{[0-9]+}}, label %for.body{{[0-9]+}}, label %for.end{{[0-9]+}}, !llvm.loop ![[MD_B:[0-9]+]]
; CHECK-LLVM: br i1 %cmp{{[0-9]+}}, label %for.body{{[0-9]+}}, label %for.end{{[0-9]+}}, !llvm.loop ![[MD_C:[0-9]+]]
; CHECK-LLVM: br i1 %cmp{{[0-9]+}}, label %for.body{{[0-9]+}}, label %for.end{{[0-9]+}}, !llvm.loop ![[MD_D:[0-9]+]]
; CHECK-LLVM: br i1 %cmp{{[0-9]+}}, label %for.body{{[0-9]+}}, label %for.end{{[0-9]+}}, !llvm.loop ![[MD_E:[0-9]+]]

; CHECK-LLVM: ![[MD_A]] = distinct !{![[MD_A]], ![[MD_ivdep_enable:[0-9]+]]}
; CHECK-LLVM: ![[MD_ivdep_enable]] = !{!"llvm.loop.ivdep.enable"}
; CHECK-LLVM: ![[MD_B]] = distinct !{![[MD_B]], ![[MD_ivdep:[0-9]+]]}
; CHECK-LLVM: ![[MD_ivdep]] = !{!"llvm.loop.ivdep.safelen", i32 2}
; CHECK-LLVM: ![[MD_C]] = distinct !{![[MD_C]], ![[MD_ii:[0-9]+]]}
; CHECK-LLVM: ![[MD_ii]] = !{!"llvm.loop.ii.count", i32 2}
; CHECK-LLVM: ![[MD_D]] = distinct !{![[MD_D]], ![[MD_max_concurrency:[0-9]+]]}
; CHECK-LLVM: ![[MD_max_concurrency]] = !{!"llvm.loop.max_concurrency.count", i32 2}
; CHECK-LLVM: ![[MD_E]] = distinct !{![[MD_E]], ![[MD_ii:[0-9]+]], ![[MD_max_concurrency:[0-9]+]]}

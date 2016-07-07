; RUN: opt < %s -O3 -S | FileCheck %s

; Make sure the call to syncthreads is not duplicate here by the LLVM
; optimizations, because it has the noduplicate attribute set.

; CHECK: call void @llvm.nvvm.barrier0
; CHECK-NOT: call void @llvm.nvvm.barrier0

; Function Attrs: nounwind
define void @foo(float* %output) #1 {
entry:
  %output.addr = alloca float*, align 8
  store float* %output, float** %output.addr, align 8
  %0 = load float*, float** %output.addr, align 8
  %arrayidx = getelementptr inbounds float, float* %0, i64 0
  %1 = load float, float* %arrayidx, align 4
  %conv = fpext float %1 to double
  %cmp = fcmp olt double %conv, 1.000000e+01
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %2 = load float*, float** %output.addr, align 8
  %3 = load float, float* %2, align 4
  %conv1 = fpext float %3 to double
  %add = fadd double %conv1, 1.000000e+00
  %conv2 = fptrunc double %add to float
  store float %conv2, float* %2, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %4 = load float*, float** %output.addr, align 8
  %5 = load float, float* %4, align 4
  %conv3 = fpext float %5 to double
  %add4 = fadd double %conv3, 2.000000e+00
  %conv5 = fptrunc double %add4 to float
  store float %conv5, float* %4, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @llvm.nvvm.barrier0()
  %6 = load float*, float** %output.addr, align 8
  %arrayidx6 = getelementptr inbounds float, float* %6, i64 0
  %7 = load float, float* %arrayidx6, align 4
  %conv7 = fpext float %7 to double
  %cmp8 = fcmp olt double %conv7, 1.000000e+01
  br i1 %cmp8, label %if.then9, label %if.else13

if.then9:                                         ; preds = %if.end
  %8 = load float*, float** %output.addr, align 8
  %9 = load float, float* %8, align 4
  %conv10 = fpext float %9 to double
  %add11 = fadd double %conv10, 3.000000e+00
  %conv12 = fptrunc double %add11 to float
  store float %conv12, float* %8, align 4
  br label %if.end17

if.else13:                                        ; preds = %if.end
  %10 = load float*, float** %output.addr, align 8
  %11 = load float, float* %10, align 4
  %conv14 = fpext float %11 to double
  %add15 = fadd double %conv14, 4.000000e+00
  %conv16 = fptrunc double %add15 to float
  store float %conv16, float* %10, align 4
  br label %if.end17

if.end17:                                         ; preds = %if.else13, %if.then9
  ret void
}

; Function Attrs: noduplicate nounwind
declare void @llvm.nvvm.barrier0() #2

!0 = !{void (float*)* @foo, !"kernel", i32 1}
!1 = !{null, !"align", i32 8}

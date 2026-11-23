; RUN: opt < %s -O3 -S | FileCheck %s

; Make sure the call to syncthreads is not duplicate here by the LLVM
; optimizations, because it has the noduplicate attribute set.

; CHECK: call void @llvm.nvvm.barrier.cta.sync.aligned.all
; CHECK-NOT: call void @llvm.nvvm.barrier.cta.sync.aligned.all

; Function Attrs: nounwind
define void @foo(ptr %output) #1 {
entry:
  %output.addr = alloca ptr, align 8
  store ptr %output, ptr %output.addr, align 8
  %0 = load ptr, ptr %output.addr, align 8
  %1 = load float, ptr %0, align 4
  %conv = fpext float %1 to double
  %cmp = fcmp olt double %conv, 1.000000e+01
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %2 = load ptr, ptr %output.addr, align 8
  %3 = load float, ptr %2, align 4
  %conv1 = fpext float %3 to double
  %add = fadd double %conv1, 1.000000e+00
  %conv2 = fptrunc double %add to float
  store float %conv2, ptr %2, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %4 = load ptr, ptr %output.addr, align 8
  %5 = load float, ptr %4, align 4
  %conv3 = fpext float %5 to double
  %add4 = fadd double %conv3, 2.000000e+00
  %conv5 = fptrunc double %add4 to float
  store float %conv5, ptr %4, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %6 = load ptr, ptr %output.addr, align 8
  %7 = load float, ptr %6, align 4
  %conv7 = fpext float %7 to double
  %cmp8 = fcmp olt double %conv7, 1.000000e+01
  br i1 %cmp8, label %if.then9, label %if.else13

if.then9:                                         ; preds = %if.end
  %8 = load ptr, ptr %output.addr, align 8
  %9 = load float, ptr %8, align 4
  %conv10 = fpext float %9 to double
  %add11 = fadd double %conv10, 3.000000e+00
  %conv12 = fptrunc double %add11 to float
  store float %conv12, ptr %8, align 4
  br label %if.end17

if.else13:                                        ; preds = %if.end
  %10 = load ptr, ptr %output.addr, align 8
  %11 = load float, ptr %10, align 4
  %conv14 = fpext float %11 to double
  %add15 = fadd double %conv14, 4.000000e+00
  %conv16 = fptrunc double %add15 to float
  store float %conv16, ptr %10, align 4
  br label %if.end17

if.end17:                                         ; preds = %if.else13, %if.then9
  ret void
}

; Function Attrs: noduplicate nounwind
declare void @llvm.nvvm.barrier0() #2
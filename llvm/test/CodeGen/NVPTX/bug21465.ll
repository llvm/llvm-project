; RUN: opt < %s -nvptx-lower-args -S | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_35 | FileCheck %s --check-prefix PTX
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_35 | %ptxas-verify %}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-unknown-unknown"

%struct.S = type { i32, i32 }

; Function Attrs: nounwind
define ptx_kernel void @_Z11TakesStruct1SPi(ptr byval(%struct.S) nocapture readonly %input, ptr nocapture %output) #0 {
entry:
; CHECK-LABEL: @_Z11TakesStruct1SPi
; PTX-LABEL: .visible .entry _Z11TakesStruct1SPi(
; CHECK: addrspacecast ptr %input to ptr addrspace(101)
  %b = getelementptr inbounds %struct.S, ptr %input, i64 0, i32 1
  %0 = load i32, ptr %b, align 4
; PTX-NOT: ld.param.u32 {{%r[0-9]+}}, [{{%rd[0-9]+}}]
; PTX: ld.param.u32 [[value:%r[0-9]+]], [_Z11TakesStruct1SPi_param_0+4]
  store i32 %0, ptr %output, align 4
; PTX-NEXT: st.global.u32 [{{%rd[0-9]+}}], [[value]]
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

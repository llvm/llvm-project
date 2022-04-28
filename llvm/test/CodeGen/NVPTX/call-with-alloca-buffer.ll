; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; Checks how NVPTX lowers alloca buffers and their passing to functions.
;
; Produced with the following CUDA code:
;  extern "C" __attribute__((device)) void callee(float* f, char* buf);
;
;  extern "C" __attribute__((global)) void kernel_func(float* a) {
;    char buf[4 * sizeof(float)];
;    *(reinterpret_cast<float*>(&buf[0])) = a[0];
;    *(reinterpret_cast<float*>(&buf[1])) = a[1];
;    *(reinterpret_cast<float*>(&buf[2])) = a[2];
;    *(reinterpret_cast<float*>(&buf[3])) = a[3];
;    callee(a, buf);
;  }

; CHECK: .visible .entry kernel_func
define void @kernel_func(float* %a) {
entry:
  %buf = alloca [16 x i8], align 4

; CHECK: .local .align 4 .b8 	__local_depot0[16]
; CHECK: mov.u64 %SPL

; CHECK: ld.param.u64 %rd[[A_REG:[0-9]+]], [kernel_func_param_0]
; CHECK: cvta.to.global.u64 %rd[[A1_REG:[0-9]+]], %rd[[A_REG]]
; CHECK: add.u64 %rd[[SP_REG:[0-9]+]], %SP, 0
; CHECK: ld.global.f32 %f[[A0_REG:[0-9]+]], [%rd[[A1_REG]]]
; CHECK: st.local.f32 [{{%rd[0-9]+}}], %f[[A0_REG]]

  %0 = load float, float* %a, align 4
  %1 = bitcast [16 x i8]* %buf to float*
  store float %0, float* %1, align 4
  %arrayidx2 = getelementptr inbounds float, float* %a, i64 1
  %2 = load float, float* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [16 x i8], [16 x i8]* %buf, i64 0, i64 1
  %3 = bitcast i8* %arrayidx3 to float*
  store float %2, float* %3, align 4
  %arrayidx4 = getelementptr inbounds float, float* %a, i64 2
  %4 = load float, float* %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds [16 x i8], [16 x i8]* %buf, i64 0, i64 2
  %5 = bitcast i8* %arrayidx5 to float*
  store float %4, float* %5, align 4
  %arrayidx6 = getelementptr inbounds float, float* %a, i64 3
  %6 = load float, float* %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds [16 x i8], [16 x i8]* %buf, i64 0, i64 3
  %7 = bitcast i8* %arrayidx7 to float*
  store float %6, float* %7, align 4

; CHECK:        .param .b64 param0;
; CHECK-NEXT:   st.param.b64  [param0+0], %rd[[A_REG]]
; CHECK-NEXT:   .param .b64 param1;
; CHECK-NEXT:   st.param.b64  [param1+0], %rd[[SP_REG]]
; CHECK-NEXT:   call.uni
; CHECK-NEXT:   callee,

  %arraydecay = getelementptr inbounds [16 x i8], [16 x i8]* %buf, i64 0, i64 0
  call void @callee(float* %a, i8* %arraydecay) #2
  ret void
}

declare void @callee(float*, i8*)

!nvvm.annotations = !{!0}

!0 = !{void (float*)* @kernel_func, !"kernel", i32 1}

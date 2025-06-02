; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; Checks how NVPTX lowers alloca buffers and their passing to functions.
;
; Produced with the following CUDA code:
;  extern "C" __attribute__((device)) void callee(ptr f, char* buf);
;
;  extern "C" __attribute__((global)) void kernel_func(ptr a) {
;    char buf[4 * sizeof(float)];
;    *(reinterpret_cast<ptr>(&buf[0])) = a[0];
;    *(reinterpret_cast<ptr>(&buf[1])) = a[1];
;    *(reinterpret_cast<ptr>(&buf[2])) = a[2];
;    *(reinterpret_cast<ptr>(&buf[3])) = a[3];
;    callee(a, buf);
;  }

; CHECK: .visible .entry kernel_func
define ptx_kernel void @kernel_func(ptr %a) {
entry:
  %buf = alloca [16 x i8], align 4

; CHECK: .local .align 4 .b8 	__local_depot0[16]
; CHECK: mov.b64 %SPL

; CHECK: ld.param.b64 %rd[[A_REG:[0-9]+]], [kernel_func_param_0]
; CHECK: cvta.to.global.u64 %rd[[A1_REG:[0-9]+]], %rd[[A_REG]]
; CHECK: add.u64 %rd[[SP_REG:[0-9]+]], %SP, 0
; CHECK: ld.global.b32 %r[[A0_REG:[0-9]+]], [%rd[[A1_REG]]]
; CHECK: st.local.b32 [{{%rd[0-9]+}}], %r[[A0_REG]]

  %0 = load float, ptr %a, align 4
  store float %0, ptr %buf, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %a, i64 1
  %1 = load float, ptr %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [16 x i8], ptr %buf, i64 0, i64 1
  store float %1, ptr %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds float, ptr %a, i64 2
  %2 = load float, ptr %arrayidx4, align 4
  %arrayidx5 = getelementptr inbounds [16 x i8], ptr %buf, i64 0, i64 2
  store float %2, ptr %arrayidx5, align 4
  %arrayidx6 = getelementptr inbounds float, ptr %a, i64 3
  %3 = load float, ptr %arrayidx6, align 4
  %arrayidx7 = getelementptr inbounds [16 x i8], ptr %buf, i64 0, i64 3
  store float %3, ptr %arrayidx7, align 4

; CHECK:        .param .b64 param0;
; CHECK-NEXT:   st.param.b64  [param0], %rd[[A_REG]]
; CHECK-NEXT:   .param .b64 param1;
; CHECK-NEXT:   st.param.b64  [param1], %rd[[SP_REG]]
; CHECK-NEXT:   call.uni
; CHECK-NEXT:   callee,

  call void @callee(ptr %a, ptr %buf) #2
  ret void
}

declare void @callee(ptr, ptr)

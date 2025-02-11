; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 -verify-machineinstrs | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -verify-machineinstrs | FileCheck %s --check-prefix=PTX64
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}

; Ensure we access the local stack properly

; PTX32:        mov.u32          %SPL, __local_depot{{[0-9]+}};
; PTX32:        cvta.local.u32   %SP, %SPL;
; PTX32:        ld.param.u32     %r{{[0-9]+}}, [foo_param_0];
; PTX32:        st.volatile.u32  [%SP], %r{{[0-9]+}};
; PTX64:        mov.u64          %SPL, __local_depot{{[0-9]+}};
; PTX64:        cvta.local.u64   %SP, %SPL;
; PTX64:        ld.param.u32     %r{{[0-9]+}}, [foo_param_0];
; PTX64:        st.volatile.u32  [%SP], %r{{[0-9]+}};
define void @foo(i32 %a) {
  %local = alloca i32, align 4
  store volatile i32 %a, ptr %local
  ret void
}

; PTX32:        mov.u32          %SPL, __local_depot{{[0-9]+}};
; PTX32:        cvta.local.u32   %SP, %SPL;
; PTX32:        ld.param.u32     %r{{[0-9]+}}, [foo2_param_0];
; PTX32:        add.u32          %r[[SP_REG:[0-9]+]], %SPL, 0;
; PTX32:        st.local.u32  [%r[[SP_REG]]], %r{{[0-9]+}};
; PTX64:        mov.u64          %SPL, __local_depot{{[0-9]+}};
; PTX64:        cvta.local.u64   %SP, %SPL;
; PTX64:        ld.param.u32     %r{{[0-9]+}}, [foo2_param_0];
; PTX64:        add.u64          %rd[[SP_REG:[0-9]+]], %SPL, 0;
; PTX64:        st.local.u32  [%rd[[SP_REG]]], %r{{[0-9]+}};
define ptx_kernel void @foo2(i32 %a) {
  %local = alloca i32, align 4
  store i32 %a, ptr %local
  call void @bar(ptr %local)
  ret void
}

declare void @bar(ptr %a)


; PTX32:        mov.u32          %SPL, __local_depot{{[0-9]+}};
; PTX32-NOT:    cvta.local.u32   %SP, %SPL;
; PTX32:        ld.param.u32     %r{{[0-9]+}}, [foo3_param_0];
; PTX32:        add.u32          %r{{[0-9]+}}, %SPL, 0;
; PTX32:        st.local.u32  [%r{{[0-9]+}}], %r{{[0-9]+}};
; PTX64:        mov.u64          %SPL, __local_depot{{[0-9]+}};
; PTX64-NOT:    cvta.local.u64   %SP, %SPL;
; PTX64:        ld.param.u32     %r{{[0-9]+}}, [foo3_param_0];
; PTX64:        add.u64          %rd{{[0-9]+}}, %SPL, 0;
; PTX64:        st.local.u32  [%rd{{[0-9]+}}], %r{{[0-9]+}};
define void @foo3(i32 %a) {
  %local = alloca [3 x i32], align 4
  %1 = getelementptr inbounds i32, ptr %local, i32 %a
  store i32 %a, ptr %1
  ret void
}

; PTX32:        cvta.local.u32   %SP, %SPL;
; PTX32:        add.u32          {{%r[0-9]+}}, %SP, 0;
; PTX32:        add.u32          {{%r[0-9]+}}, %SPL, 0;
; PTX32:        add.u32          {{%r[0-9]+}}, %SP, 4;
; PTX32:        add.u32          {{%r[0-9]+}}, %SPL, 4;
; PTX32:        st.local.u32     [{{%r[0-9]+}}], {{%r[0-9]+}}
; PTX32:        st.local.u32     [{{%r[0-9]+}}], {{%r[0-9]+}}
; PTX64:        cvta.local.u64   %SP, %SPL;
; PTX64:        add.u64          {{%rd[0-9]+}}, %SP, 0;
; PTX64:        add.u64          {{%rd[0-9]+}}, %SPL, 0;
; PTX64:        add.u64          {{%rd[0-9]+}}, %SP, 4;
; PTX64:        add.u64          {{%rd[0-9]+}}, %SPL, 4;
; PTX64:        st.local.u32     [{{%rd[0-9]+}}], {{%r[0-9]+}}
; PTX64:        st.local.u32     [{{%rd[0-9]+}}], {{%r[0-9]+}}
define void @foo4() {
  %A = alloca i32
  %B = alloca i32
  store i32 0, ptr %A
  store i32 0, ptr %B
  call void @bar(ptr %A)
  call void @bar(ptr %B)
  ret void
}

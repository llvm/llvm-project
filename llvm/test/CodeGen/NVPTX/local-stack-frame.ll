; RUN: llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs | FileCheck %s --check-prefix=PTX64
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}

; Ensure we access the local stack properly

; PTX32:        mov.u32          %SPL, __local_depot{{[0-9]+}};
; PTX32:        cvta.local.u32   %SP, %SPL;
; PTX32:        ld.param.u32     %r{{[0-9]+}}, [foo_param_0];
; PTX32:        st.volatile.u32  [%SP+0], %r{{[0-9]+}};
; PTX64:        mov.u64          %SPL, __local_depot{{[0-9]+}};
; PTX64:        cvta.local.u64   %SP, %SPL;
; PTX64:        ld.param.u32     %r{{[0-9]+}}, [foo_param_0];
; PTX64:        st.volatile.u32  [%SP+0], %r{{[0-9]+}};
define void @foo(i32 %a) {
  %local = alloca i32, align 4
  store volatile i32 %a, i32* %local
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
define void @foo2(i32 %a) {
  %local = alloca i32, align 4
  store i32 %a, i32* %local
  call void @bar(i32* %local)
  ret void
}

declare void @bar(i32* %a)

!nvvm.annotations = !{!0}
!0 = !{void (i32)* @foo2, !"kernel", i32 1}

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
  %1 = bitcast [3 x i32]* %local to i32*
  %2 = getelementptr inbounds i32, i32* %1, i32 %a
  store i32 %a, i32* %2
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
  store i32 0, i32* %A
  store i32 0, i32* %B
  call void @bar(i32* %A)
  call void @bar(i32* %B)
  ret void
}

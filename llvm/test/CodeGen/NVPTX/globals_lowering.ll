; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -relocation-model=static | FileCheck %s --check-prefix CHK
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 -relocation-model=static | %ptxas-verify %}

%MyStruct = type { i32, i32, float }
@Gbl = internal addrspace(3) global [1024 x %MyStruct] zeroinitializer

; CHK-LABEL: foo
define void @foo(float %f) {
entry:
  ; CHK: ld.shared.f32  %{{[a-zA-Z0-9]+}}, [Gbl+8];
  %0 = load float, ptr addrspace(3) getelementptr inbounds ([1024 x %MyStruct], ptr addrspace(3) @Gbl, i32 0, i32 0, i32 2)
  %add = fadd float %0, %f
  ; CHK: st.shared.f32   [Gbl+8], %{{[a-zA-Z0-9]+}};
  store float %add, ptr addrspace(3) getelementptr inbounds ([1024 x %MyStruct], ptr addrspace(3) @Gbl, i32 0, i32 0, i32 2)
  ret void
}

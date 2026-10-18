; RUN: opt -S -dxil-resource-type -dxil-resource-access -mtriple=dxil-pc-shadermodel6.3-library %s -o /dev/null
;
; Reproducer: both incoming pointers address row 0, but select different i32
; components. The merged byte offset is 0 or 4. Resource access lowering
; incorrectly uses udiv exact by 16 (poison for offset 4) and always extracts
; component 0, losing the component-1 access on the then branch.

%struct.S = type { i32, i32 }

@CB.str = private unnamed_addr constant [3 x i8] c"CB\00"

define i32 @cbuffer_gep_phi_subrow(i1 %cond) {
entry:
  %handle0 = call target("dx.CBuffer", %struct.S) @llvm.dx.resource.handlefromimplicitbinding(i32 1, i32 0, i32 2, i32 0, ptr @CB.str)
  %ptr0 = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %struct.S) %handle0, i32 0)
  br i1 %cond, label %then, label %merge

then:
  %handle1 = call target("dx.CBuffer", %struct.S) @llvm.dx.resource.handlefromimplicitbinding(i32 1, i32 0, i32 2, i32 1, ptr @CB.str)
  %ptr1 = call ptr addrspace(2) @llvm.dx.resource.getpointer(target("dx.CBuffer", %struct.S) %handle1, i32 0)
  %field1 = getelementptr %struct.S, ptr addrspace(2) %ptr1, i32 0, i32 1
  br label %merge

merge:
  %ptr = phi ptr addrspace(2) [ %ptr0, %entry ], [ %field1, %then ]
  %value = load i32, ptr addrspace(2) %ptr, align 4
  ret i32 %value
}

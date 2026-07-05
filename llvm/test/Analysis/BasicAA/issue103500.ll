; RUN: opt -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output %s 2>&1 | FileCheck %s

target datalayout = "p0:64:64-p5:32:32"

; CHECK: Function: indexing_different_sized_addrspace: 2 pointers, 0 call sites
; CHECK: MayAlias:	i32* %gep.in.0, i32 addrspace(5)* %gep.in.5.1

define i1 @indexing_different_sized_addrspace(ptr addrspace(5) %arg, i64 %arg1, i32 %arg2) {
bb:
  %arg.addrspacecast = addrspacecast ptr addrspace(5) %arg to ptr
  %gep.in.5 = getelementptr i8, ptr addrspace(5) %arg, i32 16
  %gep.in.0 = getelementptr i8, ptr %arg.addrspacecast, i64 %arg1
  %gep.in.5.1 = getelementptr i8, ptr addrspace(5) %gep.in.5, i32 %arg2
  %load.0 = load i32, ptr %gep.in.0, align 4
  %load.1 = load i32, ptr addrspace(5) %gep.in.5.1, align 4
  %cmp = icmp slt i32 %load.0, %load.1
  ret i1 %cmp
}

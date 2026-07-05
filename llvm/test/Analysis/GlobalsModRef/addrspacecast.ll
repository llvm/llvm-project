; RUN: opt -aa-pipeline=globals-aa,basic-aa -passes='require<globals-aa>,aa-eval' -print-all-alias-modref-info -disable-output %s 2>&1 | FileCheck %s

@g0 = internal addrspace(3) global i32 undef

; CHECK-LABEL: test1
; CHECK-DAG: NoAlias: i32* %gp, i32* %p
; CHECK-DAG: NoAlias: i32* %p, i32 addrspace(3)* @g0
; CHECK-DAG: MustAlias: i32* %gp, i32 addrspace(3)* @g0
define i32 @test1(ptr %p) {
  load i32, ptr addrspace(3) @g0
  %gp = addrspacecast ptr addrspace(3) @g0 to ptr
  store i32 0, ptr %gp
  store i32 1, ptr %p
  %v = load i32, ptr %gp
  ret i32 %v
}

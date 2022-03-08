; RUN: opt -aa-pipeline=globals-aa,basic-aa -passes='require<globals-aa>,aa-eval' -print-all-alias-modref-info -disable-output %s 2>&1 | FileCheck %s

@g0 = internal addrspace(3) global i32 undef

; CHECK-LABEL: test1
; CHECK-DAG: NoAlias: i32* %gp, i32* %p
; CHECK-DAG: NoAlias: i32 addrspace(3)* @g0, i32* %p
; CHECK-DAG: MustAlias: i32 addrspace(3)* @g0, i32* %gp
define i32 @test1(i32* %p) {
  %gp = addrspacecast i32 addrspace(3)* @g0 to i32*
  store i32 0, i32* %gp
  store i32 1, i32* %p
  %v = load i32, i32* %gp
  ret i32 %v
}

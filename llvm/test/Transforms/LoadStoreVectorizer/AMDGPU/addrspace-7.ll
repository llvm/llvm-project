; REQUIRES: asserts
; RUN: not --crash opt -mtriple=amdgcn-amd-amdhsa -passes=load-store-vectorizer -S -o - %s
; RUN: not --crash opt -mtriple=amdgcn-amd-amdhsa -aa-pipeline=basic-aa -passes='function(load-store-vectorizer)' -S -o - %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

define { float, float } @f() {
bb:
  %l1 = load float, ptr addrspace(7) null
  %l2 = load float, ptr addrspace(7) getelementptr (i8, ptr addrspace(7) null, i64 24)
  %iv1 = insertvalue { float, float } zeroinitializer, float %l1, 0
  %iv2 = insertvalue { float, float } %iv1, float %l2, 1
  ret { float, float } %iv2
}

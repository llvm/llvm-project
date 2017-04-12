; RUN: llc -march=amdgcn -mtriple=amdgcn-amd-amdhsa-amdgiz -verify-machineinstrs < %s
; Just check the target feature and data layout is accepted without error.

target datalayout = "e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5"
target triple = "amdgcn-amd-amdhsa-amdgiz"

define void @foo() {
entry:
  ret void
}


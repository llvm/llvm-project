; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -o /dev/null %s
;
; Regression test for https://github.com/llvm/llvm-project/issues/176966
; Ensures we don't crash in DAG type legalization when widening
; EXTEND_VECTOR_INREG where the widened input becomes larger than the result.

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define <32 x i8> @backsmith_pure_2() {
entry:
  %0 = load <6 x i8>, ptr addrspace(5) null, align 8
  %shuffle5 = shufflevector <6 x i8> %0, <6 x i8> zeroinitializer,
              <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison,
                          i32 poison, i32 poison, i32 poison, i32 poison,
                          i32 poison, i32 poison, i32 poison, i32 poison,
                          i32 poison, i32 poison, i32 poison, i32 poison,
                          i32 poison, i32 poison, i32 poison, i32 poison,
                          i32 5, i32 poison, i32 poison, i32 poison,
                          i32 poison, i32 poison, i32 poison, i32 poison,
                          i32 poison, i32 poison, i32 poison, i32 poison>
  ret <32 x i8> %shuffle5
}

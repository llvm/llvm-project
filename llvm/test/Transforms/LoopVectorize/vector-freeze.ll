; RUN: opt -passes=loop-vectorize -force-vector-width=16 -force-vector-interleave=1 -S < %s | FileCheck %s
; RUN: opt -passes=loop-vectorize -scalable-vectorization=on -force-target-supports-scalable-vectors=true -force-vector-width=16 -force-vector-interleave=1 -S < %s | FileCheck %s --check-prefix=SVE

define i64 @test(ptr noalias readonly %addr) {
; CHECK-LABEL: @test(
; CHECK:       vector.body:
; CHECK:       freeze <16 x i64>

; SVE-LABEL: @test(
; SVE:       vector.body:
; SVE:       freeze <vscale x 16 x i64>

entry:
  br label %loop

exit:
  ret i64 %tmp4

loop:
  %tmp3 = phi ptr [ %tmp6, %loop ], [ %addr, %entry ]
  %tmp4 = freeze i64 0
  %tmp5 = add i64 0, 0
  %tmp6 = getelementptr inbounds ptr, ptr %tmp3, i64 1
  %tmp7 = icmp eq ptr %tmp6, null
  br i1 %tmp7, label %exit, label %loop
}

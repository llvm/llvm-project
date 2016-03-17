;RUN: llc -march=amdgcn -mcpu=verde < %s | FileCheck %s --check-prefix=CHECK

; Test a simple uniform loop that lives inside non-uniform control flow.

;CHECK-LABEL: {{^}}test1:
;CHECK: s_cbranch_execz
;CHECK: %loop_body
define void @test1(<8 x i32> inreg %rsrc, <2 x i32> %addr.base, i32 %y, i32 %p) #0 {
main_body:
  %cc = icmp eq i32 %p, 0
  br i1 %cc, label %out, label %loop_body

loop_body:
  %counter = phi i32 [ 0, %main_body ], [ %incr, %loop_body ]

  ; Prevent the loop from being optimized out
  call void asm sideeffect "", "" ()

  %incr = add i32 %counter, 1
  %lc = icmp sge i32 %incr, 1000
  br i1 %lc, label %out, label %loop_body

out:
  ret void
}

attributes #0 = { "ShaderType"="0" }
attributes #1 = { nounwind readonly }

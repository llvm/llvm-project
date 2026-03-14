; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -verify-machineinstrs < %s

; Test that the Machine Instruction PHI node doesn't have more than one operand
; from the same predecessor.
define i32 @foo(i32 %a, i32 %b, i1 %c) {
entry:
  br i1 %c, label %switch, label %direct

switch:
  switch i32 %a, label %exit [
    i32 43, label %continue
    i32 45, label %continue
  ]

direct:
  %var = add i32 %b, 1
  br label %continue

continue:
  %var.phi = phi i32 [ %var, %direct ], [ 0, %switch ], [ 0, %switch ]
  ret i32 %var.phi

exit:
  ret i32 1
}

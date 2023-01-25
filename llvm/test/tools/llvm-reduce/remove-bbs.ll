; Test that llvm-reduce can remove uninteresting Basic Blocks, and remove them from instructions (i.e. SwitchInst, BranchInst and IndirectBrInst)
; Note: if an uninteresting BB is the default case for a switch, the instruction is removed altogether (since the default case cannot be replaced)
;
; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=basic-blocks --test FileCheck --test-arg --check-prefix=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -implicit-check-not=uninteresting %s < %t

; CHECK-INTERESTINGNESS: store i32 0
; CHECK-INTERESTINGNESS: store i32 1
; CHECK-INTERESTINGNESS: store i32 2

define void @main() {
interesting:
  store i32 0, ptr null
  ; CHECK-NOT: switch i32 0, label %uninteresting
  switch i32 0, label %uninteresting [
    i32 1, label %interesting2
  ]

uninteresting:
  ret void

interesting2:
  store i32 1, ptr null
  ; CHECK: switch i32 1, label %interesting3
  switch i32 1, label %interesting3 [
    ; CHECK-NOT: i32 0, label %uninteresting
    i32 0, label %uninteresting
    ; CHECK: i32 1, label %interesting3
    i32 1, label %interesting3
  ]

interesting3:
  store i32 2, ptr null
  ; CHECK: br label %interesting2
  br i1 true, label %interesting2, label %uninteresting
}

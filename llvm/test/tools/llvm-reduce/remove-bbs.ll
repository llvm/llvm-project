; Test that llvm-reduce can remove uninteresting Basic Blocks, and remove them from instructions (i.e. SwitchInst, BranchInst and IndirectBrInst)
; Note: if an uninteresting BB is the default case for a switch, the instruction is removed altogether (since the default case cannot be replaced)
;
; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=basic-blocks --test %python --test-arg %p/Inputs/remove-bbs.py %s -o %t
; RUN: FileCheck -implicit-check-not=uninteresting %s < %t

define void @main() {
interesting:
  ; CHECK-NOT: switch i32 0, label %uninteresting
  switch i32 0, label %uninteresting [
    i32 0, label %uninteresting
  ]

uninteresting:
  ret void

interesting2:
  ; CHECK: switch i32 1, label %interesting3
  switch i32 1, label %interesting3 [
    ; CHECK-NOT: i32 0, label %uninteresting
    i32 0, label %uninteresting
    ; CHECK: i32 1, label %interesting3
    i32 1, label %interesting3
  ]

interesting3:
  ; CHECK: br label %interesting2
  br i1 true, label %interesting2, label %uninteresting
}

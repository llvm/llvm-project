; RUN: llc -march=hexagon -O2 < %s | FileCheck %s

; Check that we don't generate .falign directives after function calls at O2.
; We need more than one basic block for this test because MachineBlockPlacement
; will not run on single basic block functions.

declare i32 @f0()

; We don't want faligns after the calls to foo.
; CHECK:     call f0
; CHECK-NOT: falign
; CHECK:     call f0
; CHECK-NOT: falign
; CHECK:     dealloc_return
define i32 @f1(i32 %a0) #0 {
b0:
  %v0 = icmp eq i32 %a0, 0
  br i1 %v0, label %b1, label %b2

b1:                                               ; preds = %b0
  %v1 = call i32 @f0()
  %v2 = call i32 @f0()
  %v3 = add i32 %v1, %v2
  ret i32 %v3

b2:                                               ; preds = %b0
  %v4 = add i32 %a0, 5
  ret i32 %v4
}

attributes #0 = { nounwind }

; RUN: llc -mtriple=powerpc-ibm-aix-xcoff %s -filetype=obj -o %t
; RUN: llvm-objdump %t -d --symbolize-operands --no-show-raw-insn \
; RUN:   | FileCheck %s

;; Expect to find the branch labels.
; CHECK-LABEL: <.text>:
;; TODO: <.internal> should be printed instead of <.text>.
; CHECK-NEXT:         0:      mr 4, 3
; CHECK-NEXT:         4:      li 3, 0
; CHECK-NEXT:         8:      mtctr 4
; CHECK-NEXT:  <L0>:
; CHECK-NEXT:         c:      addi 3, 3, 1
; CHECK-NEXT:        10:      bdnz 0xc <L0>
; CHECK-NEXT:        14:      blr

; CHECK-LABEL: <.foo>:
; CHECK:             5c:      	b 0x64 <L0>
; CHECK-NEXT:  <L2>:
; CHECK-NEXT:        60:      	bf	8, 0x84 <L1>
; CHECK-NEXT:  <L0>:
; CHECK-NEXT:        64:      	mr	3, 31
; CHECK-NEXT:        68:      	bl 0x0 <.text>
; CHECK-NEXT:        6c:      	mr	31, 3
; CHECK-NEXT:        70:      	cmplwi	3, 11
; CHECK-NEXT:        74:      	bt	0, 0x60 <L2>
; CHECK-NEXT:        78:      	bl 0x0 <.text>
; CHECK-NEXT:        7c:      	nop
; CHECK-NEXT:        80:      	b 0x60 <L2>
; CHECK-NEXT:  <L1>:
; CHECK-NEXT:        84:      	lwz 31, 60(1)

define internal i32 @internal(i32 %a) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %next, %for.body ]
  %next = add nuw nsw i32 %i, 1
  %cond = icmp eq i32 %next, %a
  br i1 %cond, label %exit, label %for.body

exit:
  ret i32 %next
}

declare void @extern()

define void @foo(i1 %breakcond) {
entry:
  br label %loop
loop:
  %tmp23phi = phi i32 [ %tmp23, %endif ], [ 0, %entry ]
  %tmp23 = call i32 @internal(i32 %tmp23phi)
  %tmp27 = icmp ult i32 10, %tmp23
  br i1 %tmp27, label %then, label %endif
then:                                             ; preds = %bb
  call void @extern()
  br label %endif
endif:                                             ; preds = %bb28, %bb
  br i1 %breakcond, label %loop, label %loopexit
loopexit:
  ret void
}

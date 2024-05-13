; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu %s -filetype=obj -o %t
; RUN: llvm-objdump %t -d --symbolize-operands --no-show-raw-insn \
; RUN:   | FileCheck %s

;; Expect to find the branch labels.
; CHECK-LABEL: <internal>:
; CHECK:       <L0>:
; CHECK-NEXT:        20:      addi 3, 3, 1
; CHECK-NEXT:        24:      bdnz 0x20 <L0>
; CHECK-NEXT:        28:      blr

; CHECK-LABEL: <foo>:
; CHECK:             6c:      	b 0x74 <L0>
; CHECK-NEXT:  <L2>:
; CHECK-NEXT:        70:      	bf	8, 0x94 <L1>
; CHECK-NEXT:  <L0>:
; CHECK-NEXT:        74:      	clrldi	3, 30, 32
; CHECK-NEXT:        78:      	bl 0x0 <internal>
; CHECK-NEXT:        7c:      	mr	30, 3
; CHECK-NEXT:        80:      	cmplwi	30, 11
; CHECK-NEXT:        84:      	bt	0, 0x70 <L2>
; CHECK-NEXT:        88:      	bl 0x88 <foo+0x48>
; CHECK-NEXT:        8c:      	nop
; CHECK-NEXT:        90:      	b 0x70 <L2>
; CHECK-NEXT:  <L1>:
; CHECK-NEXT:        94:      	ld 30, 32(1)

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

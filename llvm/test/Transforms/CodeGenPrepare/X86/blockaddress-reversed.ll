; RUN: llc <%s | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

; CHECK:      .Ltmp0: # Address of block that was removed by CodeGen
; CHECK-NEXT: .Ltmp1: # Address of block that was removed by CodeGen

define i64 @main(ptr %p, i1 %c1) {
first:
  br label %loop_head

loop_head:                                        ; preds = %loop_end, %first
  br i1 %c1, label %scope0, label %scope1

scope1:                                           ; preds = %loop_head
  br i1 %c1, label %scope1_exit, label %scope2

scope2:                                           ; preds = %scope1
  br i1 %c1, label %scope2_exit, label %inner

inner:                                            ; preds = %scope2
  br label %scope2_exit

scope2_exit:                                      ; preds = %inner, %scope2
  %phi2 = phi ptr [ %p, %inner ], [ null, %scope2 ]
  br label %scope1_exit

scope1_exit:                                      ; preds = %scope2_exit, %scope1
  %phi1 = phi ptr [ %phi2, %scope2_exit ], [ null, %scope1 ]
  %val1 = load i128, ptr %phi1, align 16
  br label %loop_end

scope0:                                           ; preds = %loop_head
  %ptr0 = select i1 %c1, ptr null, ptr %p
  %val0 = load i128, ptr %ptr0, align 16
  br label %loop_end

loop_end:                                         ; preds = %scope0, %scope1_exit
  %storemerge = phi i128 [ %val1, %scope1_exit ], [ %val0, %scope0 ]
  store i128 %storemerge, ptr %p, align 16
  br label %loop_head
}

define void @foo0(ptr %jumpAddr) {
; CHECK:        movq $.Ltmp0, (%rdi)
  store ptr blockaddress(@main, %scope2_exit), ptr %jumpAddr, align 8
  ret void
}

define void @foo1(ptr %jumpAddr) {
; CHECK:        movq $.Ltmp1, (%rdi)
  store ptr blockaddress(@main, %scope1_exit), ptr %jumpAddr, align 8
  ret void
}

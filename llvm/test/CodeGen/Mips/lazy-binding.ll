; RUN: llc -mtriple=mipsel -relocation-model=pic -mips-tail-calls=1 < %s | FileCheck %s

; CHECK-LABEL: foo6:
; CHECK: %while.body
; CHECK: lw  $25, %call16(foo2)(${{[0-9]+}})
; CHECK: jalr $25
; CHECK: %while.end

define void @foo6(i32 %n) {
entry:
  %tobool1 = icmp eq i32 %n, 0
  br i1 %tobool1, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %n.addr.02 = phi i32 [ %dec, %while.body ], [ %n, %entry ]
  %dec = add nsw i32 %n.addr.02, -1
  tail call void @foo2()
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
}

declare void @foo2()

; CHECK-LABEL: foo1:
; CHECK: lw $25, %call16(foo2)(${{[0-9]+}})
; CHECK: jalr $25
; CHECK: lw $25, %call16(foo2)(${{[0-9]+}})
; CHECK: jalr $25
; CHECK: lw $25, %call16(foo2)(${{[0-9]+}})
; CHECK: jalr $25

define void @foo1() {
entry:
  tail call void @foo2()
  tail call void @foo2()
  tail call void @foo2()
  ret void
}

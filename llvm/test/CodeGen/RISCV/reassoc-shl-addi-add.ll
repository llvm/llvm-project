; RUN: llc -mtriple=riscv32-pc-unknown-gnu -mattr=+zba %s -o - | FileCheck %s

declare dso_local i32 @callee1(i32 noundef) local_unnamed_addr
declare dso_local i32 @callee2(i32 noundef, i32 noundef) local_unnamed_addr
declare dso_local i32 @callee(i32 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr

; CHECK-LABEL: t1:
; CHECK: sh2add
; CHECK: sh2add
; CHECK: sh2add
; CHECK: tail callee

define dso_local void @t1(i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d) local_unnamed_addr #0 {
entry:
  %shl = shl i32 %a, 2
  %add = add nsw i32 %shl, 45
  %add1 = add nsw i32 %add, %b
  %add3 = add nsw i32 %add, %c
  %add5 = add nsw i32 %shl, %d
  %call = tail call i32 @callee(i32 noundef %add1, i32 noundef %add1, i32 noundef %add3, i32 noundef %add5)
  ret void
}

; CHECK-LABEL: t2:
; CHECK: slli
; CHECK-NOT: sh2add
; CHECK: tail callee 

define dso_local void @t2(i32 noundef %a, i32 noundef %b, i32 noundef %c) local_unnamed_addr #0 {
entry:
  %shl = shl i32 %a, 2
  %add = add nsw i32 %shl, 42
  %add4 = add nsw i32 %add, %b
  %add7 = add nsw i32 %add, %c
  %call = tail call i32 @callee(i32 noundef %shl, i32 noundef %add, i32 noundef %add4, i32 noundef %add7)
  ret void
}

; CHECK-LABEL: t3
; CHECK slli
; CHECK-NOT: sh2add
; CHECK: tail callee 

define dso_local void @t3(i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d, i32 noundef %e) local_unnamed_addr #0 {
entry:
  %shl = shl i32 %a, 2
  %add = add nsw i32 %shl, 42
  %add1 = add nsw i32 %add, %b
  %add2 = add nsw i32 %add, %c
  %add3 = add nsw i32 %add, %d
  %add4 = add nsw i32 %add, %e
  %call = tail call i32 @callee(i32 noundef %add1, i32 noundef %add2, i32 noundef %add3, i32 noundef %add4)
  ret void
}

; CHECK-LABEL: t4
; CHECK: sh2add
; CHECK-NEXT: addi
; CHECK-NEXT: tail callee1

define dso_local void @t4(i32 noundef %a, i32 noundef %b) local_unnamed_addr #0 {
entry:
  %shl = shl i32 %a, 2
  %add = add nsw i32 %shl, 42
  %add1 = add nsw i32 %add, %b
  %call = tail call i32 @callee1(i32 noundef %add1)
  ret void
}

; CHECK-LABEL: t5
; CHECK: sh2add
; CHECK: sh2add
; CHECK: tail callee2

define dso_local void @t5(i32 noundef %a, i32 noundef %b, i32 noundef %c) local_unnamed_addr #0 {
entry:
  %shl = shl i32 %a, 2
  %add = add nsw i32 %shl, 42
  %add1 = add nsw i32 %add, %b
  %add2 = add nsw i32 %add, %c
  %call = tail call i32 @callee2(i32 noundef %add1, i32 noundef %add2)
  ret void
}

; CHECK-LABEL: t6
; CHECK-DAG: sh2add
; CHECK-DAG: slli
; CHECK: tail callee

define dso_local void @t6(i32 noundef %a, i32 noundef %b) local_unnamed_addr #0 {
entry:
  %shl = shl i32 %a, 2
  %add = add nsw i32 %shl, 42
  %add1 = add nsw i32 %add, %b
  %call = tail call i32 @callee(i32 noundef %add1, i32 noundef %shl, i32 noundef %shl, i32 noundef %shl)
  ret void
}

; CHECK-LABEL: t7
; CHECK: slli
; CHECK-NOT: sh2add
; CHECK: tail callee

define dso_local void @t7(i32 noundef %a, i32 noundef %b) local_unnamed_addr #0 {
entry:
  %shl = shl i32 %a, 2
  %add = add nsw i32 %shl, 42
  %add1 = add nsw i32 %add, %b
  %call = tail call i32 @callee(i32 noundef %add1, i32 noundef %add, i32 noundef %add, i32 noundef %add)
  ret void
}

attributes #0 = { nounwind optsize }

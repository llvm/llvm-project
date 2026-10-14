; RUN: opt < %s -passes=infer-alignment -S | FileCheck %s

define void @alignment_guard(ptr %p) {
; CHECK-LABEL: define void @alignment_guard(
; CHECK:       aligned.path:
; CHECK-NEXT:    [[LOAD:%.*]] = load i64, ptr [[P:%.*]], align 8
entry:
  %p.int = ptrtoint ptr %p to i64
  %masked = and i64 %p.int, 7
  %aligned = icmp eq i64 %masked, 0
  br i1 %aligned, label %aligned.path, label %exit

aligned.path:
  %load = load i64, ptr %p, align 1
  %misaligned = getelementptr i8, ptr %p, i64 4
  %misaligned.load = load i32, ptr %misaligned, align 1
  ; CHECK-NEXT: [[MISALIGNED:%.*]] = getelementptr i8, ptr [[P]], i64 4
  ; CHECK-NEXT: [[MISALIGNED_LOAD:%.*]] = load i32, ptr [[MISALIGNED]], align 4
  br label %exit

exit:
  ret void
}

define void @unaligned_successor(ptr %p) {
; CHECK-LABEL: define void @unaligned_successor(
; CHECK:       unaligned:
; CHECK-NEXT:    [[LOAD:%.*]] = load i64, ptr [[P:%.*]], align 1
entry:
  %p.int = ptrtoint ptr %p to i64
  %masked = and i64 %p.int, 7
  %aligned = icmp eq i64 %masked, 0
  br i1 %aligned, label %exit, label %unaligned

unaligned:
  %load = load i64, ptr %p, align 1
  br label %exit

exit:
  ret void
}
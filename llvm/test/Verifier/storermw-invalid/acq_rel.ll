; RUN: not opt -S -passes=verify < %s 2>&1 | FileCheck %s
; CHECK: storermw cannot have acquire semantics

define void @test(ptr %ptr, i32 %val) {
  storermw add ptr %ptr, i32 %val acq_rel, align 4
  ret void
}

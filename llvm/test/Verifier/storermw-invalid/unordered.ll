; RUN: not opt -S -passes=verify < %s 2>&1 | FileCheck %s
; CHECK: storermw must be at least monotonic

define void @test(ptr %ptr, i32 %val) {
  storermw add ptr %ptr, i32 %val unordered, align 4
  ret void
}

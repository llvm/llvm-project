; RUN: not opt -S -passes=verify < %s 2>&1 | FileCheck %s
; CHECK: xchg operation not valid for storermw

define void @test(ptr %ptr, i32 %val) {
  storermw xchg ptr %ptr, i32 %val seq_cst, align 4
  ret void
}

; RUN: llc < %s -fast-isel -stop-after=finalize-isel -mtriple=x86_64-unknown-linux | FileCheck %s
;
; Verify that llvm.fake.use is lowered to FAKE_USE by FastISel.
; This is relevant for compilers (e.g. Flang) that use llvm.fake.use at O0
; for addresses of variables that are in registers rather than in allocas (e.g.
; addresses of arguments in Flang).

; CHECK-LABEL: name: test_fake_use
; CHECK: hasFakeUses: true
; CHECK: FAKE_USE %{{[0-9]+}}
; CHECK: FAKE_USE %{{[0-9]+}}
; CHECK: FAKE_USE %{{[0-9]+}}
; CHECK: RET64
define void @test_fake_use(ptr %p, i32 %x, i64 %y) {
entry:
  notail call void (...) @llvm.fake.use(ptr %p)
  notail call void (...) @llvm.fake.use(i32 %x)
  notail call void (...) @llvm.fake.use(i64 %y)
  ret void
}

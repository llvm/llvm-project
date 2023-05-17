; RUN: opt -S -passes=gvn-hoist,newgvn -verify-memoryssa < %s | FileCheck %s

; Check that we end up with one load and one store, in the right order
; CHECK-LABEL:  define void @test_it(
; CHECK: store
; CHECK-NOT: store
; CHECK-NOT: load

%rec894.0.1.2.3.12 = type { i16 }

@a = external global %rec894.0.1.2.3.12

define void @test_it() {
bb2:
  store i16 undef, ptr @a, align 1
  %_tmp61 = load i16, ptr @a, align 1
  store i16 undef, ptr @a, align 1
  %_tmp92 = load i16, ptr @a, align 1
  ret void
}

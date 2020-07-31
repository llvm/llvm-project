; RUN: opt -S -enable-loop-simplifycfg-term-folding=true -loop-simplifycfg %s | FileCheck %s
; RUN: opt -S -enable-loop-simplifycfg-term-folding=true -passes='require<domtree>,loop(simplify-cfg)' %s | FileCheck %s

declare i32* @fake_personality_function()
declare void @foo()

define i32 @test_remove_lpad(i1 %exitcond) personality i32* ()* @fake_personality_function {
; CHECK-LABEL: @test_remove_lpad(
entry:
  br label %for.body

for.body:
  br i1 0, label %never, label %next

next:
  br label %latch

latch:
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret i32 0

never:
  invoke void @foo() to label %next unwind label %never-unwind

never-unwind:
; CHECK: never-unwind:
; CHECK-NEXT: unreachable
  %res = landingpad token cleanup
  unreachable
}

define i32 @test_remove_phi_lpad(i1 %exitcond) personality i32* ()* @fake_personality_function {
; CHECK-LABEL: @test_remove_phi_lpad(
entry:
  br label %for.body

for.body:
  br i1 0, label %never, label %next

next:
  br label %latch

latch:
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret i32 0

never:
  invoke void @foo() to label %next unwind label %never-unwind

never-unwind:
; CHECK: never-unwind:
; CHECK-NEXT: ret i32 undef
  %p = phi i32 [1, %never]
  %res = landingpad token cleanup
  ret i32 %p
}

define i32 @test_split_remove_phi_lpad_(i1 %exitcond) personality i32* ()* @fake_personality_function {
; CHECK-LABEL: @test_split_remove_phi_lpad_(
entry:
  invoke void @foo() to label %for.body unwind label %unwind-bb

for.body:
  br i1 0, label %never, label %next

next:
  br label %latch

latch:
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret i32 0

never:
  invoke void @foo() to label %next unwind label %unwind-bb

unwind-bb:
; CHECK: unwind-bb.loopexit:
; CHECK-NEXT: br label %unwind-bb
  %p = phi i32 [1, %never], [2, %entry]
  %res = landingpad token cleanup
  ret i32 %p
}

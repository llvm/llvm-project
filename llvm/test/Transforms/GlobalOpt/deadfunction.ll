; RUN: opt < %s -passes=globalopt -S | FileCheck %s

; CHECK-NOT: test

declare void @aa()
declare void @bb()

; Test that we can erase a function which has a blockaddress referring to it
@test.x = internal unnamed_addr constant [3 x ptr] [ptr blockaddress(@test, %a), ptr blockaddress(@test, %b), ptr blockaddress(@test, %c)], align 16
define internal void @test(i32 %n) nounwind noinline {
entry:
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds [3 x ptr], ptr @test.x, i64 0, i64 %idxprom
  %0 = load ptr, ptr %arrayidx, align 8
  indirectbr ptr %0, [label %a, label %b, label %c]

a:
  tail call void @aa() nounwind
  br label %b

b:
  tail call void @bb() nounwind
  br label %c

c:
  ret void
}

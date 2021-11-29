; RUN: opt < %s -passes=globalopt -S | FileCheck %s

; CHECK-NOT: test

declare void @aa()
declare void @bb()

; Test that we can erase a function which has a blockaddress referring to it
@test.x = internal unnamed_addr constant [3 x i8*] [i8* blockaddress(@test, %a), i8* blockaddress(@test, %b), i8* blockaddress(@test, %c)], align 16
define internal void @test(i32 %n) nounwind noinline {
entry:
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds [3 x i8*], [3 x i8*]* @test.x, i64 0, i64 %idxprom
  %0 = load i8*, i8** %arrayidx, align 8
  indirectbr i8* %0, [label %a, label %b, label %c]

a:
  tail call void @aa() nounwind
  br label %b

b:
  tail call void @bb() nounwind
  br label %c

c:
  ret void
}

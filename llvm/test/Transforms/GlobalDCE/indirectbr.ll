; RUN: opt -S -passes=globaldce < %s | FileCheck %s

@L = internal unnamed_addr constant [3 x ptr] [ptr blockaddress(@test1, %L1), ptr blockaddress(@test1, %L2), ptr null], align 16

; CHECK: @L = internal unnamed_addr constant

define void @test1(i32 %idx) {
entry:
  br label %L1

L1:
  %arrayidx = getelementptr inbounds [3 x ptr], ptr @L, i32 0, i32 %idx
  %l = load ptr, ptr %arrayidx
  indirectbr ptr %l, [label %L1, label %L2]

L2:
  ret void
}

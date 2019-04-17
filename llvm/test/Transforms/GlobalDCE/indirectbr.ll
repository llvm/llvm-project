; RUN: opt -S -globaldce < %s | FileCheck %s

@L = internal unnamed_addr constant [3 x i8*] [i8* blockaddress(@test1, %L1), i8* blockaddress(@test1, %L2), i8* null], align 16

; CHECK: @L = internal unnamed_addr constant

define void @test1(i32 %idx) {
entry:
  br label %L1

L1:
  %arrayidx = getelementptr inbounds [3 x i8*], [3 x i8*]* @L, i32 0, i32 %idx
  %l = load i8*, i8** %arrayidx
  indirectbr i8* %l, [label %L1, label %L2]

L2:
  ret void
}

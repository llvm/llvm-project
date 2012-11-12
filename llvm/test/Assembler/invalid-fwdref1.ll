; RUN: not llvm-as %s -disable-output 2>&1 | grep "invalid forward reference to function as global value!"

define i8* @test1() { ret i8* @test1a }
define void @test1a() { }

; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder < %s

define void @test(i8* swiftself)
; CHECK: define void @test(i8* swiftself)
{
        ret void;
}

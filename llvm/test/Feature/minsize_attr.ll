; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define void @test1() minsize {
; CHECK: define void @test1() minsize
        ret void
}


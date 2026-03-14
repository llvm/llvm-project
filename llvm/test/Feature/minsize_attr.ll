; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define void @test1() minsize {
; CHECK: define void @test1() #0
        ret void
}

; CHECK: attributes #0 = { minsize }

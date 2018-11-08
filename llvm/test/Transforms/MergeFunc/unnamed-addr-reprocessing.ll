; RUN: opt -S -mergefunc < %s | FileCheck %s

; After test3 and test4 have been merged, we should detect that
; test1 and test2 can also be merged.

; CHECK: define void @test4() unnamed_addr
; CHECK-NEXT: tail call void @test3()
; CHECK: define void @test2() unnamed_addr
; CHECK-NEXT: tail call void @test1()

declare void @dummy()
  
define void @test1() unnamed_addr {
    call void @test3()
    call void @test3()
    ret void
}

define void @test2() unnamed_addr {
    call void @test4()
    call void @test4()
    ret void
}

define void @test3() unnamed_addr {
    call void @dummy()
    call void @dummy()
    ret void
}

define void @test4() unnamed_addr {
    call void @dummy()
    call void @dummy()
    ret void
}

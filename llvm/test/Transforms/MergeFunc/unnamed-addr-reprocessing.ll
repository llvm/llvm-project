; RUN: opt -S -mergefunc < %s | FileCheck %s

; After the merge of test5 and test6, we can merge test3 and test4,
; then test1 and test2.

; CHECK: define void @test6() unnamed_addr
; CHECK-NEXT: tail call void @test5()
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
    call void @test5()
    call void @test5()
    ret void
}

define void @test4() unnamed_addr {
    call void @test6()
    call void @test6()
    ret void
}

define void @test5() unnamed_addr {
    call void @dummy()
    call void @dummy()
    ret void
}

define void @test6() unnamed_addr {
    call void @dummy()
    call void @dummy()
    ret void
}

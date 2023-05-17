; RUN: opt -S -passes=mergefunc < %s | FileCheck %s

; We should normalize to test2 rather than test1,
; because it allows us to drop test1 entirely

; CHECK-NOT: define internal void @test1() unnamed_addr
; CHECK: define void @test3() unnamed_addr
; CHECK-NEXT: call void @test2()
; CHECK-NEXT: call void @test2()

declare void @dummy()

define internal void @test1() unnamed_addr {
    call void @dummy()
    call void @dummy()
    ret void
}

define void @test2() unnamed_addr {
    call void @dummy()
    call void @dummy()
    ret void
}

define void @test3() unnamed_addr {
    call void @test1()
    call void @test2()
    ret void
}

; We should normalize to the existing test6 rather than
; to a new anonymous strong backing function

; CHECK: define weak void @test5()
; CHECK-NEXT: tail call void @test6()
; CHECK: define weak void @test4()
; CHECK-NEXT: tail call void @test6()

declare void @dummy2()

define weak void @test4() {
    call void @dummy2()
    call void @dummy2()
    ret void
}
define weak void @test5() {
    call void @dummy2()
    call void @dummy2()
    ret void
}
define void @test6() {
    call void @dummy2()
    call void @dummy2()
    ret void
}

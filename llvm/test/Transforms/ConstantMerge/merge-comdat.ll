; RUN: opt -passes=constmerge -S < %s | FileCheck %s

; @ComdatGlobal1 can be merged into @RegularGlobal which does not have
; a comdat group and should not get dropped by the linker.
; Same for @ComdatGlobal2 into @PrivateGlobal.

$test = comdat any

@ComdatGlobal1 = private unnamed_addr constant i32 111, comdat($test)
@RegularGlobal = constant i32 111

; CHECK NOT: @ComdatGlobal1
; CHECK: @RegularGlobal

@ComdatGlobal2 = private unnamed_addr constant i32 222, comdat($test)
@PrivateGlobal = private unnamed_addr constant i32 222

; CHECK NOT: @ComdatGlobal2
; CHECK: @PrivateGlobal

define void @test(ptr %P) comdat {
        store ptr @ComdatGlobal1, ptr %P
        store ptr @RegularGlobal, ptr %P
        store ptr @ComdatGlobal2, ptr %P
        store ptr @PrivateGlobal, ptr %P
        ret void
}

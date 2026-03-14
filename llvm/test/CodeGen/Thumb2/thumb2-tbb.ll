; RUN: llc < %s -mtriple=thumbv7-apple-darwin | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -relocation-model=pic | FileCheck %s
; RUN: llc < %s -mtriple=thumbv6m-apple-darwin -relocation-model=static | FileCheck %s --check-prefix=THUMB1
; RUN: llc < %s -mtriple=thumbv6m-apple-darwin -relocation-model=pic | FileCheck %s --check-prefix=THUMB1

define void @bar(i32 %n.u) {
entry:
; CHECK-LABEL: bar:
; CHECK: tbb
; CHECK: .data_region jt8
; CHECK: .end_data_region
; CHECK-NEXT: .p2align 1

; THUMB1-LABEL: bar:
; THUMB1: add pc, r0
; THUMB1: .data_region jt8
; THUMB1: .byte (LBB0_3-(LCPI0_0+4))/2
; THUMB1: .end_data_region
; THUMB1-NEXT: .p2align 1

    switch i32 %n.u, label %bb12 [i32 1, label %bb i32 2, label %bb6 i32 4, label %bb7 i32 5, label %bb8 i32 6, label %bb10 i32 7, label %bb1 i32 8, label %bb3 i32 9, label %bb4 i32 10, label %bb9 i32 11, label %bb2 i32 12, label %bb5 i32 13, label %bb11 ]
bb:
    tail call void(...) @foo1()
    ret void
bb1:
    tail call void(...) @foo2()
    ret void
bb2:
    tail call void(...) @foo6()
    ret void
bb3:
    tail call void(...) @foo3()
    ret void
bb4:
    tail call void(...) @foo4()
    ret void
bb5:
    tail call void(...) @foo5()
    ret void
bb6:
    tail call void(...) @foo1()
    ret void
bb7:
    tail call void(...) @foo2()
    ret void
bb8:
    tail call void(...) @foo6()
    ret void
bb9:
    tail call void(...) @foo3()
    ret void
bb10:
    tail call void(...) @foo4()
    ret void
bb11:
    tail call void(...) @foo5()
    ret void
bb12:
    tail call void(...) @foo6()
    ret void
}

declare void @foo1(...)
declare void @foo2(...)
declare void @foo6(...)
declare void @foo3(...)
declare void @foo4(...)
declare void @foo5(...)

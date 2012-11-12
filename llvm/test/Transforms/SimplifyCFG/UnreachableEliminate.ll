; RUN: opt < %s -simplifycfg -S | FileCheck %s

define void @test1(i1 %C, i1* %BP) {
; CHECK: @test1
; CHECK: entry:
; CHECK-NEXT: ret void
entry:
        br i1 %C, label %T, label %F
T:
        store i1 %C, i1* %BP
        unreachable
F:
        ret void
}

define void @test2() {
; CHECK: @test2
; CHECK: entry:
; CHECK-NEXT: call void @test2()
; CHECK-NEXT: ret void
entry:
        invoke void @test2( )
                        to label %N unwind label %U
U:
        unreachable
N:
        ret void
}

define i32 @test3(i32 %v) {
; CHECK: @test3
; CHECK: entry:
; CHECK-NEXT: [[CMP:%[A-Za-z0-9]+]] = icmp eq i32 %v, 2
; CHECK-NEXT: select i1 [[CMP]], i32 2, i32 1
; CHECK-NEXT: ret
entry:
        switch i32 %v, label %default [
                 i32 1, label %U
                 i32 2, label %T
        ]
default:
        ret i32 1
U:
        unreachable
T:
        ret i32 2
}

; PR9450
define i32 @test4(i32 %v) {
; CHECK: entry:
; CHECK-NEXT:  switch i32 %v, label %T [
; CHECK-NEXT:    i32 3, label %V
; CHECK-NEXT:    i32 2, label %U
; CHECK-NEXT:  ]

entry:
        br label %SWITCH
V:
        ret i32 7
SWITCH:
        switch i32 %v, label %default [
                 i32 1, label %T
                 i32 2, label %U
                 i32 3, label %V
        ]
default:
        unreachable
U:
        ret i32 1
T:
        ret i32 2
}

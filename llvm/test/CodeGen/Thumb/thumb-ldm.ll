; RUN: llc < %s -mtriple=thumbv6m-eabi -verify-machineinstrs -o - | FileCheck %s

@X = external global [0 x i32]          ; <ptr> [#uses=5]

define i32 @t1() {
; CHECK-LABEL: t1:
; CHECK: push {r7, lr}
; CHECK: ldm
; CHECK: pop {r7, pc}
        %tmp = load i32, ptr @X            ; <i32> [#uses=1]
        %tmp3 = load i32, ptr getelementptr ([0 x i32], ptr @X, i32 0, i32 1)           ; <i32> [#uses=1]
        %tmp4 = call i32 @f1( i32 %tmp, i32 %tmp3 )                ; <i32> [#uses=1]
        ret i32 %tmp4
}

define i32 @t2() {
; CHECK-LABEL: t2:
; CHECK: push {r7, lr}
; CHECK: ldm
; CHECK: pop {r7, pc}
        %tmp = load i32, ptr getelementptr ([0 x i32], ptr @X, i32 0, i32 2)            ; <i32> [#uses=1]
        %tmp3 = load i32, ptr getelementptr ([0 x i32], ptr @X, i32 0, i32 3)           ; <i32> [#uses=1]
        %tmp5 = load i32, ptr getelementptr ([0 x i32], ptr @X, i32 0, i32 4)           ; <i32> [#uses=1]
        %tmp6 = call i32 @f2( i32 %tmp, i32 %tmp3, i32 %tmp5 )             ; <i32> [#uses=1]
        ret i32 %tmp6
}

define i32 @t3() {
; CHECK-LABEL: t3:
; CHECK: push {r7, lr}
; CHECK: ldm
; CHECK: pop {r7, pc}
        %tmp = load i32, ptr getelementptr ([0 x i32], ptr @X, i32 0, i32 1)            ; <i32> [#uses=1]
        %tmp3 = load i32, ptr getelementptr ([0 x i32], ptr @X, i32 0, i32 2)           ; <i32> [#uses=1]
        %tmp5 = load i32, ptr getelementptr ([0 x i32], ptr @X, i32 0, i32 3)           ; <i32> [#uses=1]
        %tmp6 = call i32 @f2( i32 %tmp, i32 %tmp3, i32 %tmp5 )             ; <i32> [#uses=1]
        ret i32 %tmp6
}

declare i32 @f1(i32, i32)

declare i32 @f2(i32, i32, i32)

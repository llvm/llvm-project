; RUN: llc -mtriple=xcore < %s | FileCheck %s

define i32 @switch(i32 %i) {
entry:
        switch i32 %i, label %default [
                 i32 0, label %bb0
                 i32 1, label %bb1
                 i32 2, label %bb2
                 i32 3, label %bb3
                 i32 4, label %bb4
                 i32 5, label %bb5
                 i32 6, label %bb6
                 i32 7, label %bb7
                 i32 8, label %bb8
                 i32 9, label %bb9
                 i32 10, label %bb10
                 i32 11, label %bb11
                 i32 12, label %bb12
                 i32 13, label %bb13
                 i32 14, label %bb14
                 i32 15, label %bb15
                 i32 16, label %bb16
                 i32 17, label %bb17
                 i32 18, label %bb18
                 i32 19, label %bb19
                 i32 20, label %bb20
                 i32 21, label %bb21
                 i32 22, label %bb22
                 i32 23, label %bb23
                 i32 24, label %bb24
                 i32 25, label %bb25
                 i32 26, label %bb26
                 i32 27, label %bb27
                 i32 28, label %bb28
                 i32 29, label %bb29
                 i32 30, label %bb30
                 i32 31, label %bb31
                 i32 32, label %bb32
                 i32 33, label %bb33
                 i32 34, label %bb34
                 i32 35, label %bb35
                 i32 36, label %bb36
                 i32 37, label %bb37
                 i32 38, label %bb38
                 i32 39, label %bb39
        ]
; CHECK: shl
; CHECK: bru
; CHECK: .jmptable
bb0:
        ret i32 0
bb1:
        ret i32 1
bb2:
        ret i32 2
bb3:
        ret i32 3
bb4:
        ret i32 4
bb5:
        ret i32 5
bb6:
        ret i32 6
bb7:
        ret i32 7
bb8:
        ret i32 8
bb9:
        ret i32 9
bb10:
        ret i32 0
bb11:
        ret i32 1
bb12:
        ret i32 2
bb13:
        ret i32 3
bb14:
        ret i32 4
bb15:
        ret i32 5
bb16:
        ret i32 6
bb17:
        ret i32 7
bb18:
        ret i32 8
bb19:
        ret i32 9
bb20:
        ret i32 0
bb21:
        ret i32 1
bb22:
        ret i32 2
bb23:
        ret i32 3
bb24:
        ret i32 4
bb25:
        ret i32 5
bb26:
        ret i32 6
bb27:
        ret i32 7
bb28:
        ret i32 8
bb29:
        ret i32 9
bb30:
        ret i32 0
bb31:
        ret i32 1
bb32:
        ret i32 2
bb33:
        ret i32 3
bb34:
        ret i32 4
bb35:
        ret i32 5
bb36:
        ret i32 6
bb37:
        ret i32 7
bb38:
        ret i32 8
bb39:
        ret i32 9
default:
        ret i32 0
}

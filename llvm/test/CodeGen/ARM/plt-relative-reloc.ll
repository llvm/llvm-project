; RUN: llc -mtriple=armv7-unknown-linux -o - %s | FileCheck %s

@vtable = constant [4 x i32] [i32 0,
    i32 sub (i32 ptrtoint (void ()* @fn1 to i32), i32 ptrtoint (i32* getelementptr ([4 x i32], [4 x i32]* @vtable, i32 0, i32 1) to i32)),
    i32 sub (i32 ptrtoint (void ()* @fn2 to i32), i32 ptrtoint (i32* getelementptr ([4 x i32], [4 x i32]* @vtable, i32 0, i32 1) to i32)),
    i32 sub (i32 ptrtoint (void ()* @fn3 to i32), i32 ptrtoint (i32* getelementptr ([4 x i32], [4 x i32]* @vtable, i32 0, i32 1) to i32))
]

declare void @fn1() unnamed_addr
declare void @fn2() unnamed_addr
declare void @fn3()

; CHECK: .long 0
; CHECK-NEXT: .long (fn1(prel31)-vtable)-4
; CHECK-NEXT: .long (fn2(prel31)-vtable)-4
; CHECK-NEXT: .long (fn3-vtable)-4

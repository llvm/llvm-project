; RUN: llc -mtriple=armv7-unknown-linux -o - %s | FileCheck %s

@vtable = constant [4 x i32] [i32 0,
    i32 sub (i32 ptrtoint (ptr @fn1 to i32), i32 ptrtoint (ptr getelementptr ([4 x i32], ptr @vtable, i32 0, i32 1) to i32)),
    i32 sub (i32 ptrtoint (ptr @fn2 to i32), i32 ptrtoint (ptr getelementptr ([4 x i32], ptr @vtable, i32 0, i32 1) to i32)),
    i32 sub (i32 ptrtoint (ptr @fn3 to i32), i32 ptrtoint (ptr getelementptr ([4 x i32], ptr @vtable, i32 0, i32 1) to i32))
]

declare void @fn1() unnamed_addr
declare void @fn2() unnamed_addr
declare void @fn3()

;; Create a PC-relative relocation that the linker might decline if the addend symbol is preemptible.
; CHECK: .long 0
; CHECK-NEXT: .long fn1-vtable-4
; CHECK-NEXT: .long fn2-vtable-4
; CHECK-NEXT: .long fn3-vtable-4

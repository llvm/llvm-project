; Verify the assembler produces the expected expressions
; RUN: llc -mtriple=x86_64-pc-win32 %s -o - | FileCheck %s

;;;; some globals

@g1 = constant i32 1;
@g2 = constant i32 2;
@g3 = constant i32 3;
@g4 = constant i32 4;
@__ImageBase = external global ptr;

;;;; cross-section relative relocations

; CHECK: .quad (g3-t1)+4
@t1 = global i64 add(i64 sub(i64 ptrtoint(ptr @g3 to i64), i64 ptrtoint(ptr @t1 to i64)), i64 4), section ".fix"
; CHECK: .quad g3-t2
@t2 = global i64 sub(i64 ptrtoint(ptr @g3 to i64), i64 ptrtoint(ptr @t2 to i64)), section ".fix"
; CHECK: .quad (g3-t3)-4
@t3 = global i64 sub(i64 sub(i64 ptrtoint(ptr @g3 to i64), i64 ptrtoint(ptr @t3 to i64)), i64 4), section ".fix"
; CHECK: .long g3-t4
@t4 = global i32 trunc(i64 sub(i64 ptrtoint(ptr @g3 to i64), i64 ptrtoint(ptr @t4 to i64)) to i32), section ".fix"

;;;; image base relocation

; CHECK: .long g3@IMGREL{{$}}
@t5 = global i32 trunc(i64 sub(i64 ptrtoint(ptr @g3 to i64), i64 ptrtoint(ptr @__ImageBase to i64)) to i32), section ".fix"

; CHECK: .long g3@IMGREL+4{{$}}
@t6 = global i32 trunc(i64 sub(i64 ptrtoint(ptr getelementptr (i32, ptr @g3, i32 1) to i64), i64 ptrtoint(ptr @__ImageBase to i64)) to i32), section ".fix"

;;;; cross-section relative with source offset

%struct.EEType = type { [2 x i8], i64, i32}

; CHECK: .long (g3-t7)-16
@t7 = global %struct.EEType { 
        [2 x i8] c"\01\02", 
        i64 256,
        i32 trunc(i64 sub(i64 ptrtoint(ptr @g3 to i64), i64 ptrtoint(ptr getelementptr inbounds (%struct.EEType, ptr @t7, i32 0, i32 2) to i64)) to i32 )
}, section ".fix"

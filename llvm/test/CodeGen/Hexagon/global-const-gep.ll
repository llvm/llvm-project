; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that global constant GEPs are calculated correctly
;
target triple = "hexagon-unknown--elf"

%s.0 = type { i32, i64, [100 x i8] }

@g0 = common global %s.0 zeroinitializer, align 8
@g1 = global ptr getelementptr inbounds (%s.0, ptr @g0, i32 0, i32 2, i32 10), align 4
; CHECK-LABEL: g1:
; CHECK: .word g0+26

@g2 = common global [100 x i8] zeroinitializer, align 8
@g3 = global ptr getelementptr inbounds ([100 x i8], ptr @g2, i32 0, i32 10), align 4
; CHECK-LABEL: g3:
; CHECK: .word g2+10

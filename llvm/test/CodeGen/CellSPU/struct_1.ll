; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: llvm-as -o - %s | llc -march=cellspu -mattr=large_mem > %t2.s
; RUN: grep lqa     %t1.s | count 5
; RUN: grep lqd     %t1.s | count 11
; RUN: grep rotqbyi %t1.s | count 7
; RUN: grep xshw    %t1.s | count 1
; RUN: grep andi    %t1.s | count 5
; RUN: grep cbd     %t1.s | count 3
; RUN: grep chd     %t1.s | count 1
; RUN: grep cwd     %t1.s | count 3
; RUN: grep shufb   %t1.s | count 7
; RUN: grep stqd    %t1.s | count 7
; RUN: grep iohl    %t2.s | count 16
; RUN: grep ilhu    %t2.s | count 16
; RUN: grep lqd     %t2.s | count 16
; RUN: grep rotqbyi %t2.s | count 7
; RUN: grep xshw    %t2.s | count 1
; RUN: grep andi    %t2.s | count 5
; RUN: grep cbd     %t2.s | count 3
; RUN: grep chd     %t2.s | count 1
; RUN: grep cwd     %t2.s | count 3
; RUN: grep shufb   %t2.s | count 7
; RUN: grep stqd    %t2.s | count 7

; ModuleID = 'struct_1.bc'
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

; struct hackstate {
;   unsigned char c1;   // offset 0 (rotate left by 13 bytes to byte 3)
;   unsigned char c2;   // offset 1 (rotate left by 14 bytes to byte 3)
;   unsigned char c3;   // offset 2 (rotate left by 15 bytes to byte 3)
;   int           i1;   // offset 4 (rotate left by 4 bytes to byte 0)
;   short         s1;   // offset 8 (rotate left by 6 bytes to byte 2)
;   int           i2;   // offset 12 [ignored]
;   unsigned char c4;   // offset 16 [ignored]
;   unsigned char c5;   // offset 17 [ignored]
;   unsigned char c6;   // offset 18 (rotate left by 14 bytes to byte 3)
;   unsigned char c7;   // offset 19 (no rotate, in preferred slot)
;   int           i3;   // offset 20 [ignored]
;   int           i4;   // offset 24 [ignored]
;   int           i5;   // offset 28 [ignored]
;   int           i6;   // offset 32 (no rotate, in preferred slot)
; }
%struct.hackstate = type { i8, i8, i8, i32, i16, i32, i8, i8, i8, i8, i32, i32, i32, i32 }

; struct hackstate state = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
@state = global %struct.hackstate zeroinitializer, align 16

define i8 @get_hackstate_c1() zeroext nounwind  {
entry:
        %tmp2 = load i8* getelementptr (%struct.hackstate* @state, i32 0, i32 0), align 16
        ret i8 %tmp2
}

define i8 @get_hackstate_c2() zeroext nounwind  {
entry:
        %tmp2 = load i8* getelementptr (%struct.hackstate* @state, i32 0, i32 1), align 16
        ret i8 %tmp2
}

define i8 @get_hackstate_c3() zeroext nounwind  {
entry:
        %tmp2 = load i8* getelementptr (%struct.hackstate* @state, i32 0, i32 2), align 16
        ret i8 %tmp2
}

define i32 @get_hackstate_i1() nounwind  {
entry:
        %tmp2 = load i32* getelementptr (%struct.hackstate* @state, i32 0, i32 3), align 16
        ret i32 %tmp2
}

define i16 @get_hackstate_s1() signext nounwind  {
entry:
        %tmp2 = load i16* getelementptr (%struct.hackstate* @state, i32 0, i32 4), align 16
        ret i16 %tmp2
}

define i8 @get_hackstate_c6() zeroext nounwind  {
entry:
        %tmp2 = load i8* getelementptr (%struct.hackstate* @state, i32 0, i32 8), align 16
        ret i8 %tmp2
}

define i8 @get_hackstate_c7() zeroext nounwind  {
entry:
        %tmp2 = load i8* getelementptr (%struct.hackstate* @state, i32 0, i32 9), align 16
        ret i8 %tmp2
}

define i32 @get_hackstate_i3() nounwind  {
entry:
        %tmp2 = load i32* getelementptr (%struct.hackstate* @state, i32 0, i32 10), align 16
        ret i32 %tmp2
}

define i32 @get_hackstate_i6() nounwind  {
entry:
        %tmp2 = load i32* getelementptr (%struct.hackstate* @state, i32 0, i32 13), align 16
        ret i32 %tmp2
}

define void @set_hackstate_c1(i8 zeroext  %c) nounwind  {
entry:
        store i8 %c, i8* getelementptr (%struct.hackstate* @state, i32 0, i32 0), align 16
        ret void
}

define void @set_hackstate_c2(i8 zeroext  %c) nounwind  {
entry:
        store i8 %c, i8* getelementptr (%struct.hackstate* @state, i32 0, i32 1), align 16
        ret void
}

define void @set_hackstate_c3(i8 zeroext  %c) nounwind  {
entry:
        store i8 %c, i8* getelementptr (%struct.hackstate* @state, i32 0, i32 2), align 16
        ret void
}

define void @set_hackstate_i1(i32 %i) nounwind  {
entry:
        store i32 %i, i32* getelementptr (%struct.hackstate* @state, i32 0, i32 3), align 16
        ret void
}

define void @set_hackstate_s1(i16 signext  %s) nounwind  {
entry:
        store i16 %s, i16* getelementptr (%struct.hackstate* @state, i32 0, i32 4), align 16
        ret void
}

define void @set_hackstate_i3(i32 %i) nounwind  {
entry:
        store i32 %i, i32* getelementptr (%struct.hackstate* @state, i32 0, i32 10), align 16
        ret void
}

define void @set_hackstate_i6(i32 %i) nounwind  {
entry:
        store i32 %i, i32* getelementptr (%struct.hackstate* @state, i32 0, i32 13), align 16
        ret void
}

; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux  -tailcallopt  | FileCheck %s

; FIXME: Win64 does not support byval.

; Expect the entry point.
; CHECK-LABEL: tailcaller:

; Expect 2 rep;movs because of tail call byval lowering.
; CHECK: rep;
; CHECK: rep;

; A sequence of copyto/copyfrom virtual registers is used to deal with byval
; lowering appearing after moving arguments to registers. The following two
; checks verify that the register allocator changes those sequences to direct
; moves to argument register where it can (for registers that are not used in
; byval lowering - not rsi, not rdi, not rcx).
; Expect argument 4 to be moved directly to register edx.
; CHECK: movl $7, %edx

; Expect argument 6 to be moved directly to register r8.
; CHECK: movl $17, %r8d

; Expect not call but jmp to @tailcallee.
; CHECK: jmp tailcallee

; Expect the trailer.
; CHECK: .size tailcaller

%struct.s = type { i64, i64, i64, i64, i64, i64, i64, i64,
                   i64, i64, i64, i64, i64, i64, i64, i64,
                   i64, i64, i64, i64, i64, i64, i64, i64 }

declare  fastcc i64 @tailcallee(%struct.s* byval(%struct.s) %a, i64 %val, i64 %val2, i64 %val3, i64 %val4, i64 %val5)


define  fastcc i64 @tailcaller(i64 %b, %struct.s* byval(%struct.s) %a) {
entry:
        %tmp2 = getelementptr %struct.s, %struct.s* %a, i32 0, i32 1
        %tmp3 = load i64, i64* %tmp2, align 8
        %tmp4 = tail call fastcc i64 @tailcallee(%struct.s* byval(%struct.s) %a , i64 %tmp3, i64 %b, i64 7, i64 13, i64 17)
        ret i64 %tmp4
}

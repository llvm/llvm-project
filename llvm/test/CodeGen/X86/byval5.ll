; RUN: llc < %s -mtriple=x86_64-linux -mattr=-avx | FileCheck %s -check-prefix=X64
; X64-NOT:     movsq
; X64:     rep
; X64-NOT:     rep
; X64:     movsq
; X64-NOT:     movsq
; X64:     rep
; X64-NOT:     rep
; X64:     movsq
; X64-NOT:     rep
; X64-NOT:     movsq

; Win64 has not supported byval yet.

; RUN: llc < %s -march=x86 -mattr=-avx | FileCheck %s -check-prefix=X32
; X32-NOT:     movsl
; X32:     rep
; X32-NOT:     rep
; X32:     movsl
; X32-NOT:     movsl
; X32:     rep
; X32-NOT:     rep
; X32:     movsl
; X32-NOT:     rep
; X32-NOT:     movsl

%struct.s = type { i8, i8, i8, i8, i8, i8, i8, i8,
                   i8, i8, i8, i8, i8, i8, i8, i8,
                   i8, i8, i8, i8, i8, i8, i8, i8,
                   i8, i8, i8, i8, i8, i8, i8, i8,
                   i8, i8, i8, i8, i8, i8, i8, i8,
                   i8, i8, i8, i8, i8, i8, i8, i8,
                   i8, i8, i8, i8, i8, i8, i8, i8,
                   i8, i8, i8, i8, i8, i8, i8, i8,
                   i8, i8, i8, i8, i8, i8, i8, i8,
                   i8, i8, i8, i8, i8, i8, i8, i8,
                   i8, i8, i8, i8, i8, i8, i8, i8,
                   i8, i8, i8, i8, i8, i8, i8, i8,
                   i8, i8, i8, i8, i8, i8, i8, i8,
                   i8, i8, i8, i8, i8, i8, i8, i8,
                   i8, i8, i8, i8, i8, i8, i8, i8,
                   i8, i8, i8, i8, i8, i8, i8, i8,
                   i8 }


define void @g(i8 signext  %a1, i8 signext  %a2, i8 signext  %a3,
	 i8 signext  %a4, i8 signext  %a5, i8 signext  %a6) {
entry:
        %a = alloca %struct.s
        %tmp = getelementptr %struct.s, %struct.s* %a, i32 0, i32 0
        store i8 %a1, i8* %tmp, align 8
        %tmp2 = getelementptr %struct.s, %struct.s* %a, i32 0, i32 1
        store i8 %a2, i8* %tmp2, align 8
        %tmp4 = getelementptr %struct.s, %struct.s* %a, i32 0, i32 2
        store i8 %a3, i8* %tmp4, align 8
        %tmp6 = getelementptr %struct.s, %struct.s* %a, i32 0, i32 3
        store i8 %a4, i8* %tmp6, align 8
        %tmp8 = getelementptr %struct.s, %struct.s* %a, i32 0, i32 4
        store i8 %a5, i8* %tmp8, align 8
        %tmp10 = getelementptr %struct.s, %struct.s* %a, i32 0, i32 5
        store i8 %a6, i8* %tmp10, align 8
        call void @f( %struct.s* byval %a )
        call void @f( %struct.s* byval %a )
        ret void
}

declare void @f(%struct.s* byval)

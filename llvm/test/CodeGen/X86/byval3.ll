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

%struct.s = type { i32, i32, i32, i32, i32, i32, i32, i32,
                   i32, i32, i32, i32, i32, i32, i32, i32,
                   i32, i32, i32, i32, i32, i32, i32, i32,
                   i32, i32, i32, i32, i32, i32, i32, i32,
                   i32 }

define void @g(i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6) nounwind {
entry:
        %d = alloca %struct.s, align 16
        %tmp = getelementptr %struct.s, %struct.s* %d, i32 0, i32 0
        store i32 %a1, i32* %tmp, align 16
        %tmp2 = getelementptr %struct.s, %struct.s* %d, i32 0, i32 1
        store i32 %a2, i32* %tmp2, align 16
        %tmp4 = getelementptr %struct.s, %struct.s* %d, i32 0, i32 2
        store i32 %a3, i32* %tmp4, align 16
        %tmp6 = getelementptr %struct.s, %struct.s* %d, i32 0, i32 3
        store i32 %a4, i32* %tmp6, align 16
        %tmp8 = getelementptr %struct.s, %struct.s* %d, i32 0, i32 4
        store i32 %a5, i32* %tmp8, align 16
        %tmp10 = getelementptr %struct.s, %struct.s* %d, i32 0, i32 5
        store i32 %a6, i32* %tmp10, align 16
        call void @f( %struct.s* byval %d)
        call void @f( %struct.s* byval %d)
        ret void
}

declare void @f(%struct.s* byval)

; RUN: llc -mtriple x86_64-pc-linux < %s | FileCheck %s

%struct.note = type { %struct.Elf32_Nhdr, [7 x i8], %struct.payload }
%struct.Elf32_Nhdr = type { i32, i32, i32 }
%struct.payload = type { i16 }

@foonote = internal constant %struct.note { %struct.Elf32_Nhdr { i32 7, i32 2, i32 17 }, [7 x i8] c"foobar\00", %struct.payload { i16 23 } }, section ".note.foo", align 4

; CHECK:		.section	.note.foo,"a",@note
; CHECK-NEXT: .p2align	2
; CHECK-NEXT: foonote:
; CHECK-NEXT: 	.long	7
; CHECK-NEXT: 	.long	2
; CHECK-NEXT: 	.long	17
; CHECK-NEXT: 	.asciz	"foobar"
; CHECK-NEXT: 	.zero	1
; CHECK-NEXT: 	.short	23
; CHECK-NEXT: 	.zero	2
; CHECK-NEXT: 	.size	foonote, 24

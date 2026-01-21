; RUN: llc -mtriple aarch64-linux-gnu -mattr=+pauth -filetype=asm -o - %s | FileCheck %s

; CHECK: nullref:
; CHECK-NEXT: [[PLACE:.*]]:
; CHECK-NEXT: .section .text.startup
; CHECK-NEXT: [[FUNC:.*]]:
; CHECK-NEXT: movz x0, #0
; CHECK-NEXT: mov x1, #1
; CHECK-NEXT: b __emupac_pacda
; CHECK-NEXT: .section .rodata
; CHECK-NEXT: .xword [[FUNC]]@FUNCINIT
@nullref = constant ptr ptrauth (ptr null, i32 2, i64 1, ptr null), align 8

@dsolocal = external dso_local global i8

; CHECK: dsolocalref:
; CHECK-NEXT: [[PLACE:.*]]:
; CHECK-NEXT: .section .text.startup
; CHECK-NEXT: [[FUNC:.*]]:
; CHECK-NEXT: adrp x0, dsolocal
; CHECK-NEXT: add x0, x0, :lo12:dsolocal
; CHECK-NEXT: mov x1, #2
; CHECK-NEXT: b __emupac_pacda
; CHECK-NEXT: .section .data.rel.ro
; CHECK-NEXT: .xword [[FUNC]]@FUNCINIT
@dsolocalref = constant ptr ptrauth (ptr @dsolocal, i32 2, i64 2, ptr null), align 8

@ds = external global i8

; CHECK: dsolocalrefds:
; CHECK-NEXT: [[PLACE:.*]]:
; CHECK-NEXT: .section .text.startup
; CHECK-NEXT: [[FUNC:.*]]:
; CHECK-NEXT: adrp x0, dsolocal
; CHECK-NEXT: add x0, x0, :lo12:dsolocal
; CHECK-NEXT: mov x1, #2
; CHECK-NEXT: [[LABEL:.L.*]]:
; CHECK-NEXT: .reloc [[LABEL]], R_AARCH64_PATCHINST, ds
; CHECK-NEXT: b __emupac_pacda
; CHECK-NEXT: ret
; CHECK-NEXT: .section .data.rel.ro
; CHECK-NEXT: .xword [[FUNC]]@FUNCINIT
@dsolocalrefds = constant ptr ptrauth (ptr @dsolocal, i32 2, i64 2, ptr null, ptr @ds), align 8

; CHECK: dsolocalref8:
; CHECK-NEXT: [[PLACE:.*]]:
; CHECK-NEXT: .section .text.startup
; CHECK-NEXT: [[FUNC:.*]]:
; CHECK-NEXT: adrp x0, dsolocal+8
; CHECK-NEXT: add x0, x0, :lo12:dsolocal+8
; CHECK-NEXT: mov x1, #3
; CHECK-NEXT: b __emupac_pacda
; CHECK-NEXT: .section .data.rel.ro
; CHECK-NEXT: .xword [[FUNC]]@FUNCINIT
@dsolocalref8 = constant ptr ptrauth (ptr getelementptr (i8, ptr @dsolocal, i64 8), i32 2, i64 3, ptr null), align 8

; CHECK: disc:
; CHECK-NEXT: [[PLACE:.*]]:
; CHECK-NEXT: .section .text.startup
; CHECK-NEXT: [[FUNC:.*]]:
; CHECK-NEXT: adrp x0, dsolocal
; CHECK-NEXT: add x0, x0, :lo12:dsolocal
; CHECK-NEXT: adrp x1, [[PLACE]]
; CHECK-NEXT: add x1, x1, :lo12:[[PLACE]]
; CHECK-NEXT: b __emupac_pacda
; CHECK-NEXT: .section .data.rel.ro
; CHECK-NEXT: .xword [[FUNC]]@FUNCINIT
@disc = constant ptr ptrauth (ptr @dsolocal, i32 2, i64 0, ptr @disc), align 8

; CHECK: disc65536:
; CHECK-NEXT: [[PLACE:.*]]:
; CHECK-NEXT: .section .text.startup
; CHECK-NEXT: [[FUNC:.*]]:
; CHECK-NEXT: adrp x0, dsolocal
; CHECK-NEXT: add x0, x0, :lo12:dsolocal
; CHECK-NEXT: adrp x1, [[PLACE]]+65536
; CHECK-NEXT: add x1, x1, :lo12:[[PLACE]]+65536
; CHECK-NEXT: b __emupac_pacda
; CHECK-NEXT: .section .data.rel.ro
; CHECK-NEXT: .xword [[FUNC]]@FUNCINIT
@disc65536 = constant ptr ptrauth (ptr @dsolocal, i32 2, i64 65536, ptr @disc), align 8

@global = external global i8

; CHECK: globalref:
; CHECK-NEXT: [[PLACE:.*]]:
; CHECK-NEXT: .section .text.startup
; CHECK-NEXT: [[FUNC:.*]]:
; CHECK-NEXT: adrp x0, :got:global
; CHECK-NEXT: ldr x0, [x0, :got_lo12:global]
; CHECK-NEXT: mov x1, #4
; CHECK-NEXT: b __emupac_pacda
; CHECK-NEXT: .section .data.rel.ro
; CHECK-NEXT: .xword [[FUNC]]@FUNCINIT
@globalref = constant ptr ptrauth (ptr @global, i32 2, i64 4, ptr null), align 8

; CHECK: globalref8:
; CHECK-NEXT: [[PLACE:.*]]:
; CHECK-NEXT: .section .text.startup
; CHECK-NEXT: [[FUNC:.*]]:
; CHECK-NEXT: adrp x0, :got:global
; CHECK-NEXT: ldr x0, [x0, :got_lo12:global]
; CHECK-NEXT: add x0, x0, #8
; CHECK-NEXT: mov x1, #5
; CHECK-NEXT: b __emupac_pacda
; CHECK-NEXT: .section .data.rel.ro
; CHECK-NEXT: .xword [[FUNC]]@FUNCINIT
@globalref8 = constant ptr ptrauth (ptr getelementptr (i8, ptr @global, i64 8), i32 2, i64 5, ptr null), align 8

; CHECK: globalref16777216:
; CHECK-NEXT: [[PLACE:.*]]:
; CHECK-NEXT: .section .text.startup
; CHECK-NEXT: [[FUNC:.*]]:
; CHECK-NEXT: adrp x0, :got:global
; CHECK-NEXT: ldr x0, [x0, :got_lo12:global]
; CHECK-NEXT: mov x16, #0
; CHECK-NEXT: movk x16, #256, lsl #16
; CHECK-NEXT: add x0, x0, x16
; CHECK-NEXT: mov x1, #5
; CHECK-NEXT: b __emupac_pacda
; CHECK-NEXT: .section .data.rel.ro
; CHECK-NEXT: .xword [[FUNC]]@FUNCINIT
@globalref16777216 = constant ptr ptrauth (ptr getelementptr (i8, ptr @global, i64 16777216), i32 2, i64 5, ptr null), align 8

$comdat = comdat any
@comdat = constant ptr ptrauth (ptr null, i32 2, i64 1, ptr null), align 8, comdat
; CHECK: comdat:
; CHECK-NEXT: [[PLACE:.*]]:
; CHECK-NEXT: .section	.text.startup,"axG",@progbits,comdat,comdat

; RUN: llc -mtriple aarch64-elf --relocation-model=static         -mattr=+pauth < %s | FileCheck %s
; RUN: llc -mtriple aarch64-elf --relocation-model=dynamic-no-pic -mattr=+pauth < %s | FileCheck %s

;; A constant value, use .rodata
; CHECK:         .section        .rodata,"a",@progbits
; CHECK:         .globl  Const
; CHECK: Const:
; CHECK:         .xword  37

;; An AUTH reloc is needed, use .data.rel.ro
; CHECK:         .section        .data.rel.ro,"aw",@progbits
; CHECK:         .globl  PtrAuthExtern
; CHECK: PtrAuthExtern:
; CHECK:         .xword  ConstExtern@AUTH(da,0)

; CHECK-NOT:     .section
; CHECK:         .globl  PtrAuth
; CHECK: PtrAuth:
; CHECK:         .xword  Const@AUTH(da,0)

; CHECK-NOT:     .section
; CHECK:         .globl  PtrAuthExternNested1
; CHECK: PtrAuthExternNested1:
; CHECK:         .xword  ConstExtern@AUTH(da,0)

;; The address could be filled statically, use .rodata
; CHECK:         .section        .rodata,"a",@progbits
; CHECK:         .globl  PtrAuthExternNested2
; CHECK: PtrAuthExternNested2:
; CHECK:         .xword  PtrAuthExtern

;; An AUTH reloc is needed, use .data.rel.ro
; CHECK:         .section        .data.rel.ro,"aw",@progbits
; CHECK:         .globl  PtrAuthNested1
; CHECK: PtrAuthNested1:
; CHECK:         .xword  Const@AUTH(da,0)

;; The address could be filled statically, use .rodata
; CHECK:         .section        .rodata,"a",@progbits
; CHECK:         .globl  PtrAuthNested2
; CHECK: PtrAuthNested2:
; CHECK:         .xword  PtrAuth

@ConstExtern = external global i64
@Const       = constant i64 37

@PtrAuthExtern = constant ptr ptrauth (ptr @ConstExtern, i32 2)
@PtrAuth       = constant ptr ptrauth (ptr @Const,  i32 2)

@PtrAuthExternNested1 = constant { ptr } { ptr ptrauth (ptr @ConstExtern, i32 2) }
@PtrAuthExternNested2 = constant { ptr } { ptr @PtrAuthExtern }
@PtrAuthNested1       = constant { ptr } { ptr ptrauth (ptr @Const, i32 2) }
@PtrAuthNested2       = constant { ptr } { ptr @PtrAuth }

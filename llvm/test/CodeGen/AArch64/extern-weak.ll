; RUN: llc -mtriple=aarch64-none-linux-gnu -relocation-model=pic -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -relocation-model=static -o - < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -code-model=large -o - %s | FileCheck --check-prefix=CHECK-LARGE %s
; RUN: llc -mtriple=aarch64-none-elf -code-model=tiny -o - %s | FileCheck --check-prefix=CHECK-TINY %s

declare extern_weak dso_local i32 @var()

define ptr @foo() {
; The usual ADRP/ADD pair can't be used for a weak reference because it must
; evaluate to 0 if the symbol is undefined. We use a GOT entry for PIC
; otherwise a litpool entry.
  ret ptr @var

; CHECK:            adrp x[[ADDRHI:[0-9]+]], :got:var
; CHECK-NEXT:       ldr x0, [x[[ADDRHI]], :got_lo12:var]

  ; In the large model, the usual relocations are absolute and can
  ; materialise 0.
; CHECK-LARGE:      movz x0, #:abs_g0_nc:var
; CHECK-LARGE-NEXT: movk x0, #:abs_g1_nc:var
; CHECK-LARGE-NEXT: movk x0, #:abs_g2_nc:var
; CHECK-LARGE-NEXT: movk x0, #:abs_g3:var

  ; In the tiny codemodel we us a got relocated LDR.
; CHECK-TINY:       ldr x0, :got:var
}


@arr_var = extern_weak global [10 x i32]

define ptr @bar() {
  %addr = getelementptr [10 x i32], ptr @arr_var, i32 0, i32 5

; CHECK:            adrp x[[ADDRHI:[0-9]+]], :got:arr_var
; CHECK-NEXT:       ldr [[BASE:x[0-9]+]], [x[[ADDRHI]], :got_lo12:arr_var]
; CHECK-NEXT:       add x0, [[BASE]], #20

  ret ptr %addr

  ; Note, In the large model, if dso_local, the relocations are absolute and can materialise 0.
; CHECK-LARGE:      adrp x[[ADDR:[0-9]+]], :got:arr_var
; CHECK-LARGE-NEXT: ldr x[[ADDR]], [x[[ADDR]], :got_lo12:arr_var]
; CHECK-LARGE-NEXT: add x0, x[[ADDR]], #20

; CHECK-TINY:       ldr [[BASE:x[0-9]+]], :got:arr_var
; CHECK-TINY-NEXT:  add x0, [[BASE]], #20
}

@defined_weak_var = internal unnamed_addr global i32 0

define ptr @wibble() {
  ret ptr @defined_weak_var

; CHECK:            adrp [[BASE:x[0-9]+]], defined_weak_var
; CHECK-NEXT:       add x0, [[BASE]], :lo12:defined_weak_var

; CHECK-LARGE:      movz x0, #:abs_g0_nc:defined_weak_var
; CHECK-LARGE-NEXT: movk x0, #:abs_g1_nc:defined_weak_var
; CHECK-LARGE-NEXT: movk x0, #:abs_g2_nc:defined_weak_var
; CHECK-LARGE-NEXT: movk x0, #:abs_g3:defined_weak_var

; CHECK-TINY:       adr x0, defined_weak_var
}

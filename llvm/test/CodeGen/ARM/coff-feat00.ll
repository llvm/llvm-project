; RUN: llc -mtriple=thumbv7-windows-msvc -o - %s | FileCheck %s --check-prefix=PLAIN
; RUN: sed -e s/.cfg:// %s | llc -mtriple=thumbv7-windows-msvc -o - | FileCheck %s --check-prefix=CFG
; RUN: sed -e s/.cfg:// %s | llc -mtriple=thumbv7-windows-msvc -filetype=obj -o %t.obj
; RUN: llvm-readobj --symbols %t.obj | FileCheck %s --check-prefix=OBJ

; ARM COFF objects carry @feat.00. The linker reads the /guard:cf tables only
; when it has the /guard:cf bit (0x800).

declare dllimport i32 @imported()

define ptr @take_address() {
  ret ptr @imported
}

; ARM assembly quotes the name, since @ is not an identifier character here.
; PLAIN: "@feat.00" = 0
; CFG: "@feat.00" = 2048
; CFG: .section .giats$y,"dr"
; CFG-NEXT: .symidx __imp_imported

; OBJ:      Name: @feat.00
; OBJ-NEXT: Value: 2048
; OBJ-NEXT: Section: IMAGE_SYM_ABSOLUTE

!0 = !{i32 2, !"cfguard", i32 2}
;cfg: !llvm.module.flags = !{!0}

; RUN: llc -mtriple aarch64-linux-pauthtest -mattr +pauth -filetype=asm %s -o - | \
; RUN:   FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple aarch64-linux-pauthtest -mattr +pauth -filetype=obj %s -o - | \
; RUN:   llvm-readelf -s - | FileCheck %s --check-prefix=OBJ

; ASM:               .type   foo,@function
; ASM-LABEL: foo:
; ASM:               adrp    x17, :got_auth:bar
; ASM-NEXT:          add     x17, x17, :got_auth_lo12:bar
; ASM-NEXT:          ldr     x16, [x17]
; ASM-NEXT:          autia   x16, x17
; ASM-NEXT:          mov     x17, x16
; ASM-NEXT:          xpaci   x17
; ASM-NEXT:          cmp     x16, x17
; ASM-NEXT:          b.eq    .Lauth_success_0
; ASM-NEXT:          brk     #0xc470
; ASM-NEXT:  .Lauth_success_0:
; ASM-NEXT:          paciza  x16
; ASM-NEXT:          adrp    x8, .Lfptr
; ASM-NEXT:          str     x16, [x8, :lo12:.Lfptr]
; ASM-NEXT:          ret
; ASM:               .type   .Lfptr,@object
; ASM-NEXT:          .local  .Lfptr
; ASM-NEXT:          .comm   .Lfptr,8,8
; ASM:               .type   bar,@function

; OBJ:      Symbol table '.symtab' contains [[#]] entries:
; OBJ-NEXT:    Num:    Value          Size Type    Bind   Vis       Ndx Name
; OBJ:              0000000000000000     0 FUNC    GLOBAL DEFAULT   UND bar

@fptr = private global ptr null

define void @foo() {
  store ptr ptrauth (ptr @bar, i32 0), ptr @fptr
  ret void
}

declare i32 @bar()

!llvm.module.flags = !{!0}

!0 = !{i32 8, !"ptrauth-elf-got", i32 1}

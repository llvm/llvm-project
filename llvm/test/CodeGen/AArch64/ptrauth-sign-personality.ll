; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: cat common.ll authflag.ll   > auth.ll
; RUN: cat common.ll noauthflag.ll > noauth1.ll
; RUN: cat common.ll               > noauth2.ll

; RUN: llc -mtriple=aarch64-linux -filetype=asm auth.ll -o - | \
; RUN:   FileCheck --check-prefix=AUTH-ASM %s
; RUN: llc -mtriple=aarch64-linux -filetype=obj auth.ll -o - | \
; RUN:   llvm-readelf -r -x .data.DW.ref.__gxx_personality_v0 - | \
; RUN:   FileCheck --check-prefix=AUTH-RELOC %s

; AUTH-ASM:      DW.ref.__gxx_personality_v0:
; AUTH-ASM-NEXT:     .xword  __gxx_personality_v0@AUTH(ia,32429,addr)

; AUTH-RELOC:      Relocation section '.rela.data.DW.ref.__gxx_personality_v0' at offset 0x2a0 contains 1 entries:
; AUTH-RELOC-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
; AUTH-RELOC-NEXT: 0000000000000000  0000000f00000244 R_AARCH64_AUTH_ABS64   0000000000000000 __gxx_personality_v0 + 0

; AUTH-RELOC:      Hex dump of section '.data.DW.ref.__gxx_personality_v0':
; AUTH-RELOC-NEXT: 0x00000000 00000000 ad7e0080
;                                      ^^^^ 0x7EAD = discriminator
;                                            ^^ 0b10000000: bit 63 = 1 -> address diversity enabled, bits 61:60 = 0b00 -> key is IA

; RUN: llc -mtriple=aarch64-linux -filetype=asm noauth1.ll -o - | \
; RUN:   FileCheck --check-prefix=NOAUTH-ASM %s
; RUN: llc -mtriple=aarch64-linux -filetype=obj noauth1.ll -o - | \
; RUN:   llvm-readelf -r -x .data.DW.ref.__gxx_personality_v0 - | \
; RUN:   FileCheck --check-prefix=NOAUTH-RELOC %s

; RUN: llc -mtriple=aarch64-linux -filetype=asm noauth2.ll -o - | \
; RUN:   FileCheck --check-prefix=NOAUTH-ASM %s
; RUN: llc -mtriple=aarch64-linux -filetype=obj noauth2.ll -o - | \
; RUN:   llvm-readelf -r -x .data.DW.ref.__gxx_personality_v0 - | \
; RUN:   FileCheck --check-prefix=NOAUTH-RELOC %s

; NOAUTH-ASM:      DW.ref.__gxx_personality_v0:
; NOAUTH-ASM-NEXT:     .xword  __gxx_personality_v0{{$}}

; NOAUTH-RELOC:      Relocation section '.rela.data.DW.ref.__gxx_personality_v0' at offset 0x2a0 contains 1 entries:
; NOAUTH-RELOC-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
; NOAUTH-RELOC-NEXT: 0000000000000000  0000000f00000101 R_AARCH64_ABS64   0000000000000000 __gxx_personality_v0 + 0

; NOAUTH-RELOC:      Hex dump of section '.data.DW.ref.__gxx_personality_v0':
; NOAUTH-RELOC-NEXT: 0x00000000 00000000 00000000

;--- common.ll
@_ZTISt9exception = external constant ptr

define i32 @main() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @foo() to label %cont unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 }
    catch ptr null
    catch ptr @_ZTISt9exception
  ret i32 0

cont:
  ret i32 0
}

declare i32 @__gxx_personality_v0(...)

declare void @foo()

;--- authflag.ll
!llvm.module.flags = !{!0}
!0 = !{i32 8, !"ptrauth-sign-personality", i32 1}

;--- noauthflag.ll
!llvm.module.flags = !{!0}
!0 = !{i32 8, !"ptrauth-sign-personality", i32 0}

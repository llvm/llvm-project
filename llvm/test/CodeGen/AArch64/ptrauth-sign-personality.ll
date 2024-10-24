; RUN: llc -mtriple=aarch64-linux -filetype=asm %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-linux -filetype=obj %s -o - | \
; RUN:   llvm-readelf -r -x .data.DW.ref.__gxx_personality_v0 - | \
; RUN:   FileCheck --check-prefix=RELOC %s

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

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"ptrauth-sign-personality", i32 1}

; CHECK:      DW.ref.__gxx_personality_v0:
; CHECK-NEXT:     .xword  __gxx_personality_v0@AUTH(ia,32429,addr)

; RELOC:      Relocation section '.rela.data.DW.ref.__gxx_personality_v0' at offset 0x2a0 contains 1 entries:
; RELOC-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
; RELOC-NEXT: 0000000000000000  0000000f00000244 R_AARCH64_AUTH_ABS64   0000000000000000 __gxx_personality_v0 + 0

; RELOC:      Hex dump of section '.data.DW.ref.__gxx_personality_v0':
; RELOC-NEXT: 0x00000000 00000000 ad7e0080
;                                 ^^^^ 0x7EAD = discriminator
;                                       ^^ 0b10000000: bit 63 = 1 -> address diversity enabled, bits 61:60 = 0b00 -> key is IA

; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   --relocation-model=pic -mcpu=pwr10 -ppc-asm-full-reg-names < %s | FileCheck %s --check-prefix=CHECK-S
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   --relocation-model=pic -mcpu=pwr10 -ppc-asm-full-reg-names --filetype=obj -o %t.o < %s
; RUN: llvm-objdump --mcpu=pwr10 -dr %t.o |FileCheck %s --check-prefix=CHECK-O
; RUN: llvm-readelf -s %t.o | FileCheck %s --check-prefix=CHECK-SYM

; These test cases are to ensure that when using pc relative memory operations
; ABI correct code and relocations are produced for General Dynamic TLS Model.

@x = external thread_local global i32, align 4

define nonnull i32* @GeneralDynamicAddressLoad() {
  ; CHECK-S-LABEL: GeneralDynamicAddressLoad:
  ; CHECK-S:         paddi r3, 0, x@got@tlsgd@pcrel, 1
  ; CHECK-S-NEXT:    bl __tls_get_addr@notoc(x@tlsgd)
  ; CHECK-S-NEXT:    addi r1, r1, 32
  ; CHECK-S-NEXT:    ld r0, 16(r1)
  ; CHECK-S-NEXT:    mtlr r0
  ; CHECK-S-NEXT:    blr
  ; CHECK-O-LABEL: <GeneralDynamicAddressLoad>:
  ; CHECK-O:         c: 00 00 10 06 00 00 60 38       paddi 3, 0, 0, 1
  ; CHECK-O-NEXT:    000000000000000c:  R_PPC64_GOT_TLSGD_PCREL34    x
  ; CHECK-O-NEXT:    14: 01 00 00 48                   bl 0x14
  ; CHECK-O-NEXT:    0000000000000014:  R_PPC64_TLSGD        x
  ; CHECK-O-NEXT:    0000000000000014:  R_PPC64_REL24_NOTOC  __tls_get_addr
  entry:
    ret i32* @x
}

define i32 @GeneralDynamicValueLoad() {
  ; CHECK-S-LABEL: GeneralDynamicValueLoad:
  ; CHECK-S:         paddi r3, 0, x@got@tlsgd@pcrel, 1
  ; CHECK-S-NEXT:    bl __tls_get_addr@notoc(x@tlsgd)
  ; CHECK-S-NEXT:    lwz r3, 0(r3)
  ; CHECK-S-NEXT:    addi r1, r1, 32
  ; CHECK-S-NEXT:    ld r0, 16(r1)
  ; CHECK-S-NEXT:    mtlr r0
  ; CHECK-S-NEXT:    blr
  ; CHECK-O-LABEL: <GeneralDynamicValueLoad>:
  ; CHECK-O:         4c: 00 00 10 06 00 00 60 38       paddi 3, 0, 0, 1
  ; CHECK-O-NEXT:    000000000000004c:  R_PPC64_GOT_TLSGD_PCREL34    x
  ; CHECK-O-NEXT:    54: 01 00 00 48                   bl 0x54
  ; CHECK-O-NEXT:    0000000000000054:  R_PPC64_TLSGD        x
  ; CHECK-O-NEXT:    0000000000000054:  R_PPC64_REL24_NOTOC  __tls_get_addr
  ; CHECK-O-NEXT:    58: 00 00 63 80                   lwz 3, 0(3)

  ; CHECK-SYM-LABEL: Symbol table '.symtab' contains 7 entries
  ; CHECK-SYM:       0000000000000000     0 TLS     GLOBAL DEFAULT  UND x
  entry:
    %0 = load i32, i32* @x, align 4
    ret i32 %0
}

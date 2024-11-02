; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr10 \
; RUN:   -ppc-asm-full-reg-names --relocation-model=pic < %s | FileCheck %s --check-prefix=CHECK-S
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr10 \
; RUN:   -ppc-asm-full-reg-names --relocation-model=pic --filetype=obj < %s | \
; RUN:   llvm-objdump --mcpu=pwr10 --no-show-raw-insn -dr - | FileCheck %s --check-prefix=CHECK-O

; These test cases are to ensure that when using pc relative memory operations
; ABI correct code and relocations are produced for Local Dynamic TLS Model.

@x = hidden thread_local global i32 0, align 4

define nonnull ptr @LocalDynamicAddressLoad() {
  ; CHECK-S-LABEL: LocalDynamicAddressLoad:
  ; CHECK-S:         paddi r3, 0, x@got@tlsld@pcrel, 1
  ; CHECK-S-NEXT:    bl __tls_get_addr@notoc(x@tlsld)
  ; CHECK-S-NEXT:    paddi r3, r3, x@DTPREL, 0
  ; CHECK-S-NEXT:    addi r1, r1, 32
  ; CHECK-S-NEXT:    ld r0, 16(r1)
  ; CHECK-S-NEXT:    mtlr r0
  ; CHECK-S-NEXT:    blr
  ; CHECK-O-LABEL: <LocalDynamicAddressLoad>:
  ; CHECK-O:         c: paddi 3, 0, 0, 1
  ; CHECK-O-NEXT:    000000000000000c: R_PPC64_GOT_TLSLD_PCREL34 x
  ; CHECK-O-NEXT:    14: bl 0x14
  ; CHECK-O-NEXT:    0000000000000014: R_PPC64_TLSLD x
  ; CHECK-O-NEXT:    0000000000000014: R_PPC64_REL24_NOTOC __tls_get_addr
  ; CHECK-O-NEXT:    18: paddi 3, 3, 0, 0
  ; CHECK-O-NEXT:    0000000000000018: R_PPC64_DTPREL34 x
  entry:
    ret ptr @x
}

define i32 @LocalDynamicValueLoad() {
  ; CHECK-S-LABEL: LocalDynamicValueLoad:
  ; CHECK-S:         paddi r3, 0, x@got@tlsld@pcrel, 1
  ; CHECK-S-NEXT:    bl __tls_get_addr@notoc(x@tlsld)
  ; CHECK-S-NEXT:    paddi r3, r3, x@DTPREL, 0
  ; CHECK-S-NEXT:    lwz r3, 0(r3)
  ; CHECK-S-NEXT:    addi r1, r1, 32
  ; CHECK-S-NEXT:    ld r0, 16(r1)
  ; CHECK-S-NEXT:    mtlr r0
  ; CHECK-S-NEXT:    blr
  ; CHECK-O-LABEL: <LocalDynamicValueLoad>:
  ; CHECK-O:         4c: paddi 3, 0, 0, 1
  ; CHECK-O-NEXT:    000000000000004c: R_PPC64_GOT_TLSLD_PCREL34 x
  ; CHECK-O-NEXT:    54: bl 0x54
  ; CHECK-O-NEXT:    0000000000000054: R_PPC64_TLSLD x
  ; CHECK-O-NEXT:    0000000000000054: R_PPC64_REL24_NOTOC __tls_get_addr
  ; CHECK-O-NEXT:    58: paddi 3, 3, 0, 0
  ; CHECK-O-NEXT:    0000000000000058: R_PPC64_DTPREL34 x
  ; CHECK-O-NEXT:    60: lwz 3, 0(3)
  entry:
    %0 = load i32, ptr @x, align 4
    ret i32 %0
}

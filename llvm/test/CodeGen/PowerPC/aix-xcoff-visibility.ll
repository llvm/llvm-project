; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 -mattr=-altivec -data-sections=false < %s | \
; RUN:   FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 -mattr=-altivec -data-sections=false < %s |\
; RUN:   FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=ppc -filetype=obj %s -o %t.o
; RUN: llvm-readobj --syms --auxiliary-header %t.o | FileCheck %s --check-prefixes=SYM,AUX32

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=ppc -filetype=obj %s -o %t64.o
; RUN: llvm-readobj --syms --auxiliary-header %t64.o | FileCheck %s --check-prefixes=SYM,AUX64

@b =  global i32 0, align 4
@b_h = hidden global i32 0, align 4
@b_e = dllexport global i32 0, align 4

define void @foo() {
entry:
  ret void
}

define hidden void @foo_h(ptr %ip) {
entry:
  ret void
}

define dllexport void @foo_e(ptr %ip) {
entry:
  ret void
}

define protected void @foo_protected(ptr %ip) {
entry:
  ret void
}

define weak hidden void @foo_weak_h() {
entry:
  ret void
}

define weak dllexport void @foo_weak_e() {
entry:
  ret void
}

@foo_p = global ptr @zoo_weak_extern_h, align 4
declare extern_weak hidden void @zoo_weak_extern_h()
declare extern_weak dllexport void @zoo_weak_extern_e()

define i32 @main() {
entry:
  %call1= call i32 @bar_h(ptr @b_h)
  call void @foo_weak_h()
  %0 = load ptr, ptr @foo_p, align 4
  call void %0()
  ret i32 0
}

declare hidden i32 @bar_h(ptr)
declare dllexport i32 @bar_e(ptr)

; CHECK:        .globl  foo[DS]{{[[:space:]]*([#].*)?$}}
; CHECK:        .globl  .foo{{[[:space:]]*([#].*)?$}}
; CHECK:        .globl  foo_h[DS],hidden
; CHECK:        .globl  .foo_h,hidden
; CHECK:        .globl  foo_e[DS],exported
; CHECK:        .globl  .foo_e,exported
; CHECK:        .globl  foo_protected[DS],protected
; CHECK:        .globl  .foo_protected,protected
; CHECK:        .weak   foo_weak_h[DS],hidden
; CHECK:        .weak   .foo_weak_h,hidden
; CHECK:        .weak   foo_weak_e[DS],exported
; CHECK:        .weak   .foo_weak_e,exported

; CHECK:        .globl  b{{[[:space:]]*([#].*)?$}}
; CHECK:        .globl  b_h,hidden
; CHECK:        .globl  b_e,exported

; CHECK:        .weak   .zoo_weak_extern_h[PR],hidden
; CHECK:        .weak   zoo_weak_extern_h[DS],hidden
; CHECK:        .weak   .zoo_weak_extern_e[PR],exported
; CHECK:        .extern .bar_h[PR],hidden
; CHECK:        .extern .bar_e[PR],exported

; AUX32:       AuxiliaryHeader {
; AUX32-NEXT:    Magic: 0x0
; AUX32-NEXT:    Version: 0x2
; AUX32-NEXT:    Size of .text section: 0x134
; AUX32-NEXT:    Size of .data section: 0x6C
; AUX32-NEXT:    Size of .bss section: 0x0
; AUX32-NEXT:    Entry point address: 0x0
; AUX32-NEXT:    .text section start address: 0x0
; AUX32-NEXT:    .data section start address: 0x134
; AUX32-NEXT:  }

; AUX64:       AuxiliaryHeader {
; AUX64-NEXT:  }

; SYM:         Name: .bar_h
; SYM-NEXT:    Value (RelocatableAddress): 0x0
; SYM-NEXT:    Section: N_UNDEF
; SYM-NEXT:    Type: 0x2000
; SYM-NEXT:    StorageClass: C_EXT (0x2)

; SYM:         Name: zoo_weak_extern_h
; SYM-NEXT:    Value (RelocatableAddress): 0x0
; SYM-NEXT:    Section: N_UNDEF
; SYM-NEXT:    Type: 0x2000
; SYM-NEXT:    StorageClass: C_WEAKEXT (0x6F)

; SYM:         Name: .zoo_weak_extern_h
; SYM-NEXT:    Value (RelocatableAddress): 0x0
; SYM-NEXT:    Section: N_UNDEF
; SYM-NEXT:    Type: 0x2000
; SYM-NEXT:    StorageClass: C_WEAKEXT (0x6F)

; SYM:         Name: .zoo_weak_extern_e
; SYM-NEXT:    Value (RelocatableAddress): 0x0
; SYM-NEXT:    Section: N_UNDEF
; SYM-NEXT:    Type: 0x4000
; SYM-NEXT:    StorageClass: C_WEAKEXT (0x6F)

; SYM:         Name: .bar_e
; SYM-NEXT:    Value (RelocatableAddress): 0x0
; SYM-NEXT:    Section: N_UNDEF
; SYM-NEXT:    Type: 0x4000
; SYM-NEXT:    StorageClass: C_EXT (0x2)

; SYM:         Name: .foo
; SYM-NEXT:    Value (RelocatableAddress): 0x0
; SYM-NEXT:    Section: .text
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_EXT (0x2)

; SYM:         Name: .foo_h
; SYM-NEXT:    Value (RelocatableAddress): 0x1C
; SYM-NEXT:    Section: .text
; SYM-NEXT:    Type: 0x2000
; SYM-NEXT:    StorageClass: C_EXT (0x2)

; SYM:         Name: .foo_e
; SYM-NEXT:    Value (RelocatableAddress): 0x3C
; SYM-NEXT:    Section: .text
; SYM-NEXT:    Type: 0x4000
; SYM-NEXT:    StorageClass: C_EXT (0x2)

; SYM:         Name: .foo_protected
; SYM-NEXT:    Value (RelocatableAddress): 0x5C
; SYM-NEXT:    Section: .text
; SYM-NEXT:    Type: 0x3000
; SYM-NEXT:    StorageClass: C_EXT (0x2)

; SYM:         Name: .foo_weak_h
; SYM-NEXT:    Value (RelocatableAddress): 0x84
; SYM-NEXT:    Section: .text
; SYM-NEXT:    Type: 0x2000
; SYM-NEXT:    StorageClass: C_WEAKEXT (0x6F)

; SYM:         Name: .foo_weak_e
; SYM-NEXT:    Value (RelocatableAddress): 0xA4
; SYM-NEXT:    Section: .text
; SYM-NEXT:    Type: 0x4000
; SYM-NEXT:    StorageClass: C_WEAKEXT (0x6F)

; SYM:         Name: b
; SYM-NEXT:    Value (RelocatableAddress): 0x134
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_EXT (0x2)

; SYM:         Name: b_h
; SYM-NEXT:    Value (RelocatableAddress): 0x138
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x2000
; SYM-NEXT:    StorageClass: C_EXT (0x2)

; SYM:         Name: b_e
; SYM-NEXT:    Value (RelocatableAddress): 0x13C
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x4000
; SYM-NEXT:    StorageClass: C_EXT (0x2)

; SYM:         Name: foo
; SYM-NEXT:    Value (RelocatableAddress): {{[0-9]+}}
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_EXT (0x2)

; SYM:         Name: foo_h
; SYM-NEXT:    Value (RelocatableAddress): {{[0-9]+}}
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x2000
; SYM-NEXT:    StorageClass: C_EXT (0x2)

; SYM:         Name: foo_e
; SYM-NEXT:    Value (RelocatableAddress): {{[0-9]+}}
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x4000
; SYM-NEXT:    StorageClass: C_EXT (0x2)

; SYM:         Name: foo_protected
; SYM-NEXT:    Value (RelocatableAddress): {{[0-9]+}}
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x3000
; SYM-NEXT:    StorageClass: C_EXT (0x2)

; SYM:         Name: foo_weak_h
; SYM-NEXT:    Value (RelocatableAddress): {{[0-9]+}}
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x2000
; SYM-NEXT:    StorageClass: C_WEAKEXT (0x6F)

; SYM:         Name: foo_weak_e
; SYM-NEXT:    Value (RelocatableAddress): {{[0-9]+}}
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x4000
; SYM-NEXT:    StorageClass: C_WEAKEXT (0x6F)

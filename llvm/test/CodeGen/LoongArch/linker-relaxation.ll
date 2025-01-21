; RUN: llc --mtriple=loongarch64 --filetype=obj -mattr=-relax \
; RUN:     --relocation-model=pic --code-model=medium < %s \
; RUN:     | llvm-readobj -r - | FileCheck --check-prefixes=CHECK-RELOC,PCALA-RELOC %s
; RUN: llc --mtriple=loongarch64 --filetype=obj -mattr=+relax \
; RUN:     --relocation-model=pic --code-model=medium < %s \
; RUN:     | llvm-readobj -r - | FileCheck --check-prefixes=CHECK-RELOC,RELAX %s

; RUN: llc --mtriple=loongarch64 --filetype=obj -mattr=-relax --enable-tlsdesc \
; RUN:     --relocation-model=pic --code-model=medium < %s \
; RUN:     | llvm-readobj -r - | FileCheck --check-prefix=DESC-RELOC %s
; RUN: llc --mtriple=loongarch64 --filetype=obj -mattr=+relax --enable-tlsdesc \
; RUN:     --relocation-model=pic --code-model=medium < %s \
; RUN:     | llvm-readobj -r - | FileCheck --check-prefixes=DESC-RELOC,DESC-RELAX %s

;; Check relocations when disable or enable linker relaxation.
;; This tests are also able to test for removing relax mask flags
;; after loongarch-merge-base-offset pass because no relax relocs
;; are emitted after being optimized by it.

@g_e = external global i32
@g_i = internal global i32 0
@g_i1 = internal global i32 1
@t_un = external thread_local global i32
@t_ld = external thread_local(localdynamic) global i32
@t_ie = external thread_local(initialexec) global i32
@t_le = external thread_local(localexec) global i32

declare void @callee1() nounwind
declare dso_local void @callee2() nounwind
declare dso_local void @callee3() nounwind

define ptr @caller() nounwind {
; RELAX:            R_LARCH_ALIGN - 0x1C
; CHECK-RELOC:      R_LARCH_GOT_PC_HI20 g_e 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; CHECK-RELOC-NEXT: R_LARCH_GOT_PC_LO12 g_e 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; PCALA-RELOC:      R_LARCH_PCALA_HI20 .bss 0x0
; RELAX-NEXT:       R_LARCH_PCALA_HI20 g_i 0x0
; PCALA-RELOC:      R_LARCH_PCALA_LO12 .bss 0x0
; RELAX-NEXT:       R_LARCH_PCALA_LO12 g_i 0x0
; CHECK-RELOC:      R_LARCH_TLS_GD_PC_HI20 t_un 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; CHECK-RELOC-NEXT: R_LARCH_GOT_PC_LO12 t_un 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; CHECK-RELOC-NEXT: R_LARCH_CALL36 __tls_get_addr 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; DESC-RELOC:       R_LARCH_TLS_DESC_PC_HI20 t_un 0x0
; DESC-RELAX:       R_LARCH_RELAX - 0x0
; DESC-RELOC-NEXT:  R_LARCH_TLS_DESC_PC_LO12 t_un 0x0
; DESC-RELAX-NEXT:  R_LARCH_RELAX - 0x0
; DESC-RELOC-NEXT:  R_LARCH_TLS_DESC_LD t_un 0x0
; DESC-RELAX-NEXT:  R_LARCH_RELAX - 0x0
; DESC-RELOC-NEXT:  R_LARCH_TLS_DESC_CALL t_un 0x0
; DESC-RELAX-NEXT:  R_LARCH_RELAX - 0x0
; CHECK-RELOC-NEXT: R_LARCH_TLS_LD_PC_HI20 t_ld 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; CHECK-RELOC-NEXT: R_LARCH_GOT_PC_LO12 t_ld 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; CHECK-RELOC-NEXT: R_LARCH_CALL36 __tls_get_addr 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; DESC-RELOC-NEXT:  R_LARCH_TLS_DESC_PC_HI20 t_ld 0x0
; DESC-RELAX-NEXT:  R_LARCH_RELAX - 0x0
; DESC-RELOC-NEXT:  R_LARCH_TLS_DESC_PC_LO12 t_ld 0x0
; DESC-RELAX-NEXT:  R_LARCH_RELAX - 0x0
; DESC-RELOC-NEXT:  R_LARCH_TLS_DESC_LD t_ld 0x0
; DESC-RELAX-NEXT:  R_LARCH_RELAX - 0x0
; DESC-RELOC-NEXT:  R_LARCH_TLS_DESC_CALL t_ld 0x0
; DESC-RELAX-NEXT:  R_LARCH_RELAX - 0x0
; CHECK-RELOC-NEXT: R_LARCH_TLS_IE_PC_HI20 t_ie 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; CHECK-RELOC-NEXT: R_LARCH_TLS_IE_PC_LO12 t_ie 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; CHECK-RELOC-NEXT: R_LARCH_TLS_LE_HI20_R t_le 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; CHECK-RELOC-NEXT: R_LARCH_TLS_LE_ADD_R t_le 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; CHECK-RELOC-NEXT: R_LARCH_TLS_LE_LO12_R t_le 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; CHECK-RELOC-NEXT: R_LARCH_CALL36 callee1 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; CHECK-RELOC-NEXT: R_LARCH_CALL36 callee2 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; CHECK-RELOC-NEXT: R_LARCH_CALL36 callee3 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; PCALA-RELOC:      R_LARCH_PCALA_HI20 .data 0x0
; RELAX-NEXT:       R_LARCH_PCALA_HI20 g_i1 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
; PCALA-RELOC:      R_LARCH_PCALA_LO12 .data 0x0
; RELAX-NEXT:       R_LARCH_PCALA_LO12 g_i1 0x0
; RELAX-NEXT:       R_LARCH_RELAX - 0x0
  %a = load volatile i32, ptr @g_e
  %b = load volatile i32, ptr @g_i
  %c = load volatile i32, ptr @t_un
  %d = load volatile i32, ptr @t_ld
  %e = load volatile i32, ptr @t_ie
  %f = load volatile i32, ptr @t_le
  call i32 @callee1()
  call i32 @callee2()
  tail call i32 @callee3()
  ret ptr @g_i1
}

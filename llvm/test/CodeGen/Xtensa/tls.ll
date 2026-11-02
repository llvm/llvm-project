; RUN: llc -mtriple=xtensa -function-sections --mcpu=esp32 --filetype=obj < %s \
; RUN: | llvm-objdump -r -s --triple=xtensa --mcpu=esp32 - | FileCheck -check-prefix=XTENSA-CHECK-OBJ %s
; RUN: llc -mtriple=xtensa -function-sections --mcpu=esp32 --filetype=asm < %s | FileCheck -check-prefix=XTENSA-CHECK-ASM %s

; XTENSA-CHECK-OBJ-LABEL: RELOCATION RECORDS FOR [.literal.get_tls]:
; XTENSA-CHECK-OBJ:       OFFSET   TYPE                     VALUE
; XTENSA-CHECK-OBJ-NEXT:  00000000 R_XTENSA_TLS_TPOFF       tls_var

; XTENSA-CHECK-ASM-LABEL: .literal_position
; XTENSA-CHECK-ASM:       .literal .LCPI0_0, tls_var@TPOFF
; XTENSA-CHECK-ASM-LABEL: get_tls:
; XTENSA-CHECK-ASM:       .cfi_startproc
; XTENSA-CHECK-ASM-NEXT:  # %bb.0:
; XTENSA-CHECK-ASM-NEXT:  entry	a1, 32
; XTENSA-CHECK-ASM-NEXT:  .cfi_def_cfa_offset 32
; XTENSA-CHECK-ASM-NEXT:  l32r	a8, .LCPI0_0
; XTENSA-CHECK-ASM-NEXT:  rur	a9, threadptr
; XTENSA-CHECK-ASM-NEXT:  add	a8, a9, a8
; XTENSA-CHECK-ASM-NEXT:  l32i	a2, a8, 0
; XTENSA-CHECK-ASM-NEXT:  retw.n


@tls_var = dso_local thread_local local_unnamed_addr global i32 42, align 4

define dso_local i32 @get_tls() {
entry:
  %v = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @tls_var)
  %res = load i32, ptr %v, align 4
  ret i32 %res
}

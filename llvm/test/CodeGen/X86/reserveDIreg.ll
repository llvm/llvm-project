;; Check if manually reserved EDI is always excluded from being saved by the
;; function prolog/epilog, as per GCC behavior, and that REP MOVS/STOS are not
;; selected when EDI is reserved on x86-32.

; RUN: llc < %s -mtriple=i386-unknown-linux-gnu -verify-machineinstrs | FileCheck %s

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1 immarg)
declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1 immarg)

define void @tedi() "target-features"="+reserve-edi" {
; CHECK-LABEL: tedi:
; CHECK: # %bb.0:
; CHECK-NEXT:        movl    $256, %edi
; CHECK-NEXT:        #APP
; CHECK-NEXT:        #NO_APP
; CHECK-NEXT:        retl
  call i32 asm sideeffect "", "={edi},{edi}"(i32 256)
  ret void
}

define void @no_reserve_edi() {
; CHECK-LABEL: no_reserve_edi:
; CHECK: # %bb.0:
; CHECK-NEXT:        pushl %edi
; CHECK-NEXT:        .cfi_def_cfa_offset 8
; CHECK-NEXT:        .cfi_offset %edi, -8
; CHECK-NEXT:        movl    $256, %edi
; CHECK-NEXT:        #APP
; CHECK-NEXT:        #NO_APP
; CHECK-NEXT:        popl %edi
; CHECK-NEXT:        .cfi_def_cfa_offset 4
; CHECK-NEXT:        retl
  call i32 asm sideeffect "", "={edi},{edi}"(i32 256)
  ret void
}

define void @copy_reserved_edi(ptr %dst, ptr %src) minsize "target-features"="+reserve-edi" {
; CHECK-LABEL: copy_reserved_edi:
; CHECK-NOT:        rep;movs
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %dst, ptr align 4 %src, i32 128, i1 false)
  ret void
}

define void @set_reserved_edi(ptr %dst) minsize "target-features"="+reserve-edi" {
; CHECK-LABEL: set_reserved_edi:
; CHECK-NOT:        rep;stos
  call void @llvm.memset.p0.i32(ptr align 4 %dst, i8 0, i32 128, i1 false)
  ret void
}

define void @copy_no_reserved(ptr %dst, ptr %src) minsize {
; CHECK-LABEL: copy_no_reserved:
; CHECK:       rep;movs
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %dst, ptr align 4 %src, i32 128, i1 false)
  ret void
}

define void @set_no_reserved(ptr %dst) minsize {
; CHECK-LABEL: set_no_reserved:
; CHECK:       rep;stos
  call void @llvm.memset.p0.i32(ptr align 4 %dst, i8 0, i32 128, i1 false)
  ret void
}
; RUN: llc < %s -mtriple=i386-unknown-linux-gnu | FileCheck %s

declare i32 @foo()

@init_array1 = global ptr @foo, section ".init_array"
@init_array2 = global ptr @foo, section ".init_array.00001"
@init_array3 = global ptr @foo, section ".init_arrayfoo"

; CHECK-LABEL: .section  .init_array,"aw",@init_array
; CHECK: init_array1:
; CHECK-LABEL: .section  .init_array.00001,"aw",@init_array
; CHECK: init_array2:
; CHECK-LABEL: .section  .init_arrayfoo,"aw",@progbits
; CHECK: init_array3:

@fini_array1 = global ptr @foo, section ".fini_array"
@fini_array2 = global ptr @foo, section ".fini_array.00001"
@fini_array3 = global ptr @foo, section ".fini_arrayfoo"

; CHECK-LABEL: .section  .fini_array,"aw",@fini_array
; CHECK: fini_array1:
; CHECK-LABEL: .section  .fini_array.00001,"aw",@fini_array
; CHECK: fini_array2:
; CHECK-LABEL: .section  .fini_arrayfoo,"aw",@progbits
; CHECK: fini_array3:

@preinit_array1 = global ptr @foo, section ".preinit_array"
@preinit_array2 = global ptr @foo, section ".preinit_array.00001"
@preinit_array3 = global ptr @foo, section ".preinit_arrayfoo"

; CHECK-LABEL: .section  .preinit_array,"aw",@preinit_array
; CHECK: preinit_array1:
; CHECK-LABEL: .section  .preinit_array.00001,"aw",@preinit_array
; CHECK: preinit_array2:
; CHECK-LABEL: .section  .preinit_arrayfoo,"aw",@progbits
; CHECK: preinit_array3:

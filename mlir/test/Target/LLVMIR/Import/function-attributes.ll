; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK: llvm.func internal @func_internal
define internal void @func_internal() {
  ret void
}

; CHECK: llvm.func internal spir_funccc @spir_func_internal()
define internal spir_func void @spir_func_internal() {
  ret void
}

; // -----

; CHECK-LABEL: @func_readnone
; CHECK-SAME:  attributes {llvm.readnone}
; CHECK:   llvm.return
define void @func_readnone() readnone {
  ret void
}

; CHECK-LABEL: @func_readnone_indirect
; CHECK-SAME:  attributes {llvm.readnone}
declare void @func_readnone_indirect() #0
attributes #0 = { readnone }

; // -----

; CHECK-LABEL: @func_arg_attrs
; CHECK-SAME:  !llvm.ptr {llvm.byval = i64}
; CHECK-SAME:  !llvm.ptr {llvm.byref = i64}
; CHECK-SAME:  !llvm.ptr {llvm.sret = i64}
; CHECK-SAME:  !llvm.ptr {llvm.inalloca = i64}
define void @func_arg_attrs(
    ptr byval(i64) %arg0,
    ptr byref(i64) %arg1,
    ptr sret(i64) %arg2,
    ptr inalloca(i64) %arg3) {
  ret void
}

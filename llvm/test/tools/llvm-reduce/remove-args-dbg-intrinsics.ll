; llvm-reduce shouldn't remove arguments of debug intrinsics, because the resulting module will be ill-formed.
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=arguments --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s --output %t
; RUN: FileCheck --check-prefix=CHECK-FINAL %s < %t

; CHECK-INTERESTINGNESS: declare void @llvm.dbg.addr
; CHECK-FINAL: declare void @llvm.dbg.addr(metadata, metadata, metadata)
declare void @llvm.dbg.addr(metadata, metadata, metadata)
; CHECK-INTERESTINGNESS: declare void @llvm.dbg.declare
; CHECK-FINAL: declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.declare(metadata, metadata, metadata)
; CHECK-INTERESTINGNESS: declare void @llvm.dbg.value
; CHECK-FINAL: declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)

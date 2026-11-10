; RUN: llc < %s -mtriple=s390x-linux-gnu -argext-abi-check | FileCheck %s
; REQUIRES: asserts
;
; With the "trap-func-name" attribute, llvm.ubsantrap is lowered to a call to
; the named external C function, passing the i8 check kind as argument. That
; narrow integer argument must carry an extension attribute, as required by
; the SystemZ C ABI.

define void @test_ubsantrap_function() {
; CHECK-LABEL: test_ubsantrap_function:
; CHECK: lghi %r2, 12
; CHECK: brasl %r14, wibble@PLT
  call void @llvm.ubsantrap(i8 12) "trap-func-name"="wibble"
  ret void
}

declare void @llvm.ubsantrap(i8 immarg)

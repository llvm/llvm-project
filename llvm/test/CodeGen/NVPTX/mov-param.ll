; Checks generated with utils/update_llc_test_checks.py, then common lines shared.
; RUN: llc < %s -mtriple=nvptx -mcpu=sm_52 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,PTX32
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_52 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,PTX64

; Passing the byval address to a call keeps MOV*_PARAM live for verification.

declare void @use_pointer(ptr)

define void @byval_address(ptr byval(i32) %arg) {
; CHECK-LABEL: byval_address(
; CHECK:       {
; PTX32-NEXT:    .reg .b32 %r<3>;
; PTX64-NEXT:    .reg .b64 %rd<3>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; PTX32-NEXT:    mov.b32 %r1, byval_address_param_0;
; PTX64-NEXT:    mov.b64 %rd1, byval_address_param_0;
; PTX32-NEXT:    cvta.local.u32 %r2, %r1;
; PTX64-NEXT:    cvta.local.u64 %rd2, %rd1;
; CHECK-NEXT:    { // callseq 0, 0
; PTX32-NEXT:    .param .b32 param0;
; PTX64-NEXT:    .param .b64 param0;
; PTX32-NEXT:    st.param.b32 [param0], %r2;
; PTX64-NEXT:    st.param.b64 [param0], %rd2;
; CHECK-NEXT:    call.uni use_pointer, (param0);
; CHECK-NEXT:    } // callseq 0
; CHECK-NEXT:    ret;
  call void @use_pointer(ptr %arg)
  ret void
}

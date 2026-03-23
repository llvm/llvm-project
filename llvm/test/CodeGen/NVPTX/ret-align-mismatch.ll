; RUN: llc < %s -mtriple=nvptx64 | FileCheck %s

; Verify that return value alignment is consistent between the callee
; (LowerReturn), the declaration (printReturnValStr), and the caller
; (LowerCall). All three should honor alignstack on the return index.

target triple = "nvptx64-nvidia-cuda"

%struct.big = type { i32, i32, i32, i32, i32 }

; alignstack(4) on the return forces align 4 everywhere: the declaration,
; the callee stores, and the caller loads all use scalar b32 ops.
; CHECK-LABEL: .func (.param .align 4 .b8 func_retval0[20]) internal_ret()
; CHECK-NOT:     st.param.v4
; CHECK:         st.param.b32 [func_retval0+16], 5
; CHECK:         st.param.b32 [func_retval0+12], 4
; CHECK:         st.param.b32 [func_retval0+8], 3
; CHECK:         st.param.b32 [func_retval0+4], 2
; CHECK:         st.param.b32 [func_retval0], 1

define internal alignstack(4) %struct.big @internal_ret() {
  ret %struct.big { i32 1, i32 2, i32 3, i32 4, i32 5 }
}

; The caller also reads the return value with align 4 (scalar loads).
; CHECK-LABEL: .visible .func (.param .align 4 .b8 func_retval0[20]) caller()
; CHECK:         .param .align 4 .b8 retval0[20];
; CHECK:         call.uni (retval0), internal_ret
; CHECK:         ld.param.b32 {{%r[0-9]+}}, [retval0+16];
; CHECK:         ld.param.b32 {{%r[0-9]+}}, [retval0+12];
; CHECK:         ld.param.b32 {{%r[0-9]+}}, [retval0+8];
; CHECK:         ld.param.b32 {{%r[0-9]+}}, [retval0+4];
; CHECK:         ld.param.b32 {{%r[0-9]+}}, [retval0];

define %struct.big @caller() {
  %r = call %struct.big @internal_ret()
  ret %struct.big %r
}

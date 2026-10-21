; RUN: llc < %s -mtriple=nvptx64 | FileCheck %s

; Verify that return value alignment is consistent between the callee
; (LowerReturn), the declaration (printReturnValStr), and the caller
; (LowerCall). All three should honor alignstack on the return index.

target triple = "nvptx64-nvidia-cuda"

%struct.big = type { i32, i32, i32, i32, i32 }

; alignstack(4) on the return forces align 4 everywhere: the declaration,
; the callee stores, and the caller loads all use scalar b32 ops.
; CHECK-LABEL: .func (.param .align 4 .b8 func_retval0[20]) internal_ret_align4()
; CHECK-NOT:     st.param.v4
; CHECK:         st.param.b32 [func_retval0+16], 5
; CHECK:         st.param.b32 [func_retval0+12], 4
; CHECK:         st.param.b32 [func_retval0+8], 3
; CHECK:         st.param.b32 [func_retval0+4], 2
; CHECK:         st.param.b32 [func_retval0], 1

define internal alignstack(4) %struct.big @internal_ret_align4() {
  ret %struct.big { i32 1, i32 2, i32 3, i32 4, i32 5 }
}

; The caller also reads the return value with align 4 (scalar loads).
; CHECK-LABEL: .visible .func (.param .align 4 .b8 func_retval0[20]) caller_align4()
; CHECK:         .param .align 4 .b8 retval0[20];
; CHECK:         call.uni (retval0), internal_ret_align4
; CHECK:         ld.param.b32 {{%r[0-9]+}}, [retval0+16];
; CHECK:         ld.param.b32 {{%r[0-9]+}}, [retval0+12];
; CHECK:         ld.param.b32 {{%r[0-9]+}}, [retval0+8];
; CHECK:         ld.param.b32 {{%r[0-9]+}}, [retval0+4];
; CHECK:         ld.param.b32 {{%r[0-9]+}}, [retval0];

define %struct.big @caller_align4() {
  %r = call %struct.big @internal_ret_align4()
  ret %struct.big %r
}

; alignstack(16) permits 128-bit vectorization. The declaration, callee
; stores, and caller loads all agree on align 16 and use a v4.b32 op for
; the first four elements.
; CHECK-LABEL: .func (.param .align 16 .b8 func_retval0[20]) internal_ret_align16()
; CHECK:         st.param.b32 [func_retval0+16], 5
; CHECK:         st.param.v4.b32 [func_retval0], {1, 2, 3, 4}

define internal alignstack(16) %struct.big @internal_ret_align16() {
  ret %struct.big { i32 1, i32 2, i32 3, i32 4, i32 5 }
}

; CHECK-LABEL: .visible .func (.param .align 4 .b8 func_retval0[20]) caller_align16()
; CHECK:         .param .align 16 .b8 retval0[20];
; CHECK:         call.uni (retval0), internal_ret_align16
; CHECK:         ld.param.b32 {{%r[0-9]+}}, [retval0+16];
; CHECK:         ld.param.v4.b32 {{{%r[0-9]+, %r[0-9]+, %r[0-9]+, %r[0-9]+}}}, [retval0];

define %struct.big @caller_align16() {
  %r = call %struct.big @internal_ret_align16()
  ret %struct.big %r
}

; With no explicit alignstack, an internal-linkage callee gets its return
; alignment bumped to 16 by the param-align optimization, so vectorization
; still kicks in on both sides of the call.
; CHECK-LABEL: .func (.param .align 16 .b8 func_retval0[20]) internal_ret_default()
; CHECK:         st.param.b32 [func_retval0+16], 5
; CHECK:         st.param.v4.b32 [func_retval0], {1, 2, 3, 4}

define internal %struct.big @internal_ret_default() {
  ret %struct.big { i32 1, i32 2, i32 3, i32 4, i32 5 }
}

; CHECK-LABEL: .visible .func (.param .align 4 .b8 func_retval0[20]) caller_default()
; CHECK:         .param .align 16 .b8 retval0[20];
; CHECK:         call.uni (retval0), internal_ret_default
; CHECK:         ld.param.b32 {{%r[0-9]+}}, [retval0+16];
; CHECK:         ld.param.v4.b32 {{{%r[0-9]+, %r[0-9]+, %r[0-9]+, %r[0-9]+}}}, [retval0];

define %struct.big @caller_default() {
  %r = call %struct.big @internal_ret_default()
  ret %struct.big %r
}

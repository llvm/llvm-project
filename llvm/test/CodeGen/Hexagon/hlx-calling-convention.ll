; RUN: llc -march=hexagon -mattr=+hlx -mcpu=hexagonv85 -stop-after=finalize-isel < %s | FileCheck %s

; Check that calls to functions that take or return HLX vectors use the call
; preserved mask that keeps L12-L15.

; CHECK-LABEL: name: nohlx_signature
define i32 @nohlx_signature() nounwind {
  ret i32 42
}

; CHECK-LABEL: name: hlx_signature_ret_vec
define <1024 x i16> @hlx_signature_ret_vec() nounwind {
  ret <1024 x i16> poison
}

; CHECK-LABEL: name: hlx_signature_arg_vec
define void @hlx_signature_arg_vec(<1024 x i16> %arg) nounwind {
  ret void
}

; CHECK-LABEL: name: caller_nohlx_signature
; CHECK: J2_call @nohlx_signature, hexagoncsr,
define i32 @caller_nohlx_signature() nounwind {
  %res = call i32 @nohlx_signature()
  ret i32 %res
}

; CHECK-LABEL: name: hlx_signature_ret_vec_caller
; CHECK: J2_call @hlx_signature_ret_vec, hexagoncsr_hlx,
define <1024 x i16> @hlx_signature_ret_vec_caller() nounwind {
  %res = call <1024 x i16> @hlx_signature_ret_vec()
  ret <1024 x i16> %res
}

; CHECK-LABEL: name: hlx_signature_arg_vec_caller
; CHECK: J2_call @hlx_signature_arg_vec, hexagoncsr_hlx,
define void @hlx_signature_arg_vec_caller(<1024 x i16> %arg) nounwind {
  call void @hlx_signature_arg_vec(<1024 x i16> %arg)
  ret void
}


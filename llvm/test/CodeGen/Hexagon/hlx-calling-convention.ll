; RUN: llc -march=hexagon -mattr=+hlx -mcpu=hexagonv85 -stop-after=finalize-isel < %s | FileCheck %s
; RUN: llc -march=hexagon -mattr=+hlx -mcpu=hexagonv85 -enable-misched=false -stop-after=prolog-epilog < %s | FileCheck %s --check-prefix=CHECK-PEI

; Check that calls to functions that take or return HLX vectors use the call
; preserved mask that keeps L12-L15, and that HLX vectors after the first 12
; are passed on the VTCM stack.

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


; CHECK-LABEL: name: hlx_signature_many_arg_vec
; CHECK: %[[INTREG0:[0-9]+]]:intregs = PS_fi %fixed-stack.0, 0
; CHECK: %[[HLXREG0:[0-9]+]]:hlxlr = LX_L16b_pseudo %[[INTREG0]]
; CHECK: %[[INTREG1:[0-9]+]]:intregs = PS_fi %fixed-stack.1, 0
; CHECK: %[[HLXREG1:[0-9]+]]:hlxlr = LX_L16b_pseudo %[[INTREG1]]
; CHECK: %[[HLXREG2:[0-9]+]]:hlxlr = LX_add_h %[[HLXREG1]], %[[HLXREG0]]
; CHECK: $l0 = COPY %[[HLXREG2]]
;
; CHECK-PEI-LABEL: name: hlx_signature_many_arg_vec
; CHECK-PEI-NOT: V2_vallocframe_imm
; CHECK-PEI: $[[INTREG0:r[0-9]+]] = A2_tfrcrr $vsp
; CHECK-PEI: $[[INTREG1:r[0-9]+]] = A2_addi killed $[[INTREG0]], 6144
; CHECK-PEI: $[[HLXREG0:l[0-9]+]] = LX_L16b_pseudo killed{{.*}}$[[INTREG1]]
; CHECK-PEI: $[[INTREG2:r[0-9]+]] = A2_tfrcrr $vsp
; CHECK-PEI: $[[INTREG3:r[0-9]+]] = A2_addi killed $[[INTREG2]], 4096
; CHECK-PEI: $[[HLXREG1:l[0-9]+]] = LX_L16b_pseudo killed{{.*}}$[[INTREG3]]
; CHECK-PEI: LX_add_h killed{{.*}}$[[HLXREG1]], killed{{.*}}$[[HLXREG0]]
define <1024 x i16> @hlx_signature_many_arg_vec(<1024 x i16> %arg1, <1024 x i16> %arg2, <1024 x i16> %arg3, <1024 x i16> %arg4, <1024 x i16> %arg5, <1024 x i16> %arg6, <1024 x i16> %arg7, <1024 x i16> %arg8,
                                                <1024 x i16> %arg9, <1024 x i16> %arg10, <1024 x i16> %arg11, <1024 x i16> %arg12, <1024 x i16> %arg13, <1024 x i16> %arg14, <1024 x i16> %arg15, <1024 x i16> %arg16) nounwind {
  %res = add <1024 x i16> %arg15, %arg16
  ret <1024 x i16> %res
}

; CHECK-LABEL: name: hlx_signature_many_arg_vec_caller
; CHECK-DAG: %[[HLXREG4:[0-9]+]]:hlxlr = COPY $l4
; CHECK-DAG: %[[HLXREG3:[0-9]+]]:hlxlr = COPY $l3
; CHECK-DAG: %[[HLXREG2:[0-9]+]]:hlxlr = COPY $l2
; CHECK-DAG: %[[HLXREG1:[0-9]+]]:hlxlr = COPY $l1
; CHECK-DAG: %[[HLXREG0:[0-9]+]]:hlxlr = COPY $l0
; CHECK: %[[VSP:[0-9]+]]:intregs = COPY $vsp
; CHECK-DAG: LX_S16b_pseudo %[[VSP]], %[[HLXREG1]] :: (store (s16384) into stack)
; CHECK-DAG: %[[INTREG0:[0-9]+]]:intregs = A2_addi %[[VSP]], 6144
; CHECK-DAG: LX_S16b_pseudo %[[INTREG0]], %[[HLXREG4]] :: (store (s16384) into stack + 6144)
; CHECK-DAG: %[[INTREG1:[0-9]+]]:intregs = A2_addi %[[VSP]], 4096
; CHECK-DAG: LX_S16b_pseudo %[[INTREG1]], %[[HLXREG3]] :: (store (s16384) into stack + 4096)
; CHECK-DAG: %[[INTREG2:[0-9]+]]:intregs = A2_addi %[[VSP]], 2048
; CHECK-DAG: LX_S16b_pseudo %[[INTREG2]], %[[HLXREG2]] :: (store (s16384) into stack + 2048)
; CHECK-DAG: $l0 = COPY %[[HLXREG0]]
; CHECK-DAG: $l11 = COPY %[[HLXREG0]]
; CHECK: J2_call @hlx_signature_many_arg_vec, hexagoncsr_hlx,
;
; CHECK-PEI-LABEL: name: hlx_signature_many_arg_vec_caller
; CHECK-PEI: frame-setup V2_vallocframe_imm 4
; CHECK-PEI: J2_call @hlx_signature_many_arg_vec
; CHECK-PEI: V2_vdeallocframe_imm 4
define <1024 x i16> @hlx_signature_many_arg_vec_caller(<1024 x i16> %arg, <1024 x i16> %arg1, <1024 x i16> %arg2, <1024 x i16> %arg3, <1024 x i16> %arg4) nounwind {
  %res = call <1024 x i16> @hlx_signature_many_arg_vec(<1024 x i16> %arg, <1024 x i16> %arg, <1024 x i16> %arg, <1024 x i16> %arg, <1024 x i16> %arg,
                                                       <1024 x i16> %arg, <1024 x i16> %arg, <1024 x i16> %arg, <1024 x i16> %arg, <1024 x i16> %arg,
                                                       <1024 x i16> %arg, <1024 x i16> %arg, <1024 x i16> %arg1, <1024 x i16> %arg2, <1024 x i16> %arg3,
                                                       <1024 x i16> %arg4)
  ret <1024 x i16> %res
}

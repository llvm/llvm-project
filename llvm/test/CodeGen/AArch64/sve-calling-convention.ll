; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -stop-after=finalize-isel < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; CHECK-LABEL: name: nosve_signature
define i32 @nosve_signature() nounwind {
  ret i32 42
}

; CHECK-LABEL: name: sve_signature_ret_vec
define <vscale x 4 x i32> @sve_signature_ret_vec() nounwind {
  ret <vscale x 4 x i32> undef
}

; CHECK-LABEL: name: sve_signature_ret_pred
define <vscale x 4 x i1> @sve_signature_ret_pred() nounwind {
  ret <vscale x 4 x i1> undef
}

; CHECK-LABEL: name: sve_signature_arg_vec
define void @sve_signature_arg_vec(<vscale x 4 x i32> %arg) nounwind {
  ret void
}

; CHECK-LABEL: name: sve_signature_arg_pred
define void @sve_signature_arg_pred(<vscale x 4 x i1> %arg) nounwind {
  ret void
}

; CHECK-LABEL: name: caller_nosve_signature
; CHECK: BL @nosve_signature, csr_aarch64_aapcs
define i32 @caller_nosve_signature() nounwind {
  %res = call i32 @nosve_signature()
  ret i32 %res
}

; CHECK-LABEL: name: sve_signature_ret_vec_caller
; CHECK: BL @sve_signature_ret_vec, csr_aarch64_sve_aapcs
define <vscale x 4 x i32>  @sve_signature_ret_vec_caller() nounwind {
  %res = call <vscale x 4 x i32> @sve_signature_ret_vec()
  ret <vscale x 4 x i32> %res
}

; CHECK-LABEL: name: sve_signature_ret_pred_caller
; CHECK: BL @sve_signature_ret_pred, csr_aarch64_sve_aapcs
define <vscale x 4 x i1>  @sve_signature_ret_pred_caller() nounwind {
  %res = call <vscale x 4 x i1> @sve_signature_ret_pred()
  ret <vscale x 4 x i1> %res
}

; CHECK-LABEL: name: sve_signature_arg_vec_caller
; CHECK: BL @sve_signature_arg_vec, csr_aarch64_sve_aapcs
define void @sve_signature_arg_vec_caller(<vscale x 4 x i32> %arg) nounwind {
  call void @sve_signature_arg_vec(<vscale x 4 x i32> %arg)
  ret void
}

; CHECK-LABEL: name: sve_signature_arg_pred_caller
; CHECK: BL @sve_signature_arg_pred, csr_aarch64_sve_aapcs
define void @sve_signature_arg_pred_caller(<vscale x 4 x i1> %arg) nounwind {
  call void @sve_signature_arg_pred(<vscale x 4 x i1> %arg)
  ret void
}

; CHECK-LABEL: name: sve_signature_many_arg_vec
; CHECK: [[RES:%[0-9]+]]:zpr = COPY $z7
; CHECK: $z0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $z0
define <vscale x 4 x i32> @sve_signature_many_arg_vec(<vscale x 4 x i32> %arg1, <vscale x 4 x i32> %arg2, <vscale x 4 x i32> %arg3, <vscale x 4 x i32> %arg4, <vscale x 4 x i32> %arg5, <vscale x 4 x i32> %arg6, <vscale x 4 x i32> %arg7, <vscale x 4 x i32> %arg8) nounwind {
  ret <vscale x 4 x i32> %arg8
}

; CHECK-LABEL: name: sve_signature_many_arg_pred
; CHECK: [[RES:%[0-9]+]]:ppr = COPY $p3
; CHECK: $p0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $p0
define <vscale x 4 x i1> @sve_signature_many_arg_pred(<vscale x 4 x i1> %arg1, <vscale x 4 x i1> %arg2, <vscale x 4 x i1> %arg3, <vscale x 4 x i1> %arg4) nounwind {
  ret <vscale x 4 x i1> %arg4
}

; CHECK-LABEL: name: sve_signature_vec
; CHECK: [[RES:%[0-9]+]]:zpr = COPY $z1
; CHECK: $z0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $z0
define <vscale x 4 x i32> @sve_signature_vec(<vscale x 4 x i32> %arg1, <vscale x 4 x i32> %arg2) nounwind {
 ret <vscale x 4 x i32> %arg2
}

; CHECK-LABEL: name: sve_signature_pred
; CHECK: [[RES:%[0-9]+]]:ppr = COPY $p1
; CHECK: $p0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $p0
define <vscale x 4 x i1> @sve_signature_pred(<vscale x 4 x i1> %arg1, <vscale x 4 x i1> %arg2) nounwind {
  ret <vscale x 4 x i1> %arg2
}

; CHECK-LABEL: name: sve_signature_vec_caller
; CHECK-DAG: [[ARG2:%[0-9]+]]:zpr = COPY $z1
; CHECK-DAG: [[ARG1:%[0-9]+]]:zpr = COPY $z0
; CHECK-DAG: $z0 = COPY [[ARG2]]
; CHECK-DAG: $z1 = COPY [[ARG1]]
; CHECK-NEXT: BL @sve_signature_vec, csr_aarch64_sve_aapcs
; CHECK: [[RES:%[0-9]+]]:zpr = COPY $z0
; CHECK: $z0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $z0
define <vscale x 4 x i32> @sve_signature_vec_caller(<vscale x 4 x i32> %arg1, <vscale x 4 x i32> %arg2) nounwind {
  %res = call <vscale x 4 x i32> @sve_signature_vec(<vscale x 4 x i32> %arg2, <vscale x 4 x i32> %arg1)
  ret <vscale x 4 x i32> %res
}

; CHECK-LABEL: name: sve_signature_pred_caller
; CHECK-DAG: [[ARG2:%[0-9]+]]:ppr = COPY $p1
; CHECK-DAG: [[ARG1:%[0-9]+]]:ppr = COPY $p0
; CHECK-DAG: $p0 = COPY [[ARG2]]
; CHECK-DAG: $p1 = COPY [[ARG1]]
; CHECK-NEXT: BL @sve_signature_pred, csr_aarch64_sve_aapcs
; CHECK: [[RES:%[0-9]+]]:ppr = COPY $p0
; CHECK: $p0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $p0
define <vscale x 4 x i1> @sve_signature_pred_caller(<vscale x 4 x i1> %arg1, <vscale x 4 x i1> %arg2) nounwind {
  %res = call <vscale x 4 x i1> @sve_signature_pred(<vscale x 4 x i1> %arg2, <vscale x 4 x i1> %arg1)
  ret <vscale x 4 x i1> %res
}

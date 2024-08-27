; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -stop-after=finalize-isel < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-darwin -mattr=+sve -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=DARWIN
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -stop-after=prologepilog < %s | FileCheck %s --check-prefix=CHECKCSR
; RUN: llc -mtriple=aarch64-apple-darwin -mattr=+sve -stop-after=prologepilog < %s | FileCheck %s --check-prefix=CHECKCSR

; CHECK-LABEL: name: nosve_signature
; DARWIN-LABEL: name: nosve_signature
define i32 @nosve_signature() nounwind {
  ret i32 42
}

; CHECK-LABEL: name: sve_signature_ret_vec
; DARWIN-LABEL: name: sve_signature_ret_vec
define <vscale x 4 x i32> @sve_signature_ret_vec() nounwind {
  ret <vscale x 4 x i32> undef
}

; CHECK-LABEL: name: sve_signature_ret_pred
; DARWIN-LABEL: name: sve_signature_ret_pred
define <vscale x 4 x i1> @sve_signature_ret_pred() nounwind {
  ret <vscale x 4 x i1> undef
}

; CHECK-LABEL: name: sve_signature_arg_vec
; DARWIN-LABEL: name: sve_signature_arg_vec
define void @sve_signature_arg_vec(<vscale x 4 x i32> %arg) nounwind {
  ret void
}

; CHECK-LABEL: name: sve_signature_arg_pred
; DARWIN-LABEL: name: sve_signature_arg_pred
define void @sve_signature_arg_pred(<vscale x 4 x i1> %arg) nounwind {
  ret void
}

; CHECK-LABEL: name: caller_nosve_signature
; CHECK: BL @nosve_signature, csr_aarch64_aapcs
; DARWIN-LABEL: name: caller_nosve_signature
; DARWIN: BL @nosve_signature, csr_darwin_aarch64_aapcs
define i32 @caller_nosve_signature() nounwind {
  %res = call i32 @nosve_signature()
  ret i32 %res
}

; CHECK-LABEL: name: caller_nosve_signature_fastcc
; CHECK: BL @nosve_signature, csr_aarch64_aapcs
; DARWIN-LABEL: name: caller_nosve_signature_fastcc
; DARWIN: BL @nosve_signature, csr_darwin_aarch64_aapcs
define i32 @caller_nosve_signature_fastcc() nounwind {
  %res = call fastcc i32 @nosve_signature()
  ret i32 %res
}

; CHECK-LABEL: name: sve_signature_ret_vec_caller
; CHECK: BL @sve_signature_ret_vec, csr_aarch64_sve_aapcs
; DARWIN-LABEL: name: sve_signature_ret_vec_caller
; DARWIN: BL @sve_signature_ret_vec, csr_darwin_aarch64_sve_aapcs
define <vscale x 4 x i32>  @sve_signature_ret_vec_caller() nounwind {
  %res = call <vscale x 4 x i32> @sve_signature_ret_vec()
  ret <vscale x 4 x i32> %res
}

; CHECK-LABEL: name: sve_signature_ret_vec_caller_fastcc
; CHECK: BL @sve_signature_ret_vec, csr_aarch64_sve_aapcs
; DARWIN-LABEL: name: sve_signature_ret_vec_caller_fastcc
; DARWIN: BL @sve_signature_ret_vec, csr_darwin_aarch64_sve_aapcs
define <vscale x 4 x i32>  @sve_signature_ret_vec_caller_fastcc() nounwind {
  %res = call fastcc <vscale x 4 x i32> @sve_signature_ret_vec()
  ret <vscale x 4 x i32> %res
}

; CHECK-LABEL: name: sve_signature_ret_pred_caller
; CHECK: BL @sve_signature_ret_pred, csr_aarch64_sve_aapcs
; DARWIN-LABEL: name: sve_signature_ret_pred_caller
; DARWIN: BL @sve_signature_ret_pred, csr_darwin_aarch64_sve_aapcs
define <vscale x 4 x i1>  @sve_signature_ret_pred_caller() nounwind {
  %res = call <vscale x 4 x i1> @sve_signature_ret_pred()
  ret <vscale x 4 x i1> %res
}

; CHECK-LABEL: name: sve_signature_ret_pred_caller_fastcc
; CHECK: BL @sve_signature_ret_pred, csr_aarch64_sve_aapcs
; DARWIN-LABEL: name: sve_signature_ret_pred_caller_fastcc
; DARWIN: BL @sve_signature_ret_pred, csr_darwin_aarch64_sve_aapcs
define <vscale x 4 x i1>  @sve_signature_ret_pred_caller_fastcc() nounwind {
  %res = call fastcc <vscale x 4 x i1> @sve_signature_ret_pred()
  ret <vscale x 4 x i1> %res
}

; CHECK-LABEL: name: sve_signature_arg_vec_caller
; CHECK: BL @sve_signature_arg_vec, csr_aarch64_sve_aapcs
; DARWIN-LABEL: name: sve_signature_arg_vec_caller
; DARWIN: BL @sve_signature_arg_vec, csr_darwin_aarch64_sve_aapcs
define void @sve_signature_arg_vec_caller(<vscale x 4 x i32> %arg) nounwind {
  call void @sve_signature_arg_vec(<vscale x 4 x i32> %arg)
  ret void
}

; CHECK-LABEL: name: sve_signature_arg_vec_caller_fastcc
; CHECK: BL @sve_signature_arg_vec, csr_aarch64_sve_aapcs
; DARWIN-LABEL: name: sve_signature_arg_vec_caller_fastcc
; DARWIN: BL @sve_signature_arg_vec, csr_darwin_aarch64_sve_aapcs
define void @sve_signature_arg_vec_caller_fastcc(<vscale x 4 x i32> %arg) nounwind {
  call fastcc void @sve_signature_arg_vec(<vscale x 4 x i32> %arg)
  ret void
}

; CHECK-LABEL: name: sve_signature_arg_pred_caller
; CHECK: BL @sve_signature_arg_pred, csr_aarch64_sve_aapcs
; DARWIN-LABEL: name: sve_signature_arg_pred_caller
; DARWIN: BL @sve_signature_arg_pred, csr_darwin_aarch64_sve_aapcs
define void @sve_signature_arg_pred_caller(<vscale x 4 x i1> %arg) nounwind {
  call void @sve_signature_arg_pred(<vscale x 4 x i1> %arg)
  ret void
}

; CHECK-LABEL: name: sve_signature_arg_pred_caller_fastcc
; CHECK: BL @sve_signature_arg_pred, csr_aarch64_sve_aapcs
; DARWIN-LABEL: name: sve_signature_arg_pred_caller_fastcc
; DARWIN: BL @sve_signature_arg_pred, csr_darwin_aarch64_sve_aapcs
define void @sve_signature_arg_pred_caller_fastcc(<vscale x 4 x i1> %arg) nounwind {
  call fastcc void @sve_signature_arg_pred(<vscale x 4 x i1> %arg)
  ret void
}

; CHECK-LABEL: name: sve_signature_many_arg_vec
; CHECK: [[RES:%[0-9]+]]:zpr = COPY $z7
; CHECK: $z0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $z0
; DARWIN-LABEL: name: sve_signature_many_arg_vec
; DARWIN: [[RES:%[0-9]+]]:zpr = COPY $z7
; DARWIN: $z0 = COPY [[RES]]
; DARWIN: RET_ReallyLR implicit $z0
define <vscale x 4 x i32> @sve_signature_many_arg_vec(<vscale x 4 x i32> %arg1, <vscale x 4 x i32> %arg2, <vscale x 4 x i32> %arg3, <vscale x 4 x i32> %arg4, <vscale x 4 x i32> %arg5, <vscale x 4 x i32> %arg6, <vscale x 4 x i32> %arg7, <vscale x 4 x i32> %arg8) nounwind {
  ret <vscale x 4 x i32> %arg8
}

; CHECK-LABEL: name: sve_signature_many_arg_pred
; CHECK: [[RES:%[0-9]+]]:ppr = COPY $p3
; CHECK: $p0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $p0
; DARWIN-LABEL: name: sve_signature_many_arg_pred
; DARWIN: [[RES:%[0-9]+]]:ppr = COPY $p3
; DARWIN: $p0 = COPY [[RES]]
; DARWIN: RET_ReallyLR implicit $p0
define <vscale x 4 x i1> @sve_signature_many_arg_pred(<vscale x 4 x i1> %arg1, <vscale x 4 x i1> %arg2, <vscale x 4 x i1> %arg3, <vscale x 4 x i1> %arg4) nounwind {
  ret <vscale x 4 x i1> %arg4
}

; CHECK-LABEL: name: sve_signature_vec
; CHECK: [[RES:%[0-9]+]]:zpr = COPY $z1
; CHECK: $z0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $z0
; DARWIN-LABEL: name: sve_signature_vec
; DARWIN: [[RES:%[0-9]+]]:zpr = COPY $z1
; DARWIN: $z0 = COPY [[RES]]
; DARWIN: RET_ReallyLR implicit $z0
define <vscale x 4 x i32> @sve_signature_vec(<vscale x 4 x i32> %arg1, <vscale x 4 x i32> %arg2) nounwind {
 ret <vscale x 4 x i32> %arg2
}

; CHECK-LABEL: name: sve_signature_pred
; CHECK: [[RES:%[0-9]+]]:ppr = COPY $p1
; CHECK: $p0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $p0
; DARWIN-LABEL: name: sve_signature_pred
; DARWIN: [[RES:%[0-9]+]]:ppr = COPY $p1
; DARWIN: $p0 = COPY [[RES]]
; DARWIN: RET_ReallyLR implicit $p0
define <vscale x 4 x i1> @sve_signature_pred(<vscale x 4 x i1> %arg1, <vscale x 4 x i1> %arg2) nounwind {
  ret <vscale x 4 x i1> %arg2
}

; Test that scalable predicate argument in [1 x <vscale x 4 x i1>] type are properly assigned to P registers.
; CHECK-LABEL: name: sve_signature_pred_1xv4i1
; CHECK: [[RES:%[0-9]+]]:ppr = COPY $p1
; CHECK: $p0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $p0
define [1 x <vscale x 4 x i1>] @sve_signature_pred_1xv4i1([1 x <vscale x 4 x i1>] %arg1, [1 x <vscale x 4 x i1>] %arg2) nounwind {
  ret [1 x <vscale x 4 x i1>] %arg2
}

; Test that upto to two scalable predicate arguments in [2 x <vscale x 4 x i1>] type can be assigned to P registers.
; CHECK-LABEL: name: sve_signature_pred_2xv4i1
; CHECK: [[RES1:%[0-9]+]]:ppr = COPY $p3
; CHECK: [[RES0:%[0-9]+]]:ppr = COPY $p2
; CHECK: $p0 = COPY [[RES0]]
; CHECK: $p1 = COPY [[RES1]]
; CHECK: RET_ReallyLR implicit $p0, implicit $p1
define [2 x <vscale x 4 x i1>] @sve_signature_pred_2xv4i1([2 x <vscale x 4 x i1>] %arg1, [2 x <vscale x 4 x i1>] %arg2) nounwind {
  ret [2 x <vscale x 4 x i1>] %arg2
}

; Test that a scalable predicate argument in [1 x <vscale x 32 x i1>] type is assigned to two P registers.
; CHECK-LABLE: name: sve_signature_pred_1xv32i1
; CHECK: [[RES1:%[0-9]+]]:ppr = COPY $p3
; CHECK: [[RES0:%[0-9]+]]:ppr = COPY $p2
; CHECK: $p0 = COPY [[RES0]]
; CHECK: $p1 = COPY [[RES1]]
; CHECK: RET_ReallyLR implicit $p0, implicit $p1
define [1 x <vscale x 32 x i1>] @sve_signature_pred_1xv32i1([1 x <vscale x 32 x i1>] %arg1, [1 x <vscale x 32 x i1>] %arg2) nounwind {
  ret [1 x <vscale x 32 x i1>] %arg2
}

; Test that a scalable predicate argument in [2 x <vscale x 32 x i1>] type is assigned to four P registers.
; CHECK-LABLE: name: sve_signature_pred_2xv32i1
; CHECK: [[RES3:%[0-9]+]]:ppr = COPY $p3
; CHECK: [[RES2:%[0-9]+]]:ppr = COPY $p2
; CHECK: [[RES1:%[0-9]+]]:ppr = COPY $p1
; CHECK: [[RES0:%[0-9]+]]:ppr = COPY $p0
; CHECK: $p0 = COPY [[RES0]]
; CHECK: $p1 = COPY [[RES1]]
; CHECK: $p2 = COPY [[RES2]]
; CHECK: $p3 = COPY [[RES3]]
; CHECK: RET_ReallyLR implicit $p0, implicit $p1, implicit $p2, implicit $p3
define [2 x <vscale x 32 x i1>] @sve_signature_pred_2xv32i1([2 x <vscale x 32 x i1>] %arg1) nounwind {
  ret [2 x <vscale x 32 x i1>] %arg1
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
; DARWIN-LABEL: name: sve_signature_vec_caller
; DARWIN-DAG: [[ARG2:%[0-9]+]]:zpr = COPY $z1
; DARWIN-DAG: [[ARG1:%[0-9]+]]:zpr = COPY $z0
; DARWIN-DAG: $z0 = COPY [[ARG2]]
; DARWIN-DAG: $z1 = COPY [[ARG1]]
; DARWIN-NEXT: BL @sve_signature_vec, csr_darwin_aarch64_sve_aapcs
; DARWIN: [[RES:%[0-9]+]]:zpr = COPY $z0
; DARWIN: $z0 = COPY [[RES]]
; DARWIN: RET_ReallyLR implicit $z0
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
; DARWIN-LABEL: name: sve_signature_pred_caller
; DARWIN-DAG: [[ARG2:%[0-9]+]]:ppr = COPY $p1
; DARWIN-DAG: [[ARG1:%[0-9]+]]:ppr = COPY $p0
; DARWIN-DAG: $p0 = COPY [[ARG2]]
; DARWIN-DAG: $p1 = COPY [[ARG1]]
; DARWIN-NEXT: BL @sve_signature_pred, csr_darwin_aarch64_sve_aapcs
; DARWIN: [[RES:%[0-9]+]]:ppr = COPY $p0
; DARWIN: $p0 = COPY [[RES]]
; DARWIN: RET_ReallyLR implicit $p0
define <vscale x 4 x i1> @sve_signature_pred_caller(<vscale x 4 x i1> %arg1, <vscale x 4 x i1> %arg2) nounwind {
  %res = call <vscale x 4 x i1> @sve_signature_pred(<vscale x 4 x i1> %arg2, <vscale x 4 x i1> %arg1)
  ret <vscale x 4 x i1> %res
}

; CHECK-LABEL: name: sve_signature_pred_1xv4i1_caller
; CHECK-DAG: [[ARG2:%[0-9]+]]:ppr = COPY $p1
; CHECK-DAG: [[ARG1:%[0-9]+]]:ppr = COPY $p0
; CHECK-DAG: $p0 = COPY [[ARG2]]
; CHECK-DAG: $p1 = COPY [[ARG1]]
; CHECK-NEXT: BL @sve_signature_pred_1xv4i1, csr_aarch64_sve_aapcs
; CHECK: [[RES:%[0-9]+]]:ppr = COPY $p0
; CHECK: $p0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $p0
define [1 x <vscale x 4 x i1>] @sve_signature_pred_1xv4i1_caller([1 x <vscale x 4 x i1>] %arg1, [1 x <vscale x 4 x i1>] %arg2) nounwind {
  %res = call [1 x <vscale x 4 x i1>] @sve_signature_pred_1xv4i1([1 x <vscale x 4 x i1>] %arg2, [1 x <vscale x 4 x i1>] %arg1)
  ret [1 x <vscale x 4 x i1>] %res
}

; CHECK-LABEL: name: sve_signature_pred_2xv4i1_caller
; CHECK-DAG: [[ARG2_2:%[0-9]+]]:ppr = COPY $p3
; CHECK-DAG: [[ARG2_1:%[0-9]+]]:ppr = COPY $p2
; CHECK-DAG: [[ARG1_2:%[0-9]+]]:ppr = COPY $p1
; CHECK-DAG: [[ARG1_1:%[0-9]+]]:ppr = COPY $p0
; CHECK-DAG: $p0 = COPY [[ARG2_1]]
; CHECK-DAG: $p1 = COPY [[ARG2_2]]
; CHECK-DAG: $p2 = COPY [[ARG1_1]]
; CHECK-DAG: $p3 = COPY [[ARG1_2]]
; CHECK-NEXT: BL @sve_signature_pred_2xv4i1, csr_aarch64_sve_aapcs
; CHECK: [[RES0:%[0-9]+]]:ppr = COPY $p0
; CHECK: [[RES1:%[0-9]+]]:ppr = COPY $p1
; CHECK: $p0 = COPY [[RES0]]
; CHECK: $p1 = COPY [[RES1]]
; CHECK: RET_ReallyLR implicit $p0, implicit $p1
define [2 x <vscale x 4 x i1>] @sve_signature_pred_2xv4i1_caller([2 x <vscale x 4 x i1>] %arg1, [2 x <vscale x 4 x i1>] %arg2) nounwind {
  %res = call [2 x <vscale x 4 x i1>] @sve_signature_pred_2xv4i1([2 x <vscale x 4 x i1>] %arg2, [2 x <vscale x 4 x i1>] %arg1)
  ret [2 x <vscale x 4 x i1>] %res
}

; CHECK-LABEL: name: sve_signature_pred_1xv32i1_caller
; CHECK-DAG: [[ARG2_2:%[0-9]+]]:ppr = COPY $p3
; CHECK-DAG: [[ARG2_1:%[0-9]+]]:ppr = COPY $p2
; CHECK-DAG: [[ARG1_2:%[0-9]+]]:ppr = COPY $p1
; CHECK-DAG: [[ARG1_1:%[0-9]+]]:ppr = COPY $p0
; CHECK-DAG: $p0 = COPY [[ARG2_1]]
; CHECK-DAG: $p1 = COPY [[ARG2_2]]
; CHECK-DAG: $p2 = COPY [[ARG1_1]]
; CHECK-DAG: $p3 = COPY [[ARG1_2]]
; CHECK-NEXT: BL @sve_signature_pred_1xv32i1, csr_aarch64_sve_aapcs
; CHECK: [[RES0:%[0-9]+]]:ppr = COPY $p0
; CHECK: [[RES1:%[0-9]+]]:ppr = COPY $p1
; CHECK: $p0 = COPY [[RES0]]
; CHECK: $p1 = COPY [[RES1]]
; CHECK: RET_ReallyLR implicit $p0, implicit $p1
define [1 x <vscale x 32 x i1>] @sve_signature_pred_1xv32i1_caller([1 x <vscale x 32 x i1>] %arg1, [1 x <vscale x 32 x i1>] %arg2) nounwind {
  %res = call [1 x <vscale x 32 x i1>] @sve_signature_pred_1xv32i1([1 x <vscale x 32 x i1>] %arg2, [1 x <vscale x 32 x i1>] %arg1)
  ret [1 x <vscale x 32 x i1>] %res
}

; CHECK-LABEL: name: sve_signature_pred_2xv32i1_caller
; CHECK-DAG: [[ARG3:%[0-9]+]]:ppr = COPY $p3
; CHECK-DAG: [[ARG2:%[0-9]+]]:ppr = COPY $p2
; CHECK-DAG: [[ARG1:%[0-9]+]]:ppr = COPY $p1
; CHECK-DAG: [[ARG0:%[0-9]+]]:ppr = COPY $p0
; CHECK-DAG: $p0 = COPY [[ARG0]]
; CHECK-DAG: $p1 = COPY [[ARG1]]
; CHECK-DAG: $p2 = COPY [[ARG2]]
; CHECK-DAG: $p3 = COPY [[ARG3]]
; CHECK-NEXT: BL @sve_signature_pred_2xv32i1, csr_aarch64_sve_aapcs
; CHECK: [[RES0:%[0-9]+]]:ppr = COPY $p0
; CHECK: [[RES1:%[0-9]+]]:ppr = COPY $p1
; CHECK: [[RES2:%[0-9]+]]:ppr = COPY $p2
; CHECK: [[RES3:%[0-9]+]]:ppr = COPY $p3
; CHECK: $p0 = COPY [[RES0]]
; CHECK: $p1 = COPY [[RES1]]
; CHECK: $p2 = COPY [[RES2]]
; CHECK: $p3 = COPY [[RES3]]
; CHECK: RET_ReallyLR implicit $p0, implicit $p1
define [2 x <vscale x 32 x i1>] @sve_signature_pred_2xv32i1_caller([2 x <vscale x 32 x i1>] %arg1) {
  %res = call [2 x <vscale x 32 x i1>] @sve_signature_pred_2xv32i1([2 x <vscale x 32 x i1>] %arg1)
  ret [2 x <vscale x 32 x i1>] %res
}

; Test that functions returning or taking SVE arguments use the correct
; callee-saved set when using the default C calling convention (as opposed
; to aarch64_sve_vector_pcs)

; CHECKCSR-LABEL: name: sve_signature_vec_ret_callee
; CHECKCSR: callee-saved-register: '$z8'
; CHECKCSR: callee-saved-register: '$p4'
; CHECKCSR: RET_ReallyLR
define <vscale x 4 x i32> @sve_signature_vec_ret_callee() nounwind {
  call void asm sideeffect "nop", "~{z8},~{p4}"()
  ret <vscale x 4 x i32> zeroinitializer
}

; CHECKCSR-LABEL: name: sve_signature_vec_arg_callee
; CHECKCSR: callee-saved-register: '$z8'
; CHECKCSR: callee-saved-register: '$p4'
; CHECKCSR: RET_ReallyLR
define void @sve_signature_vec_arg_callee(<vscale x 4 x i32> %v) nounwind {
  call void asm sideeffect "nop", "~{z8},~{p4}"()
  ret void
}

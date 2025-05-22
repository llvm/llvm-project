; RUN: llc -mtriple=nvptx64 -stop-before=nvptx-proxyreg-erasure < %s 2>&1 \
; RUN: | llc -x mir -mtriple=nvptx64 -start-before=nvptx-proxyreg-erasure 2>&1 \
; RUN:   | FileCheck %s --check-prefix=PTX --check-prefix=PTX-WITH

; RUN: llc -mtriple=nvptx64 -stop-before=nvptx-proxyreg-erasure < %s 2>&1 \
; RUN: | llc -x mir -mtriple=nvptx64 -start-after=nvptx-proxyreg-erasure 2>&1 \
; RUN:   | FileCheck %s --check-prefix=PTX --check-prefix=PTX-WITHOUT

; Thorough testing of ProxyRegErasure: PTX assembly with and without the pass.

declare i1 @callee_i1()
define i1 @check_i1() {
  ; PTX-LABEL: check_i1
  ; PTX-DAG: { // callseq {{[0-9]+}}, {{[0-9]+}}
  ; PTX-DAG: ld.param.b32 [[LD:%r[0-9]+]], [retval0];
  ; PTX-DAG: } // callseq {{[0-9]+}}

  ; PTX-WITHOUT-DAG: mov.b32 [[PROXY:%r[0-9]+]], [[LD]];
  ; PTX-WITHOUT-DAG: and.b32 [[RES:%r[0-9]+]], [[PROXY]], 1;
  ; PTX-WITH-DAG:    and.b32 [[RES:%r[0-9]+]], [[LD]], 1;

  ; PTX-DAG: st.param.b32 [func_retval0], [[RES]];

  %ret = call i1 @callee_i1()
  ret i1 %ret
}

declare i16 @callee_i16()
define  i16 @check_i16() {
  ; PTX-LABEL: check_i16
  ; PTX-DAG: { // callseq {{[0-9]+}}, {{[0-9]+}}
  ; PTX-DAG: ld.param.b32 [[LD:%r[0-9]+]], [retval0];
  ; PTX-DAG: } // callseq {{[0-9]+}}

  ; PTX-WITHOUT-DAG: mov.b32 [[PROXY:%r[0-9]+]], [[LD]];
  ; PTX-WITHOUT-DAG: and.b32 [[RES:%r[0-9]+]], [[PROXY]], 65535;
  ; PTX-WITH-DAG:    and.b32 [[RES:%r[0-9]+]], [[LD]], 65535;

  ; PTX-DAG: st.param.b32 [func_retval0], [[RES]];

  %ret = call i16 @callee_i16()
  ret i16 %ret
}

declare i32 @callee_i32()
define  i32 @check_i32() {
  ; PTX-LABEL: check_i32
  ; PTX-DAG: { // callseq {{[0-9]+}}, {{[0-9]+}}
  ; PTX-DAG: ld.param.b32 [[LD:%r[0-9]+]], [retval0];
  ; PTX-DAG: } // callseq {{[0-9]+}}

  ; PTX-WITHOUT-DAG: mov.b32 [[PROXY:%r[0-9]+]], [[LD]];
  ; PTX-WITHOUT-DAG: st.param.b32 [func_retval0], [[PROXY]];
  ; PTX-WITH-DAG:    st.param.b32 [func_retval0], [[LD]];

  %ret = call i32 @callee_i32()
  ret i32 %ret
}

declare i64 @callee_i64()
define  i64 @check_i64() {
  ; PTX-LABEL: check_i64
  ; PTX-DAG: { // callseq {{[0-9]+}}, {{[0-9]+}}
  ; PTX-DAG: ld.param.b64 [[LD:%rd[0-9]+]], [retval0];
  ; PTX-DAG: } // callseq {{[0-9]+}}

  ; PTX-WITHOUT-DAG: mov.b64 [[PROXY:%rd[0-9]+]], [[LD]];
  ; PTX-WITHOUT-DAG: st.param.b64 [func_retval0], [[PROXY]];
  ; PTX-WITH-DAG:    st.param.b64 [func_retval0], [[LD]];

  %ret = call i64 @callee_i64()
  ret i64 %ret
}

declare i128 @callee_i128()
define  i128 @check_i128() {
  ; PTX-LABEL: check_i128
  ; PTX-DAG: { // callseq {{[0-9]+}}, {{[0-9]+}}
  ; PTX-DAG: ld.param.v2.b64 {[[LD0:%rd[0-9]+]], [[LD1:%rd[0-9]+]]}, [retval0];
  ; PTX-DAG: } // callseq {{[0-9]+}}

  ; PTX-WITHOUT-DAG: mov.b64 [[PROXY0:%rd[0-9]+]], [[LD0]];
  ; PTX-WITHOUT-DAG: mov.b64 [[PROXY1:%rd[0-9]+]], [[LD1]];
  ; PTX-WITHOUT-DAG: st.param.v2.b64 [func_retval0], {[[PROXY0]], [[PROXY1]]};
  ; PTX-WITH-DAG:    st.param.v2.b64 [func_retval0], {[[LD0]], [[LD1]]};

  %ret = call i128 @callee_i128()
  ret i128 %ret
}

declare half @callee_f16()
define  half @check_f16() {
  ; PTX-LABEL: check_f16
  ; PTX-DAG: { // callseq {{[0-9]+}}, {{[0-9]+}}
  ; PTX-DAG: ld.param.b16 [[LD:%rs[0-9]+]], [retval0];
  ; PTX-DAG: } // callseq {{[0-9]+}}

  ; PTX-WITHOUT-DAG: mov.b16 [[PROXY:%rs[0-9]+]], [[LD]];
  ; PTX-WITHOUT-DAG: st.param.b16 [func_retval0], [[PROXY]];
  ; PTX-WITH-DAG:    st.param.b16 [func_retval0], [[LD]];

  %ret = call half @callee_f16()
  ret half %ret
}

declare float @callee_f32()
define  float @check_f32() {
  ; PTX-LABEL: check_f32
  ; PTX-DAG: { // callseq {{[0-9]+}}, {{[0-9]+}}
  ; PTX-DAG: ld.param.b32 [[LD:%r[0-9]+]], [retval0];
  ; PTX-DAG: } // callseq {{[0-9]+}}

  ; PTX-WITHOUT-DAG: mov.b32 [[PROXY:%r[0-9]+]], [[LD]];
  ; PTX-WITHOUT-DAG: st.param.b32 [func_retval0], [[PROXY]];
  ; PTX-WITH-DAG:    st.param.b32 [func_retval0], [[LD]];

  %ret = call float @callee_f32()
  ret float %ret
}

declare double @callee_f64()
define  double @check_f64() {
  ; PTX-LABEL: check_f64
  ; PTX-DAG: { // callseq {{[0-9]+}}, {{[0-9]+}}
  ; PTX-DAG: ld.param.b64 [[LD:%rd[0-9]+]], [retval0];
  ; PTX-DAG: } // callseq {{[0-9]+}}

  ; PTX-WITHOUT-DAG: mov.b64 [[PROXY:%rd[0-9]+]], [[LD]];
  ; PTX-WITHOUT-DAG: st.param.b64 [func_retval0], [[PROXY]];
  ; PTX-WITH-DAG:    st.param.b64 [func_retval0], [[LD]];

  %ret = call double @callee_f64()
  ret double %ret
}

declare <4 x i32> @callee_vec_i32()
define  <4 x i32> @check_vec_i32() {
  ; PTX-LABEL: check_vec_i32
  ; PTX-DAG: { // callseq {{[0-9]+}}, {{[0-9]+}}
  ; PTX-DAG: ld.param.v4.b32 {[[LD0:%r[0-9]+]], [[LD1:%r[0-9]+]], [[LD2:%r[0-9]+]], [[LD3:%r[0-9]+]]}, [retval0];
  ; PTX-DAG: } // callseq {{[0-9]+}}

  ; PTX-WITHOUT-DAG: mov.b32 [[PROXY0:%r[0-9]+]], [[LD0]];
  ; PTX-WITHOUT-DAG: mov.b32 [[PROXY1:%r[0-9]+]], [[LD1]];
  ; PTX-WITHOUT-DAG: mov.b32 [[PROXY2:%r[0-9]+]], [[LD2]];
  ; PTX-WITHOUT-DAG: mov.b32 [[PROXY3:%r[0-9]+]], [[LD3]];
  ; PTX-WITHOUT-DAG: st.param.v4.b32 [func_retval0], {[[PROXY0]], [[PROXY1]], [[PROXY2]], [[PROXY3]]};
  ; PTX-WITH-DAG:    st.param.v4.b32 [func_retval0], {[[LD0]], [[LD1]], [[LD2]], [[LD3]]};

  %ret = call <4 x i32> @callee_vec_i32()
  ret <4 x i32> %ret
}

declare <2 x half> @callee_vec_f16()
define  <2 x half> @check_vec_f16() {
  ; PTX-LABEL: check_vec_f16
  ; PTX-DAG: { // callseq {{[0-9]+}}, {{[0-9]+}}
  ; PTX-DAG: ld.param.b32 [[LD:%r[0-9]+]], [retval0];
  ; PTX-DAG: } // callseq {{[0-9]+}}

  ; PTX-WITHOUT-DAG: mov.b32 [[PROXY:%r[0-9]+]], [[LD]];
  ; PTX-WITHOUT-DAG: st.param.b32 [func_retval0], [[PROXY]];
  ; PTX-WITH-DAG:    st.param.b32 [func_retval0], [[LD]];

  %ret = call <2 x half> @callee_vec_f16()
  ret <2 x half> %ret
}

declare <2 x double> @callee_vec_f64()
define  <2 x double> @check_vec_f64() {
  ; PTX-LABEL: check_vec_f64
  ; PTX-DAG: { // callseq {{[0-9]+}}, {{[0-9]+}}
  ; PTX-DAG: ld.param.v2.b64 {[[LD0:%rd[0-9]+]], [[LD1:%rd[0-9]+]]}, [retval0];
  ; PTX-DAG: } // callseq {{[0-9]+}}

  ; PTX-WITHOUT-DAG: mov.b64 [[PROXY0:%rd[0-9]+]], [[LD0]];
  ; PTX-WITHOUT-DAG: mov.b64 [[PROXY1:%rd[0-9]+]], [[LD1]];
  ; PTX-WITHOUT-DAG: st.param.v2.b64 [func_retval0], {[[PROXY0]], [[PROXY1]]};
  ; PTX-WITH-DAG:    st.param.v2.b64 [func_retval0], {[[LD0]], [[LD1]]};

  %ret = call <2 x double> @callee_vec_f64()
  ret <2 x double> %ret
}

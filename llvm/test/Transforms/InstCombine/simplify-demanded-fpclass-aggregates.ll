; RUN: opt -S -passes=instcombine < %s | FileCheck %s

; This was separated from simplify-demanded-fpclass.ll to work around
; alive2 bug.

; Basic aggregate tests to ensure this does not crash.
define nofpclass(nan) { float } @ret_nofpclass_struct_ty() {
; CHECK-LABEL: define nofpclass(nan) { float } @ret_nofpclass_struct_ty() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret { float } zeroinitializer
;
entry:
  ret { float } zeroinitializer
}

define nofpclass(nan) { float, float } @ret_nofpclass_multiple_elems_struct_ty() {
; CHECK-LABEL: define nofpclass(nan) { float, float } @ret_nofpclass_multiple_elems_struct_ty() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret { float, float } zeroinitializer
;
entry:
  ret { float, float } zeroinitializer
}

define nofpclass(nan) { <4 x float>, <4 x float> } @ret_nofpclass_vector_elems_struct_ty() {
; CHECK-LABEL: define nofpclass(nan) { <4 x float>, <4 x float> } @ret_nofpclass_vector_elems_struct_ty() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret { <4 x float>, <4 x float> } zeroinitializer
;
entry:
  ret { <4 x float>, <4 x float> } zeroinitializer
}

define nofpclass(nan) [ 5 x float ] @ret_nofpclass_array_ty() {
; CHECK-LABEL: define nofpclass(nan) [5 x float] @ret_nofpclass_array_ty() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret [5 x float] zeroinitializer
;
entry:
  ret [ 5 x float ] zeroinitializer
}

define nofpclass(nan) [ 2 x [ 5 x float ]] @ret_nofpclass_nested_array_ty() {
; CHECK-LABEL: define nofpclass(nan) [2 x [5 x float]] @ret_nofpclass_nested_array_ty() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret [2 x [5 x float]] zeroinitializer
;
entry:
  ret [ 2 x [ 5 x float ]] zeroinitializer
}

define nofpclass(pinf) { float } @ret_nofpclass_struct_ty_pinf__ninf() {
; CHECK-LABEL: define nofpclass(pinf) { float } @ret_nofpclass_struct_ty_pinf__ninf() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret { float } { float 0xFFF0000000000000 }
;
entry:
  ret { float } { float 0xFFF0000000000000 }
}

define nofpclass(pinf) { float, float } @ret_nofpclass_multiple_elems_struct_ty_pinf__ninf() {
; CHECK-LABEL: define nofpclass(pinf) { float, float } @ret_nofpclass_multiple_elems_struct_ty_pinf__ninf() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret { float, float } { float 0xFFF0000000000000, float 0xFFF0000000000000 }
;
entry:
  ret { float, float } { float 0xFFF0000000000000, float 0xFFF0000000000000 }
}

define nofpclass(pinf) { <2 x float> } @ret_nofpclass_vector_elems_struct_ty_pinf__ninf() {
; CHECK-LABEL: define nofpclass(pinf) { <2 x float> } @ret_nofpclass_vector_elems_struct_ty_pinf__ninf() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret { <2 x float> } { <2 x float> splat (float 0xFFF0000000000000) }
;
entry:
  ret { <2 x float>} { <2 x float> <float 0xFFF0000000000000, float 0xFFF0000000000000> }
}

; UTC_ARGS: --disable
; FileCheck does not like the nested square brackets.
define nofpclass(pinf) [ 1 x [ 1 x float ]] @ret_nofpclass_nested_array_ty_pinf__ninf() {
; CHECK-LABEL: @ret_nofpclass_nested_array_ty_pinf__ninf() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret {{.*}}float 0xFFF0000000000000
;
entry:
  ret [ 1 x [ 1 x float ]] [[ 1 x float ] [float 0xFFF0000000000000]]
}
; UTC_ARGS: --enable

define nofpclass(pzero) { float, float } @ret_nofpclass_multiple_elems_struct_ty_pzero__nzero() {
; CHECK-LABEL: define nofpclass(pzero) { float, float } @ret_nofpclass_multiple_elems_struct_ty_pzero__nzero() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret { float, float } { float -0.000000e+00, float -0.000000e+00 }
;
entry:
  ret { float, float } { float -0.0, float -0.0 }
}

define nofpclass(ninf) { float, float } @ret_nofpclass_multiple_elems_struct_ty_ninf__npinf() {
; CHECK-LABEL: define nofpclass(ninf) { float, float } @ret_nofpclass_multiple_elems_struct_ty_ninf__npinf() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret { float, float } { float 0x7FF0000000000000, float 0x7FF0000000000000 }
;
entry:
  ret { float, float } { float 0x7FF0000000000000, float 0x7FF0000000000000 }
}

; FIXME (should be poison): Support computeKnownFPClass() for non-zero aggregates.
define nofpclass(inf) { float, float } @ret_nofpclass_multiple_elems_struct_ty_inf__npinf() {
; CHECK-LABEL: define nofpclass(inf) { float, float } @ret_nofpclass_multiple_elems_struct_ty_inf__npinf() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret { float, float } { float 0x7FF0000000000000, float 0x7FF0000000000000 }
;
entry:
  ret { float, float } { float 0x7FF0000000000000, float 0x7FF0000000000000 }
}

; FIXME (should be poison): Support computeKnownFPClass() for non-zero aggregates.
define nofpclass(nzero) [ 1 x float ] @ret_nofpclass_multiple_elems_struct_ty_nzero_nzero() {
; CHECK-LABEL: define nofpclass(nzero) [1 x float] @ret_nofpclass_multiple_elems_struct_ty_nzero_nzero() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret [1 x float] [float -0.000000e+00]
;
entry:
  ret [ 1 x float ] [ float -0.0 ]
}

; Fold to ret %y
define nofpclass(inf) [3 x [2 x float]] @ret_float_array(i1 %cond, [3 x [2 x float]] nofpclass(nan norm zero sub) %x, [3 x [2 x float]] %y) {
; CHECK-LABEL: define nofpclass(inf) [3 x [2 x float]] @ret_float_array
; CHECK-SAME: (i1 [[COND:%.*]], [3 x [2 x float]] nofpclass(nan zero sub norm) [[X:%.*]], [3 x [2 x float]] [[Y:%.*]]) {
; CHECK-NEXT:    ret [3 x [2 x float]] [[Y]]
;
  %select = select i1 %cond, [3 x [2 x float]] %x, [3 x [2 x float]] %y
  ret [3 x [2 x float ]] %select
}

define nofpclass(nan) float @simplify_demanded_extractvalue_array(i1 %cond, [2 x float] %arg0) {
; CHECK-LABEL: define nofpclass(nan) float @simplify_demanded_extractvalue_array
; CHECK-SAME: (i1 [[COND:%.*]], [2 x float] [[ARG0:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = extractvalue [2 x float] [[ARG0]], 0
; CHECK-NEXT:    ret float [[TMP1]]
;
  %select = select i1 %cond, [2 x float] %arg0, [2 x float] [float 0x7FF8000000000000, float 0x7FF8000000000000]
  %extract = extractvalue [2 x float] %select, 0
  ret float %extract
}

define nofpclass(nan) float @simplify_demanded_extractvalue_array_partial_positive(i1 %cond, [2 x float] %arg0) {
; CHECK-LABEL: define nofpclass(nan) float @simplify_demanded_extractvalue_array_partial_positive
; CHECK-SAME: (i1 [[COND:%.*]], [2 x float] [[ARG0:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = extractvalue [2 x float] [[ARG0]], 1
; CHECK-NEXT:    [[EXTRACT:%.*]] = select i1 [[COND]], float [[TMP1]], float 0.000000e+00
; CHECK-NEXT:    ret float [[EXTRACT]]
;
  %select = select i1 %cond, [2 x float] %arg0, [2 x float] [float 0x7FF8000000000000, float 0.0]
  %extract = extractvalue [2 x float] %select, 1
  ret float %extract
}

define nofpclass(nan) float @simplify_demanded_extractvalue_array_partialnegative(i1 %cond, [2 x float] %arg0) {
; CHECK-LABEL: define nofpclass(nan) float @simplify_demanded_extractvalue_array_partialnegative
; CHECK-SAME: (i1 [[COND:%.*]], [2 x float] [[ARG0:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = extractvalue [2 x float] [[ARG0]], 1
; CHECK-NEXT:    ret float [[TMP1]]
;
  %select = select i1 %cond, [2 x float] %arg0, [2 x float] [float 0.0, float 0x7FF8000000000000]
  %extract = extractvalue [2 x float] %select, 1
  ret float %extract
}

define nofpclass(nan) float @simplify_demanded_extractvalue_struct(i1 %cond, { float, float } %arg0) {
; CHECK-LABEL: define nofpclass(nan) float @simplify_demanded_extractvalue_struct
; CHECK-SAME: (i1 [[COND:%.*]], { float, float } [[ARG0:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = extractvalue { float, float } [[ARG0]], 0
; CHECK-NEXT:    ret float [[TMP1]]
;
  %select = select i1 %cond, { float, float } %arg0, { float, float } {float 0x7FF8000000000000, float 0x7FF8000000000000}
  %extract = extractvalue { float, float } %select, 0
  ret float %extract
}

define nofpclass(inf norm sub zero) float @simplify_demanded_extractvalue_only_nan(i1 %cond, [2 x float] %arg0) {
; CHECK-LABEL: define nofpclass(inf zero sub norm) float @simplify_demanded_extractvalue_only_nan
; CHECK-SAME: (i1 [[COND:%.*]], [2 x float] [[ARG0:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = extractvalue [2 x float] [[ARG0]], 0
; CHECK-NEXT:    [[EXTRACT:%.*]] = select i1 [[COND]], float [[TMP1]], float 0x7FF8000000000000
; CHECK-NEXT:    ret float [[EXTRACT]]
;
  %select = select i1 %cond, [2 x float] %arg0, [2 x float] [float 0x7FF8000000000000, float 0x7FF8000000000000]
  %extract = extractvalue [2 x float] %select, 0
  ret float %extract
}

define nofpclass(nan inf norm sub nzero) float @simplify_demanded_extractvalue_only_pzero(i1 %cond, [2 x float] %arg0) {
; CHECK-LABEL: define nofpclass(nan inf nzero sub norm) float @simplify_demanded_extractvalue_only_pzero
; CHECK-SAME: (i1 [[COND:%.*]], [2 x float] [[ARG0:%.*]]) {
; CHECK-NEXT:    ret float 0.000000e+00
;
  %select = select i1 %cond, [2 x float] %arg0, [2 x float] [float 0x7FF8000000000000, float 0x7FF8000000000000]
  %extract = extractvalue [2 x float] %select, 0
  ret float %extract
}

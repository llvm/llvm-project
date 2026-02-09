; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Invalid but didn't fail the verifier
define void @str_denormal_fp_math_no_val() "denormal-fp-math" {
; CHECK: Function Attrs: denormal_fpenv(ieee)
; CHECK-LABEL: define void @str_denormal_fp_math_no_val(
; CHECK-SAME: ) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

; Invalid but didn't fail the verifier
define void @str_denormal_fp_math_empty_str() "denormal-fp-math"="" {
; CHECK: Function Attrs: denormal_fpenv(ieee)
; CHECK-LABEL: define void @str_denormal_fp_math_empty_str(
; CHECK-SAME: ) #[[ATTR0]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_ieee() "denormal-fp-math"="ieee" {
; CHECK: Function Attrs: denormal_fpenv(ieee)
; CHECK-LABEL: define void @str_denormal_fp_math_ieee(
; CHECK-SAME: ) #[[ATTR0]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_ieee_ieee() "denormal-fp-math"="ieee,ieee" {
; CHECK: Function Attrs: denormal_fpenv(ieee)
; CHECK-LABEL: define void @str_denormal_fp_math_ieee_ieee(
; CHECK-SAME: ) #[[ATTR0]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_preserve_sign() "denormal-fp-math"="preserve-sign" {
; CHECK: Function Attrs: denormal_fpenv(preservesign)
; CHECK-LABEL: define void @str_denormal_fp_math_preserve_sign(
; CHECK-SAME: ) #[[ATTR1:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_preserve_sign_preserve_sign() "denormal-fp-math"="preserve-sign,preserve-sign" {
; CHECK: Function Attrs: denormal_fpenv(preservesign)
; CHECK-LABEL: define void @str_denormal_fp_math_preserve_sign_preserve_sign(
; CHECK-SAME: ) #[[ATTR1]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_dynamic() "denormal-fp-math"="dynamic" {
; CHECK: Function Attrs: denormal_fpenv(dynamic)
; CHECK-LABEL: define void @str_denormal_fp_math_dynamic(
; CHECK-SAME: ) #[[ATTR2:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_dynamic_dynamic() "denormal-fp-math"="dynamic,dynamic" {
; CHECK: Function Attrs: denormal_fpenv(dynamic)
; CHECK-LABEL: define void @str_denormal_fp_math_dynamic_dynamic(
; CHECK-SAME: ) #[[ATTR2]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_positivezero() "denormal-fp-math"="positive-zero" {
; CHECK: Function Attrs: denormal_fpenv(positivezero)
; CHECK-LABEL: define void @str_denormal_fp_math_positivezero(
; CHECK-SAME: ) #[[ATTR3:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_positivezero_positivezero() "denormal-fp-math"="positive-zero,positive-zero" {
; CHECK: Function Attrs: denormal_fpenv(positivezero)
; CHECK-LABEL: define void @str_denormal_fp_math_positivezero_positivezero(
; CHECK-SAME: ) #[[ATTR3]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_ieee_preservesign() "denormal-fp-math"="ieee,preserve-sign" {
; CHECK: Function Attrs: denormal_fpenv(ieee|preservesign)
; CHECK-LABEL: define void @str_denormal_fp_math_ieee_preservesign(
; CHECK-SAME: ) #[[ATTR4:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_preservesign_ieee() "denormal-fp-math"="preserve-sign,ieee" {
; CHECK: Function Attrs: denormal_fpenv(preservesign|ieee)
; CHECK-LABEL: define void @str_denormal_fp_math_preservesign_ieee(
; CHECK-SAME: ) #[[ATTR5:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}


; Invalid but didn't fail the verifier
define void @str_denormal_fp_math_f32_no_val() "denormal-fp-math-f32" {
; CHECK: Function Attrs: denormal_fpenv(ieee)
; CHECK-LABEL: define void @str_denormal_fp_math_f32_no_val(
; CHECK-SAME: ) #[[ATTR0]] {
; CHECK-NEXT:    ret void
;
  ret void
}

; Invalid but didn't fail the verifier
define void @str_denormal_fp_math_f32_empty_str() "denormal-fp-math-f32"="" {
; CHECK: Function Attrs: denormal_fpenv(ieee)
; CHECK-LABEL: define void @str_denormal_fp_math_f32_empty_str(
; CHECK-SAME: ) #[[ATTR0]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_f32_ieee() "denormal-fp-math-f32"="ieee" {
; CHECK: Function Attrs: denormal_fpenv(ieee)
; CHECK-LABEL: define void @str_denormal_fp_math_f32_ieee(
; CHECK-SAME: ) #[[ATTR0]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_f32_ieee_ieee() "denormal-fp-math-f32"="ieee,ieee" {
; CHECK: Function Attrs: denormal_fpenv(ieee)
; CHECK-LABEL: define void @str_denormal_fp_math_f32_ieee_ieee(
; CHECK-SAME: ) #[[ATTR0]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_f32_preserve_sign() "denormal-fp-math-f32"="preserve-sign" {
; CHECK: Function Attrs: denormal_fpenv(float: preservesign)
; CHECK-LABEL: define void @str_denormal_fp_math_f32_preserve_sign(
; CHECK-SAME: ) #[[ATTR6:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_f32_preserve_sign_preserve_sign() "denormal-fp-math-f32"="preserve-sign,preserve-sign" {
; CHECK: Function Attrs: denormal_fpenv(float: preservesign)
; CHECK-LABEL: define void @str_denormal_fp_math_f32_preserve_sign_preserve_sign(
; CHECK-SAME: ) #[[ATTR6]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_f32_dynamic() "denormal-fp-math-f32"="dynamic" {
; CHECK: Function Attrs: denormal_fpenv(float: dynamic)
; CHECK-LABEL: define void @str_denormal_fp_math_f32_dynamic(
; CHECK-SAME: ) #[[ATTR7:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_f32_dynamic_dynamic() "denormal-fp-math-f32"="dynamic,dynamic" {
; CHECK: Function Attrs: denormal_fpenv(float: dynamic)
; CHECK-LABEL: define void @str_denormal_fp_math_f32_dynamic_dynamic(
; CHECK-SAME: ) #[[ATTR7]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_f32_positivezero() "denormal-fp-math-f32"="positive-zero" {
; CHECK: Function Attrs: denormal_fpenv(float: positivezero)
; CHECK-LABEL: define void @str_denormal_fp_math_f32_positivezero(
; CHECK-SAME: ) #[[ATTR8:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_f32_positivezero_positivezero() "denormal-fp-math-f32"="positive-zero,positive-zero" {
; CHECK: Function Attrs: denormal_fpenv(float: positivezero)
; CHECK-LABEL: define void @str_denormal_fp_math_f32_positivezero_positivezero(
; CHECK-SAME: ) #[[ATTR8]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_f32_ieee_preservesign() "denormal-fp-math-f32"="ieee,preserve-sign" {
; CHECK: Function Attrs: denormal_fpenv(float: ieee|preservesign)
; CHECK-LABEL: define void @str_denormal_fp_math_f32_ieee_preservesign(
; CHECK-SAME: ) #[[ATTR9:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_f32_preservesign_ieee() "denormal-fp-math-f32"="preserve-sign,ieee" {
; CHECK: Function Attrs: denormal_fpenv(float: preservesign|ieee)
; CHECK-LABEL: define void @str_denormal_fp_math_f32_preservesign_ieee(
; CHECK-SAME: ) #[[ATTR10:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_ieee__denormal_fp_math_f32_ieee() "denormal-fp-math"="ieee" "denormal-fp-math-f32"="ieee" {
; CHECK: Function Attrs: denormal_fpenv(ieee)
; CHECK-LABEL: define void @str_denormal_fp_math_ieee__denormal_fp_math_f32_ieee(
; CHECK-SAME: ) #[[ATTR0]] {
; CHECK-NEXT:    ret void
;
  ret void
}


define void @str_denormal_fp_math_ieee__denormal_fp_math_f32_preserve_sign() "denormal-fp-math"="ieee" "denormal-fp-math-f32"="preserve-sign" {
; CHECK: Function Attrs: denormal_fpenv(float: preservesign)
; CHECK-LABEL: define void @str_denormal_fp_math_ieee__denormal_fp_math_f32_preserve_sign(
; CHECK-SAME: ) #[[ATTR6]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_ieee_ieee__denormal_fp_math_f32_preserve_sign_preserve_sign() "denormal-fp-math"="ieee,ieee" "denormal-fp-math-f32"="preserve-sign,preserve-sign" {
; CHECK: Function Attrs: denormal_fpenv(float: preservesign)
; CHECK-LABEL: define void @str_denormal_fp_math_ieee_ieee__denormal_fp_math_f32_preserve_sign_preserve_sign(
; CHECK-SAME: ) #[[ATTR6]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_ieee_ieee__denormal_fp_math_f32_preserve_sign_dynamic_dynamic() "denormal-fp-math"="ieee,ieee" "denormal-fp-math-f32"="dynamic,dynamic" {
; CHECK: Function Attrs: denormal_fpenv(float: dynamic)
; CHECK-LABEL: define void @str_denormal_fp_math_ieee_ieee__denormal_fp_math_f32_preserve_sign_dynamic_dynamic(
; CHECK-SAME: ) #[[ATTR7]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_dynamic_dynamic__denormal_fp_math_f32_preserve_sign_dynamic_dynamic() "denormal-fp-math"="dynamic,dynamic" "denormal-fp-math-f32"="dynamic,dynamic" {
; CHECK: Function Attrs: denormal_fpenv(dynamic)
; CHECK-LABEL: define void @str_denormal_fp_math_dynamic_dynamic__denormal_fp_math_f32_preserve_sign_dynamic_dynamic(
; CHECK-SAME: ) #[[ATTR2]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_dynamic_dynamic__denormal_fp_math_f32_preserve_sign_preserve_sign() "denormal-fp-math"="dynamic,dynamic" "denormal-fp-math-f32"="preserve-sign,preserve-sign" {
; CHECK: Function Attrs: denormal_fpenv(dynamic, float: preservesign)
; CHECK-LABEL: define void @str_denormal_fp_math_dynamic_dynamic__denormal_fp_math_f32_preserve_sign_preserve_sign(
; CHECK-SAME: ) #[[ATTR11:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @str_denormal_fp_math_dynamic_positive_zero__denormal_fp_math_f32_preserve_sign_dynamic_preserve_sign() "denormal-fp-math"="dynamic,positive-zero" "denormal-fp-math-f32"="dynamic,preserve-sign" {
; CHECK: Function Attrs: denormal_fpenv(dynamic|positivezero, float: dynamic|preservesign)
; CHECK-LABEL: define void @str_denormal_fp_math_dynamic_positive_zero__denormal_fp_math_f32_preserve_sign_dynamic_preserve_sign(
; CHECK-SAME: ) #[[ATTR12:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

declare void @func()

; These never did anything, but the new attribute fails the verifier
; on call sites.
define void @ignore_callsite_attrs() {
; CHECK-LABEL: define void @ignore_callsite_attrs() {
; CHECK-NEXT:    call void @func() #[[ATTR19:[0-9]+]]
; CHECK-NEXT:    call void @func() #[[ATTR20:[0-9]+]]
; CHECK-NEXT:    call void @func() #[[ATTR21:[0-9]+]]
; CHECK-NEXT:    ret void
;
  call void @func() "denormal-fp-math"="preserve-sign,preserve-sign"
  call void @func() "denormal-fp-math-f32"="preserve-sign,preserve-sign"
  call void @func() "denormal-fp-math"="dynamic,ieee" "denormal-fp-math-f32"="preserve-sign,preserve-sign"
  ret void
}

define float @test_denormal_fp_math_invalid1() "denormal-fp-math"="foo,ieee" {
; CHECK-LABEL: define float @test_denormal_fp_math_invalid1(
; CHECK-SAME: ) #[[ATTR13:[0-9]+]] {
; CHECK-NEXT:    ret float 1.000000e+00
;
  ret float 1.0
}

define float @test_denormal_fp_math_invalid2() "denormal-fp-math"="ieee,ieee,ieee" {
; CHECK-LABEL: define float @test_denormal_fp_math_invalid2(
; CHECK-SAME: ) #[[ATTR14:[0-9]+]] {
; CHECK-NEXT:    ret float 1.000000e+00
;
  ret float 1.0
}

define float @test_denormal_fp_math_f32_invalid() "denormal-fp-math-f32"="foo,ieee" {
; CHECK-LABEL: define float @test_denormal_fp_math_f32_invalid(
; CHECK-SAME: ) #[[ATTR15:[0-9]+]] {
; CHECK-NEXT:    ret float 1.000000e+00
;
  ret float 1.0
}

define float @test_both_denormal_fp_math_invalid() "denormal-fp-math"="bar" "denormal-fp-math-f32"="foo,ieee" {
; CHECK-LABEL: define float @test_both_denormal_fp_math_invalid(
; CHECK-SAME: ) #[[ATTR16:[0-9]+]] {
; CHECK-NEXT:    ret float 1.000000e+00
;
  ret float 1.0
}

define float @test_denormal_fp_math_invalid_with_invalid_f32() "denormal-fp-math"="dynamic,dynamic" "denormal-fp-math-f32"="foo,ieee" {
; CHECK: Function Attrs: denormal_fpenv(dynamic)
; CHECK-LABEL: define float @test_denormal_fp_math_invalid_with_invalid_f32(
; CHECK-SAME: ) #[[ATTR17:[0-9]+]] {
; CHECK-NEXT:    ret float 1.000000e+00
;
  ret float 1.0
}

define float @test_invalid_denormal_fp_math_with_valid_f32() "denormal-fp-math"="foo,dynamic" "denormal-fp-math-f32"="dynamic,dynamic" {
; CHECK: Function Attrs: denormal_fpenv(float: dynamic)
; CHECK-LABEL: define float @test_invalid_denormal_fp_math_with_valid_f32(
; CHECK-SAME: ) #[[ATTR18:[0-9]+]] {
; CHECK-NEXT:    ret float 1.000000e+00
;
  ret float 1.0
}

;.
; CHECK: attributes #[[ATTR0]] = { denormal_fpenv(ieee) }
; CHECK: attributes #[[ATTR1]] = { denormal_fpenv(preservesign) }
; CHECK: attributes #[[ATTR2]] = { denormal_fpenv(dynamic) }
; CHECK: attributes #[[ATTR3]] = { denormal_fpenv(positivezero) }
; CHECK: attributes #[[ATTR4]] = { denormal_fpenv(ieee|preservesign) }
; CHECK: attributes #[[ATTR5]] = { denormal_fpenv(preservesign|ieee) }
; CHECK: attributes #[[ATTR6]] = { denormal_fpenv(float: preservesign) }
; CHECK: attributes #[[ATTR7]] = { denormal_fpenv(float: dynamic) }
; CHECK: attributes #[[ATTR8]] = { denormal_fpenv(float: positivezero) }
; CHECK: attributes #[[ATTR9]] = { denormal_fpenv(float: ieee|preservesign) }
; CHECK: attributes #[[ATTR10]] = { denormal_fpenv(float: preservesign|ieee) }
; CHECK: attributes #[[ATTR11]] = { denormal_fpenv(dynamic, float: preservesign) }
; CHECK: attributes #[[ATTR12]] = { denormal_fpenv(dynamic|positivezero, float: dynamic|preservesign) }
; CHECK: attributes #[[ATTR13]] = { "denormal-fp-math"="foo,ieee" }
; CHECK: attributes #[[ATTR14]] = { "denormal-fp-math"="ieee,ieee,ieee" }
; CHECK: attributes #[[ATTR15]] = { "denormal-fp-math-f32"="foo,ieee" }
; CHECK: attributes #[[ATTR16]] = { "denormal-fp-math"="bar" "denormal-fp-math-f32"="foo,ieee" }
; CHECK: attributes #[[ATTR17]] = { denormal_fpenv(dynamic) "denormal-fp-math-f32"="foo,ieee" }
; CHECK: attributes #[[ATTR18]] = { denormal_fpenv(float: dynamic) "denormal-fp-math"="foo,dynamic" }
; CHECK: attributes #[[ATTR19]] = { "denormal-fp-math"="preserve-sign,preserve-sign" }
; CHECK: attributes #[[ATTR20]] = { "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
; CHECK: attributes #[[ATTR21]] = { "denormal-fp-math"="dynamic,ieee" "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
;.

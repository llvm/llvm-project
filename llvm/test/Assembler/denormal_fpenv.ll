; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

define void @func_ieee() denormal_fpenv(ieee) {
; CHECK: Function Attrs: denormal_fpenv(ieee)
; CHECK-LABEL: define void @func_ieee(
; CHECK-SAME: ) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_preservesign() denormal_fpenv(preservesign) {
; CHECK: Function Attrs: denormal_fpenv(preservesign)
; CHECK-LABEL: define void @func_preservesign(
; CHECK-SAME: ) #[[ATTR1:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_positivezero() denormal_fpenv(positivezero) {
; CHECK: Function Attrs: denormal_fpenv(positivezero)
; CHECK-LABEL: define void @func_positivezero(
; CHECK-SAME: ) #[[ATTR2:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_dynamic() denormal_fpenv(dynamic) {
; CHECK: Function Attrs: denormal_fpenv(dynamic)
; CHECK-LABEL: define void @func_dynamic(
; CHECK-SAME: ) #[[ATTR3:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_ieee_ieee() denormal_fpenv(ieee|ieee) {
; CHECK: Function Attrs: denormal_fpenv(ieee)
; CHECK-LABEL: define void @func_ieee_ieee(
; CHECK-SAME: ) #[[ATTR0]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_preservesign_preservesign() denormal_fpenv(preservesign|preservesign) {
; CHECK: Function Attrs: denormal_fpenv(preservesign)
; CHECK-LABEL: define void @func_preservesign_preservesign(
; CHECK-SAME: ) #[[ATTR1]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_positivezero_positivezero() denormal_fpenv(positivezero|positivezero) {
; CHECK: Function Attrs: denormal_fpenv(positivezero)
; CHECK-LABEL: define void @func_positivezero_positivezero(
; CHECK-SAME: ) #[[ATTR2]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_dynamic_dynamic() denormal_fpenv(dynamic|dynamic) {
; CHECK: Function Attrs: denormal_fpenv(dynamic)
; CHECK-LABEL: define void @func_dynamic_dynamic(
; CHECK-SAME: ) #[[ATTR3]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_ieee_preservesign() denormal_fpenv(ieee|preservesign) {
; CHECK-LABEL: define void @func_ieee_preservesign(
; CHECK-SAME: ) #[[ATTR4:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_preservesign_ieee() denormal_fpenv(preservesign|ieee) {
; CHECK-LABEL: define void @func_preservesign_ieee(
; CHECK-SAME: ) #[[ATTR5:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_ieee_dynamic() denormal_fpenv(ieee|dynamic) {
; CHECK-LABEL: define void @func_ieee_dynamic(
; CHECK-SAME: ) #[[ATTR6:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_dynamic_ieee() denormal_fpenv(dynamic|ieee) {
; CHECK-LABEL: define void @func_dynamic_ieee(
; CHECK-SAME: ) #[[ATTR7:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_ieee_positivezero() denormal_fpenv(ieee|positivezero) {
; CHECK-LABEL: define void @func_ieee_positivezero(
; CHECK-SAME: ) #[[ATTR8:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_positivezero_ieee() denormal_fpenv(positivezero|ieee) {
; CHECK-LABEL: define void @func_positivezero_ieee(
; CHECK-SAME: ) #[[ATTR9:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_positivezero_dynamic() denormal_fpenv(positivezero|dynamic) {
; CHECK-LABEL: define void @func_positivezero_dynamic(
; CHECK-SAME: ) #[[ATTR10:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_dynamic_positivezero() denormal_fpenv(dynamic|positivezero) {
; CHECK-LABEL: define void @func_dynamic_positivezero(
; CHECK-SAME: ) #[[ATTR11:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_attr_group0() #0 {
; CHECK-LABEL: define void @func_attr_group0(
; CHECK-SAME: ) #[[ATTR12:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_attr_group1() #1 {
; CHECK: Function Attrs: denormal_fpenv(preservesign)
; CHECK-LABEL: define void @func_attr_group1(
; CHECK-SAME: ) #[[ATTR1]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_f32_ieee() denormal_fpenv(float:ieee) {
; CHECK: Function Attrs: denormal_fpenv(ieee)
; CHECK-LABEL: define void @func_f32_ieee(
; CHECK-SAME: ) #[[ATTR0]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_f32_preservesign() denormal_fpenv(float: preservesign) {
; CHECK: Function Attrs: denormal_fpenv(float: preservesign)
; CHECK-LABEL: define void @func_f32_preservesign(
; CHECK-SAME: ) #[[ATTR13:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_f32_positivezero() denormal_fpenv(float: positivezero) {
; CHECK: Function Attrs: denormal_fpenv(float: positivezero)
; CHECK-LABEL: define void @func_f32_positivezero(
; CHECK-SAME: ) #[[ATTR14:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_f32_dynamic() denormal_fpenv(float: dynamic) {
; CHECK: Function Attrs: denormal_fpenv(float: dynamic)
; CHECK-LABEL: define void @func_f32_dynamic(
; CHECK-SAME: ) #[[ATTR15:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_f32_ieee_ieee() denormal_fpenv(float: ieee|ieee) {
; CHECK: Function Attrs: denormal_fpenv(ieee)
; CHECK-LABEL: define void @func_f32_ieee_ieee(
; CHECK-SAME: ) #[[ATTR0]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_f32_preservesign_preservesign() denormal_fpenv(float: preservesign|preservesign) {
; CHECK: Function Attrs: denormal_fpenv(float: preservesign)
; CHECK-LABEL: define void @func_f32_preservesign_preservesign(
; CHECK-SAME: ) #[[ATTR13]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_f32_positivezero_positivezero() denormal_fpenv(float: positivezero|positivezero) {
; CHECK: Function Attrs: denormal_fpenv(float: positivezero)
; CHECK-LABEL: define void @func_f32_positivezero_positivezero(
; CHECK-SAME: ) #[[ATTR14]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_f32_dynamic_dynamic() denormal_fpenv(float: dynamic|dynamic) {
; CHECK: Function Attrs: denormal_fpenv(float: dynamic)
; CHECK-LABEL: define void @func_f32_dynamic_dynamic(
; CHECK-SAME: ) #[[ATTR15]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_f32_preservesign_ieee() denormal_fpenv(float: preservesign|ieee) {
; CHECK-LABEL: define void @func_f32_preservesign_ieee(
; CHECK-SAME: ) #[[ATTR16:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_f32_dynamic_positivezero() denormal_fpenv(float:dynamic|positivezero) {
; CHECK-LABEL: define void @func_f32_dynamic_positivezero(
; CHECK-SAME: ) #[[ATTR17:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_f64_dynamic__f32_ieee() denormal_fpenv(dynamic, float:ieee) {
; CHECK: Function Attrs: denormal_fpenv(dynamic, float: ieee)
; CHECK-LABEL: define void @func_f64_dynamic__f32_ieee(
; CHECK-SAME: ) #[[ATTR18:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_f64_dynamic__f32_preservesign() denormal_fpenv(dynamic, float:preservesign) {
; CHECK: Function Attrs: denormal_fpenv(dynamic, float: preservesign)
; CHECK-LABEL: define void @func_f64_dynamic__f32_preservesign(
; CHECK-SAME: ) #[[ATTR19:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

define void @func_f64_dynamic_ieee__f32_preservesign_ieee() denormal_fpenv(dynamic|ieee, float:preservesign|ieee) {
;
; CHECK-LABEL: define void @func_f64_dynamic_ieee__f32_preservesign_ieee(
; CHECK-SAME: ) #[[ATTR20:[0-9]+]] {
; CHECK-NEXT:    ret void
;
  ret void
}

attributes #0 = { denormal_fpenv(preservesign|dynamic) }
attributes #1 = { denormal_fpenv(preservesign) }
;.
; CHECK: attributes #[[ATTR0]] = { denormal_fpenv(ieee) }
; CHECK: attributes #[[ATTR1]] = { denormal_fpenv(preservesign) }
; CHECK: attributes #[[ATTR2]] = { denormal_fpenv(positivezero) }
; CHECK: attributes #[[ATTR3]] = { denormal_fpenv(dynamic) }
; CHECK: attributes #[[ATTR4]] = { denormal_fpenv(ieee|preservesign) }
; CHECK: attributes #[[ATTR5]] = { denormal_fpenv(preservesign|ieee) }
; CHECK: attributes #[[ATTR6]] = { denormal_fpenv(ieee|dynamic) }
; CHECK: attributes #[[ATTR7]] = { denormal_fpenv(dynamic|ieee) }
; CHECK: attributes #[[ATTR8]] = { denormal_fpenv(ieee|positivezero) }
; CHECK: attributes #[[ATTR9]] = { denormal_fpenv(positivezero|ieee) }
; CHECK: attributes #[[ATTR10]] = { denormal_fpenv(positivezero|dynamic) }
; CHECK: attributes #[[ATTR11]] = { denormal_fpenv(dynamic|positivezero) }
; CHECK: attributes #[[ATTR12]] = { denormal_fpenv(preservesign|dynamic) }
; CHECK: attributes #[[ATTR13]] = { denormal_fpenv(float: preservesign) }
; CHECK: attributes #[[ATTR14]] = { denormal_fpenv(float: positivezero) }
; CHECK: attributes #[[ATTR15]] = { denormal_fpenv(float: dynamic) }
; CHECK: attributes #[[ATTR16]] = { denormal_fpenv(float: preservesign|ieee) }
; CHECK: attributes #[[ATTR17]] = { denormal_fpenv(float: dynamic|positivezero) }
; CHECK: attributes #[[ATTR18]] = { denormal_fpenv(dynamic, float: ieee) }
; CHECK: attributes #[[ATTR19]] = { denormal_fpenv(dynamic, float: preservesign) }
; CHECK: attributes #[[ATTR20]] = { denormal_fpenv(dynamic|ieee, float: preservesign|ieee) }
;.

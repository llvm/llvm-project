; Extracted from test/CodeGen/Generic/vector-casts.ll: used to loop indefinitely.
; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: convert_df2w

define void @a(ptr %p, ptr %q) {
  %t = load <2 x double>, ptr %p
  %r = fptosi <2 x double> %t to <2 x i8>
  store <2 x i8> %r, ptr %q
  ret void
}

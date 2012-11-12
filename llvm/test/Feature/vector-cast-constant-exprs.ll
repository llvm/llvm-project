; RUN: llvm-as < %s | llvm-dis | not grep "ret.*("

; All of these constant expressions should fold.

define <2 x float> @ga() {
  ret <2 x float> fptrunc (<2 x double><double 4.3, double 3.2> to <2 x float>)
}
define <2 x double> @gb() {
  ret <2 x double> fpext (<2 x float><float 2.0, float 8.0> to <2 x double>)
}
define <2 x i64> @gd() {
  ret <2 x i64> zext (<2 x i32><i32 3, i32 4> to <2 x i64>)
}
define <2 x i64> @ge() {
  ret <2 x i64> sext (<2 x i32><i32 3, i32 4> to <2 x i64>)
}
define <2 x i32> @gf() {
  ret <2 x i32> trunc (<2 x i64><i64 3, i64 4> to <2 x i32>)
}
define <2 x i32> @gh() {
  ret <2 x i32> fptoui (<2 x float><float 8.0, float 7.0> to <2 x i32>)
}
define <2 x i32> @gi() {
  ret <2 x i32> fptosi (<2 x float><float 8.0, float 7.0> to <2 x i32>)
}
define <2 x float> @gj() {
  ret <2 x float> uitofp (<2 x i32><i32 8, i32 7> to <2 x float>)
}
define <2 x float> @gk() {
  ret <2 x float> sitofp (<2 x i32><i32 8, i32 7> to <2 x float>)
}
define <2 x double> @gl() {
  ret <2 x double> bitcast (<2 x double><double 4.0, double 3.0> to <2 x double>)
}
define <2 x double> @gm() {
  ret <2 x double> bitcast (<2 x i64><i64 4, i64 3> to <2 x double>)
}

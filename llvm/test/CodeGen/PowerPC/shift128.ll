; RUN: llc < %s -march=ppc64 | grep sld | count 5

define i128 @foo_lshr(i128 %x, i128 %y) {
  %r = lshr i128 %x, %y
  ret i128 %r
}
define i128 @foo_ashr(i128 %x, i128 %y) {
  %r = ashr i128 %x, %y
  ret i128 %r
}
define i128 @foo_shl(i128 %x, i128 %y) {
  %r = shl i128 %x, %y
  ret i128 %r
}

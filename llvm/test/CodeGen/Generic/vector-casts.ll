; RUN: llc < %s
; PR2671

define void @a(ptr %p, ptr %q) {
  %t = load <2 x double>, ptr %p
  %r = fptosi <2 x double> %t to <2 x i8>
  store <2 x i8> %r, ptr %q
  ret void
}
define void @b(ptr %p, ptr %q) {
  %t = load <2 x double>, ptr %p
  %r = fptoui <2 x double> %t to <2 x i8>
  store <2 x i8> %r, ptr %q
  ret void
}
define void @c(ptr %p, ptr %q) {
  %t = load <2 x i8>, ptr %p
  %r = sitofp <2 x i8> %t to <2 x double>
  store <2 x double> %r, ptr %q
  ret void
}
define void @d(ptr %p, ptr %q) {
  %t = load <2 x i8>, ptr %p
  %r = uitofp <2 x i8> %t to <2 x double>
  store <2 x double> %r, ptr %q
  ret void
}
define void @e(ptr %p, ptr %q) {
  %t = load <2 x i8>, ptr %p
  %r = sext <2 x i8> %t to <2 x i16>
  store <2 x i16> %r, ptr %q
  ret void
}
define void @f(ptr %p, ptr %q) {
  %t = load <2 x i8>, ptr %p
  %r = zext <2 x i8> %t to <2 x i16>
  store <2 x i16> %r, ptr %q
  ret void
}
define void @g(ptr %p, ptr %q) {
  %t = load <2 x i16>, ptr %p
  %r = trunc <2 x i16> %t to <2 x i8>
  store <2 x i8> %r, ptr %q
  ret void
}

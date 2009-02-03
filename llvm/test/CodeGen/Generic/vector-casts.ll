; RUN: llvm-as < %s | llc
; PR2671

define void @a(<2 x double>* %p, <2 x i8>* %q) {
        %t = load <2 x double>* %p
	%r = fptosi <2 x double> %t to <2 x i8>
        store <2 x i8> %r, <2 x i8>* %q
	ret void
}
define void @b(<2 x double>* %p, <2 x i8>* %q) {
        %t = load <2 x double>* %p
	%r = fptoui <2 x double> %t to <2 x i8>
        store <2 x i8> %r, <2 x i8>* %q
	ret void
}
define void @c(<2 x i8>* %p, <2 x double>* %q) {
        %t = load <2 x i8>* %p
	%r = sitofp <2 x i8> %t to <2 x double>
        store <2 x double> %r, <2 x double>* %q
	ret void
}
define void @d(<2 x i8>* %p, <2 x double>* %q) {
        %t = load <2 x i8>* %p
	%r = uitofp <2 x i8> %t to <2 x double>
        store <2 x double> %r, <2 x double>* %q
	ret void
}
define void @e(<2 x i8>* %p, <2 x i16>* %q) {
        %t = load <2 x i8>* %p
	%r = sext <2 x i8> %t to <2 x i16>
        store <2 x i16> %r, <2 x i16>* %q
	ret void
}
define void @f(<2 x i8>* %p, <2 x i16>* %q) {
        %t = load <2 x i8>* %p
	%r = zext <2 x i8> %t to <2 x i16>
        store <2 x i16> %r, <2 x i16>* %q
	ret void
}
define void @g(<2 x i16>* %p, <2 x i8>* %q) {
        %t = load <2 x i16>* %p
	%r = trunc <2 x i16> %t to <2 x i8>
        store <2 x i8> %r, <2 x i8>* %q
	ret void
}

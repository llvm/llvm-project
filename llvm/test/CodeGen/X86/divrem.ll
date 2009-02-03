; RUN: llvm-as < %s | llc -march=x86-64 | grep div | count 8

define void @si64(i64 %x, i64 %y, i64* %p, i64* %q) {
	%r = sdiv i64 %x, %y
	%t = srem i64 %x, %y
	store i64 %r, i64* %p
	store i64 %t, i64* %q
	ret void
}
define void @si32(i32 %x, i32 %y, i32* %p, i32* %q) {
	%r = sdiv i32 %x, %y
	%t = srem i32 %x, %y
	store i32 %r, i32* %p
	store i32 %t, i32* %q
	ret void
}
define void @si16(i16 %x, i16 %y, i16* %p, i16* %q) {
	%r = sdiv i16 %x, %y
	%t = srem i16 %x, %y
	store i16 %r, i16* %p
	store i16 %t, i16* %q
	ret void
}
define void @si8(i8 %x, i8 %y, i8* %p, i8* %q) {
	%r = sdiv i8 %x, %y
	%t = srem i8 %x, %y
	store i8 %r, i8* %p
	store i8 %t, i8* %q
	ret void
}
define void @ui64(i64 %x, i64 %y, i64* %p, i64* %q) {
	%r = udiv i64 %x, %y
	%t = urem i64 %x, %y
	store i64 %r, i64* %p
	store i64 %t, i64* %q
	ret void
}
define void @ui32(i32 %x, i32 %y, i32* %p, i32* %q) {
	%r = udiv i32 %x, %y
	%t = urem i32 %x, %y
	store i32 %r, i32* %p
	store i32 %t, i32* %q
	ret void
}
define void @ui16(i16 %x, i16 %y, i16* %p, i16* %q) {
	%r = udiv i16 %x, %y
	%t = urem i16 %x, %y
	store i16 %r, i16* %p
	store i16 %t, i16* %q
	ret void
}
define void @ui8(i8 %x, i8 %y, i8* %p, i8* %q) {
	%r = udiv i8 %x, %y
	%t = urem i8 %x, %y
	store i8 %r, i8* %p
	store i8 %t, i8* %q
	ret void
}

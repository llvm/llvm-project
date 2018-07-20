; RUN: llc -mtriple=arm64-none-linux-gnu < %s | FileCheck %s

define <8 x i8> @cmeq8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp eq <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmeq16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp eq <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmeq4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: cmeq {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp eq <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmeq8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: cmeq {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp eq <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmeq2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: cmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp eq <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmeq4xi32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: cmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp eq <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmeq2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp eq <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmne8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
;CHECK-NEXT: mvn {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmne16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
;CHECK-NEXT: mvn {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmne4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: cmeq {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
;CHECK-NEXT: mvn {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmne8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: cmeq {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
;CHECK-NEXT: mvn {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmne2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: cmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
;CHECK-NEXT: mvn {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmne4xi32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: cmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
;CHECK-NEXT: mvn {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmne2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
;CHECK-NEXT: mvn {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmgt8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: cmgt {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp sgt <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmgt16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: cmgt {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp sgt <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmgt4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: cmgt {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp sgt <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmgt8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: cmgt {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp sgt <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmgt2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: cmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp sgt <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmgt4xi32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: cmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp sgt <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmgt2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: cmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp sgt <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmlt8xi8(<8 x i8> %A, <8 x i8> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
;CHECK: cmgt {{v[0-9]+}}.8b, v1.8b, v0.8b
	%tmp3 = icmp slt <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmlt16xi8(<16 x i8> %A, <16 x i8> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
;CHECK: cmgt {{v[0-9]+}}.16b, v1.16b, v0.16b
	%tmp3 = icmp slt <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmlt4xi16(<4 x i16> %A, <4 x i16> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
;CHECK: cmgt {{v[0-9]+}}.4h, v1.4h, v0.4h
	%tmp3 = icmp slt <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmlt8xi16(<8 x i16> %A, <8 x i16> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
;CHECK: cmgt {{v[0-9]+}}.8h, v1.8h, v0.8h
	%tmp3 = icmp slt <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmlt2xi32(<2 x i32> %A, <2 x i32> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
;CHECK: cmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
	%tmp3 = icmp slt <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmlt4xi32(<4 x i32> %A, <4 x i32> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
;CHECK: cmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
	%tmp3 = icmp slt <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmlt2xi64(<2 x i64> %A, <2 x i64> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
;CHECK: cmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
	%tmp3 = icmp slt <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmge8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: cmge {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp sge <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmge16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: cmge {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp sge <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmge4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: cmge {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp sge <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmge8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: cmge {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp sge <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmge2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: cmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp sge <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmge4xi32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: cmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp sge <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmge2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: cmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp sge <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmle8xi8(<8 x i8> %A, <8 x i8> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
;CHECK: cmge {{v[0-9]+}}.8b, v1.8b, v0.8b
	%tmp3 = icmp sle <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmle16xi8(<16 x i8> %A, <16 x i8> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
;CHECK: cmge {{v[0-9]+}}.16b, v1.16b, v0.16b
	%tmp3 = icmp sle <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmle4xi16(<4 x i16> %A, <4 x i16> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
;CHECK: cmge {{v[0-9]+}}.4h, v1.4h, v0.4h
	%tmp3 = icmp sle <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmle8xi16(<8 x i16> %A, <8 x i16> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
;CHECK: cmge {{v[0-9]+}}.8h, v1.8h, v0.8h
	%tmp3 = icmp sle <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmle2xi32(<2 x i32> %A, <2 x i32> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
;CHECK: cmge {{v[0-9]+}}.2s, v1.2s, v0.2s
	%tmp3 = icmp sle <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmle4xi32(<4 x i32> %A, <4 x i32> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
;CHECK: cmge {{v[0-9]+}}.4s, v1.4s, v0.4s
	%tmp3 = icmp sle <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmle2xi64(<2 x i64> %A, <2 x i64> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
;CHECK: cmge {{v[0-9]+}}.2d, v1.2d, v0.2d
	%tmp3 = icmp sle <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmhi8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: cmhi {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ugt <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmhi16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: cmhi {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ugt <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmhi4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: cmhi {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp ugt <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmhi8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: cmhi {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp ugt <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmhi2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: cmhi {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp ugt <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmhi4xi32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: cmhi {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp ugt <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmhi2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: cmhi {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp ugt <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmlo8xi8(<8 x i8> %A, <8 x i8> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: cmhi {{v[0-9]+}}.8b, v1.8b, v0.8b
	%tmp3 = icmp ult <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmlo16xi8(<16 x i8> %A, <16 x i8> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: cmhi {{v[0-9]+}}.16b, v1.16b, v0.16b
	%tmp3 = icmp ult <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmlo4xi16(<4 x i16> %A, <4 x i16> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: cmhi {{v[0-9]+}}.4h, v1.4h, v0.4h
	%tmp3 = icmp ult <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmlo8xi16(<8 x i16> %A, <8 x i16> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: cmhi {{v[0-9]+}}.8h, v1.8h, v0.8h
	%tmp3 = icmp ult <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmlo2xi32(<2 x i32> %A, <2 x i32> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: cmhi {{v[0-9]+}}.2s, v1.2s, v0.2s
	%tmp3 = icmp ult <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmlo4xi32(<4 x i32> %A, <4 x i32> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: cmhi {{v[0-9]+}}.4s, v1.4s, v0.4s
	%tmp3 = icmp ult <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmlo2xi64(<2 x i64> %A, <2 x i64> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: cmhi {{v[0-9]+}}.2d, v1.2d, v0.2d
	%tmp3 = icmp ult <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmhs8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: cmhs {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp uge <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmhs16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: cmhs {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp uge <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmhs4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: cmhs {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp uge <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmhs8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: cmhs {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp uge <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmhs2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: cmhs {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp uge <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmhs4xi32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: cmhs {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp uge <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmhs2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: cmhs {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp uge <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmls8xi8(<8 x i8> %A, <8 x i8> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: cmhs {{v[0-9]+}}.8b, v1.8b, v0.8b
	%tmp3 = icmp ule <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmls16xi8(<16 x i8> %A, <16 x i8> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: cmhs {{v[0-9]+}}.16b, v1.16b, v0.16b
	%tmp3 = icmp ule <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmls4xi16(<4 x i16> %A, <4 x i16> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: cmhs {{v[0-9]+}}.4h, v1.4h, v0.4h
	%tmp3 = icmp ule <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmls8xi16(<8 x i16> %A, <8 x i16> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: cmhs {{v[0-9]+}}.8h, v1.8h, v0.8h
	%tmp3 = icmp ule <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmls2xi32(<2 x i32> %A, <2 x i32> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: cmhs {{v[0-9]+}}.2s, v1.2s, v0.2s
	%tmp3 = icmp ule <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmls4xi32(<4 x i32> %A, <4 x i32> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: cmhs {{v[0-9]+}}.4s, v1.4s, v0.4s
	%tmp3 = icmp ule <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmls2xi64(<2 x i64> %A, <2 x i64> %B) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: cmhs {{v[0-9]+}}.2d, v1.2d, v0.2d
	%tmp3 = icmp ule <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <8 x i8> @cmeqz8xi8(<8 x i8> %A) {
;CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0
	%tmp3 = icmp eq <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmeqz16xi8(<16 x i8> %A) {
;CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0
	%tmp3 = icmp eq <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmeqz4xi16(<4 x i16> %A) {
;CHECK: cmeq {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0
	%tmp3 = icmp eq <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmeqz8xi16(<8 x i16> %A) {
;CHECK: cmeq {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0
	%tmp3 = icmp eq <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmeqz2xi32(<2 x i32> %A) {
;CHECK: cmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0
	%tmp3 = icmp eq <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmeqz4xi32(<4 x i32> %A) {
;CHECK: cmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0
	%tmp3 = icmp eq <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmeqz2xi64(<2 x i64> %A) {
;CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
	%tmp3 = icmp eq <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <8 x i8> @cmgez8xi8(<8 x i8> %A) {
;CHECK: cmge {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0
	%tmp3 = icmp sge <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmgez16xi8(<16 x i8> %A) {
;CHECK: cmge {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0
	%tmp3 = icmp sge <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmgez4xi16(<4 x i16> %A) {
;CHECK: cmge {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0
	%tmp3 = icmp sge <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmgez8xi16(<8 x i16> %A) {
;CHECK: cmge {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0
	%tmp3 = icmp sge <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmgez2xi32(<2 x i32> %A) {
;CHECK: cmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0
	%tmp3 = icmp sge <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmgez4xi32(<4 x i32> %A) {
;CHECK: cmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0
	%tmp3 = icmp sge <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmgez2xi64(<2 x i64> %A) {
;CHECK: cmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
	%tmp3 = icmp sge <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <8 x i8> @cmgtz8xi8(<8 x i8> %A) {
;CHECK: cmgt {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0
	%tmp3 = icmp sgt <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmgtz16xi8(<16 x i8> %A) {
;CHECK: cmgt {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0
	%tmp3 = icmp sgt <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmgtz4xi16(<4 x i16> %A) {
;CHECK: cmgt {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0
	%tmp3 = icmp sgt <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmgtz8xi16(<8 x i16> %A) {
;CHECK: cmgt {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0
	%tmp3 = icmp sgt <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmgtz2xi32(<2 x i32> %A) {
;CHECK: cmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0
	%tmp3 = icmp sgt <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmgtz4xi32(<4 x i32> %A) {
;CHECK: cmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0
	%tmp3 = icmp sgt <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmgtz2xi64(<2 x i64> %A) {
;CHECK: cmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
	%tmp3 = icmp sgt <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmlez8xi8(<8 x i8> %A) {
;CHECK: cmle {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0
	%tmp3 = icmp sle <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmlez16xi8(<16 x i8> %A) {
;CHECK: cmle {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0
	%tmp3 = icmp sle <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmlez4xi16(<4 x i16> %A) {
;CHECK: cmle {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0
	%tmp3 = icmp sle <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmlez8xi16(<8 x i16> %A) {
;CHECK: cmle {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0
	%tmp3 = icmp sle <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmlez2xi32(<2 x i32> %A) {
;CHECK: cmle {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0
	%tmp3 = icmp sle <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmlez4xi32(<4 x i32> %A) {
;CHECK: cmle {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0
	%tmp3 = icmp sle <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmlez2xi64(<2 x i64> %A) {
;CHECK: cmle {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
	%tmp3 = icmp sle <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmltz8xi8(<8 x i8> %A) {
;CHECK: cmlt {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0
	%tmp3 = icmp slt <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmltz16xi8(<16 x i8> %A) {
;CHECK: cmlt {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0
	%tmp3 = icmp slt <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmltz4xi16(<4 x i16> %A) {
;CHECK: cmlt {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0
	%tmp3 = icmp slt <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmltz8xi16(<8 x i16> %A) {
;CHECK: cmlt {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0
	%tmp3 = icmp slt <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmltz2xi32(<2 x i32> %A) {
;CHECK: cmlt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0
	%tmp3 = icmp slt <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmltz4xi32(<4 x i32> %A) {
;CHECK: cmlt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0
	%tmp3 = icmp slt <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmltz2xi64(<2 x i64> %A) {
;CHECK: cmlt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
	%tmp3 = icmp slt <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmneqz8xi8(<8 x i8> %A) {
;CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #0
;CHECK-NEXT: mvn {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmneqz16xi8(<16 x i8> %A) {
;CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #0
;CHECK-NEXT: mvn {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmneqz4xi16(<4 x i16> %A) {
;CHECK: cmeq {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #0
;CHECK-NEXT: mvn {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmneqz8xi16(<8 x i16> %A) {
;CHECK: cmeq {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #0
;CHECK-NEXT: mvn {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmneqz2xi32(<2 x i32> %A) {
;CHECK: cmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #0
;CHECK-NEXT: mvn {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmneqz4xi32(<4 x i32> %A) {
;CHECK: cmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #0
;CHECK-NEXT: mvn {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmneqz2xi64(<2 x i64> %A) {
;CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #0
;CHECK-NEXT: mvn {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmhsz8xi8(<8 x i8> %A) {
;CHECK: movi v[[ZERO:[0-9]+]].8b, #2
;CHECK-NEXT: cmhs {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, v[[ZERO]].8b
	%tmp3 = icmp uge <8 x i8> %A, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmhsz16xi8(<16 x i8> %A) {
;CHECK: movi v[[ZERO:[0-9]+]].16b, #2
;CHECK-NEXT: cmhs {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, v[[ZERO]].16b
	%tmp3 = icmp uge <16 x i8> %A, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmhsz4xi16(<4 x i16> %A) {
;CHECK: movi v[[ZERO:[0-9]+]].4h, #2
;CHECK-NEXT: cmhs {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, v[[ZERO]].4h
	%tmp3 = icmp uge <4 x i16> %A, <i16 2, i16 2, i16 2, i16 2>
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmhsz8xi16(<8 x i16> %A) {
;CHECK: movi v[[ZERO:[0-9]+]].8h, #2
;CHECK-NEXT: cmhs {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, v[[ZERO]].8h
	%tmp3 = icmp uge <8 x i16> %A, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmhsz2xi32(<2 x i32> %A) {
;CHECK: movi v[[ZERO:[0-9]+]].2s, #2
;CHECK-NEXT: cmhs {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, v[[ZERO]].2s
	%tmp3 = icmp uge <2 x i32> %A, <i32 2, i32 2>
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmhsz4xi32(<4 x i32> %A) {
;CHECK: movi v[[ZERO:[0-9]+]].4s, #2
;CHECK-NEXT: cmhs {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, v[[ZERO]].4s
	%tmp3 = icmp uge <4 x i32> %A, <i32 2, i32 2, i32 2, i32 2>
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmhsz2xi64(<2 x i64> %A) {
;CHECK: orr w[[TWO:[0-9]+]], wzr, #0x2
;CHECK-NEXT: dup v[[ZERO:[0-9]+]].2d, x[[TWO]]
;CHECK-NEXT: cmhs {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, v[[ZERO]].2d
	%tmp3 = icmp uge <2 x i64> %A, <i64 2, i64 2>
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <8 x i8> @cmhiz8xi8(<8 x i8> %A) {
;CHECK: movi v[[ZERO:[0-9]+]].8b, #1
;CHECK-NEXT: cmhi {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, v[[ZERO]].8b
	%tmp3 = icmp ugt <8 x i8> %A, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmhiz16xi8(<16 x i8> %A) {
;CHECK: movi v[[ZERO:[0-9]+]].16b, #1
;CHECK-NEXT: cmhi {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, v[[ZERO]].16b
	%tmp3 = icmp ugt <16 x i8> %A, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmhiz4xi16(<4 x i16> %A) {
;CHECK: movi v[[ZERO:[0-9]+]].4h, #1
;CHECK-NEXT: cmhi {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, v[[ZERO]].4h
	%tmp3 = icmp ugt <4 x i16> %A, <i16 1, i16 1, i16 1, i16 1>
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmhiz8xi16(<8 x i16> %A) {
;CHECK: movi v[[ZERO:[0-9]+]].8h, #1
;CHECK-NEXT: cmhi {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, v[[ZERO]].8h
	%tmp3 = icmp ugt <8 x i16> %A, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmhiz2xi32(<2 x i32> %A) {
;CHECK: movi v[[ZERO:[0-9]+]].2s, #1
;CHECK-NEXT: cmhi {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, v[[ZERO]].2s
	%tmp3 = icmp ugt <2 x i32> %A, <i32 1, i32 1>
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmhiz4xi32(<4 x i32> %A) {
;CHECK: movi v[[ZERO:[0-9]+]].4s, #1
;CHECK-NEXT: cmhi {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, v[[ZERO]].4s
	%tmp3 = icmp ugt <4 x i32> %A, <i32 1, i32 1, i32 1, i32 1>
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmhiz2xi64(<2 x i64> %A) {
;CHECK: orr w[[ONE:[0-9]+]], wzr, #0x1
;CHECK-NEXT: dup v[[ZERO:[0-9]+]].2d, x[[ONE]]
;CHECK-NEXT: cmhi {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, v[[ZERO]].2d
	%tmp3 = icmp ugt <2 x i64> %A, <i64 1, i64 1>
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmlsz8xi8(<8 x i8> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: movi d[[ZERO:[0-9]+]], #0
;CHECK-NEXT: cmhs {{v[0-9]+}}.8b, v[[ZERO]].8b, v0.8b
	%tmp3 = icmp ule <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmlsz16xi8(<16 x i8> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: movi v[[ZERO:[0-9]+]].2d, #0
;CHECK-NEXT: cmhs {{v[0-9]+}}.16b, v[[ZERO]].16b, v0.16b
	%tmp3 = icmp ule <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmlsz4xi16(<4 x i16> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: movi d[[ZERO:[0-9]+]], #0
;CHECK-NEXT: cmhs {{v[0-9]+}}.4h, v[[ZERO]].4h, v0.4h
	%tmp3 = icmp ule <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmlsz8xi16(<8 x i16> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: movi v[[ZERO:[0-9]+]].2d, #0
;CHECK-NEXT: cmhs {{v[0-9]+}}.8h, v[[ZERO]].8h, v0.8h
	%tmp3 = icmp ule <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmlsz2xi32(<2 x i32> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: movi d[[ZERO:[0-9]+]], #0
;CHECK-NEXT: cmhs {{v[0-9]+}}.2s, v[[ZERO]].2s, v0.2s
	%tmp3 = icmp ule <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmlsz4xi32(<4 x i32> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: movi v[[ZERO:[0-9]+]].2d, #0
;CHECK-NEXT: cmhs {{v[0-9]+}}.4s, v[[ZERO]].4s, v0.4s
	%tmp3 = icmp ule <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmlsz2xi64(<2 x i64> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
;CHECK: movi v[[ZERO:[0-9]+]].2d, #0
;CHECK-NEXT: cmhs {{v[0-9]+}}.2d, v[[ZERO]].2d, v0.2d
	%tmp3 = icmp ule <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmloz8xi8(<8 x i8> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: movi v[[ZERO:[0-9]+]].8b, #2
;CHECK-NEXT: cmhi {{v[0-9]+}}.8b, v[[ZERO]].8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ult <8 x i8> %A, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmloz16xi8(<16 x i8> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: movi v[[ZERO:[0-9]+]].16b, #2
;CHECK-NEXT: cmhi {{v[0-9]+}}.16b, v[[ZERO]].16b, v0.16b
	%tmp3 = icmp ult <16 x i8> %A, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmloz4xi16(<4 x i16> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: movi v[[ZERO:[0-9]+]].4h, #2
;CHECK-NEXT: cmhi {{v[0-9]+}}.4h, v[[ZERO]].4h, v0.4h
	%tmp3 = icmp ult <4 x i16> %A, <i16 2, i16 2, i16 2, i16 2>
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmloz8xi16(<8 x i16> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: movi v[[ZERO:[0-9]+]].8h, #2
;CHECK-NEXT: cmhi {{v[0-9]+}}.8h, v[[ZERO]].8h, v0.8h
	%tmp3 = icmp ult <8 x i16> %A, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmloz2xi32(<2 x i32> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: movi v[[ZERO:[0-9]+]].2s, #2
;CHECK-NEXT: cmhi {{v[0-9]+}}.2s, v[[ZERO]].2s, v0.2s
	%tmp3 = icmp ult <2 x i32> %A, <i32 2, i32 2>
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmloz4xi32(<4 x i32> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: movi v[[ZERO:[0-9]+]].4s, #2
;CHECK-NEXT: cmhi {{v[0-9]+}}.4s, v[[ZERO]].4s, v0.4s
	%tmp3 = icmp ult <4 x i32> %A, <i32 2, i32 2, i32 2, i32 2>
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmloz2xi64(<2 x i64> %A) {
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
;CHECK: orr w[[TWO:[0-9]+]], wzr, #0x2
;CHECK-NEXT: dup v[[ZERO:[0-9]+]].2d, x[[TWO]]
;CHECK-NEXT: cmhi {{v[0-9]+}}.2d, v[[ZERO]].2d, v0.2d
	%tmp3 = icmp ult <2 x i64> %A, <i64 2, i64 2>
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <1 x i64> @cmeqz_v1i64(<1 x i64> %A) {
; CHECK-LABEL: cmeqz_v1i64:
; CHECK: cmeq d0, d0, #0
  %tst = icmp eq <1 x i64> %A, <i64 0>
  %mask = sext <1 x i1> %tst to <1 x i64>
  ret <1 x i64> %mask
}

define <1 x i64> @cmgez_v1i64(<1 x i64> %A) {
; CHECK-LABEL: cmgez_v1i64:
; CHECK: cmge d0, d0, #0
  %tst = icmp sge <1 x i64> %A, <i64 0>
  %mask = sext <1 x i1> %tst to <1 x i64>
  ret <1 x i64> %mask
}

define <1 x i64> @cmgtz_v1i64(<1 x i64> %A) {
; CHECK-LABEL: cmgtz_v1i64:
; CHECK: cmgt d0, d0, #0
  %tst = icmp sgt <1 x i64> %A, <i64 0>
  %mask = sext <1 x i1> %tst to <1 x i64>
  ret <1 x i64> %mask
}

define <1 x i64> @cmlez_v1i64(<1 x i64> %A) {
; CHECK-LABEL: cmlez_v1i64:
; CHECK: cmle d0, d0, #0
  %tst = icmp sle <1 x i64> %A, <i64 0>
  %mask = sext <1 x i1> %tst to <1 x i64>
  ret <1 x i64> %mask
}

define <1 x i64> @cmltz_v1i64(<1 x i64> %A) {
; CHECK-LABEL: cmltz_v1i64:
; CHECK: cmlt d0, d0, #0
  %tst = icmp slt <1 x i64> %A, <i64 0>
  %mask = sext <1 x i1> %tst to <1 x i64>
  ret <1 x i64> %mask
}

define <1 x i64> @fcmeqz_v1f64(<1 x double> %A) {
; CHECK-LABEL: fcmeqz_v1f64:
; CHECK: fcmeq d0, d0, #0
  %tst = fcmp oeq <1 x double> %A, <double 0.0>
  %mask = sext <1 x i1> %tst to <1 x i64>
  ret <1 x i64> %mask
}

define <1 x i64> @fcmgez_v1f64(<1 x double> %A) {
; CHECK-LABEL: fcmgez_v1f64:
; CHECK: fcmge d0, d0, #0
  %tst = fcmp oge <1 x double> %A, <double 0.0>
  %mask = sext <1 x i1> %tst to <1 x i64>
  ret <1 x i64> %mask
}

define <1 x i64> @fcmgtz_v1f64(<1 x double> %A) {
; CHECK-LABEL: fcmgtz_v1f64:
; CHECK: fcmgt d0, d0, #0
  %tst = fcmp ogt <1 x double> %A, <double 0.0>
  %mask = sext <1 x i1> %tst to <1 x i64>
  ret <1 x i64> %mask
}

define <1 x i64> @fcmlez_v1f64(<1 x double> %A) {
; CHECK-LABEL: fcmlez_v1f64:
; CHECK: fcmle d0, d0, #0
  %tst = fcmp ole <1 x double> %A, <double 0.0>
  %mask = sext <1 x i1> %tst to <1 x i64>
  ret <1 x i64> %mask
}

define <1 x i64> @fcmltz_v1f64(<1 x double> %A) {
; CHECK-LABEL: fcmltz_v1f64:
; CHECK: fcmlt d0, d0, #0
  %tst = fcmp olt <1 x double> %A, <double 0.0>
  %mask = sext <1 x i1> %tst to <1 x i64>
  ret <1 x i64> %mask
}

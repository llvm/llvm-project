; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

define <8 x i8> @cmeq8xi8(<8 x i8> %A, <8 x i8> %B) {
; CHECK-LABEL: cmeq8xi8:
; CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp eq <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmeq16xi8(<16 x i8> %A, <16 x i8> %B) {
; CHECK-LABEL: cmeq16xi8:
; CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp eq <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmeq4xi16(<4 x i16> %A, <4 x i16> %B) {
; CHECK-LABEL: cmeq4xi16:
; CHECK: cmeq {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp eq <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmeq8xi16(<8 x i16> %A, <8 x i16> %B) {
; CHECK-LABEL: cmeq8xi16:
; CHECK: cmeq {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp eq <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmeq2xi32(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: cmeq2xi32:
; CHECK: cmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp eq <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmeq4xi32(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: cmeq4xi32:
; CHECK: cmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp eq <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmeq2xi64(<2 x i64> %A, <2 x i64> %B) {
; CHECK-LABEL: cmeq2xi64:
; CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp eq <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmne8xi8(<8 x i8> %A, <8 x i8> %B) {
; CHECK-LABEL: cmne8xi8:
; CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmne16xi8(<16 x i8> %A, <16 x i8> %B) {
; CHECK-LABEL: cmne16xi8:
; CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmne4xi16(<4 x i16> %A, <4 x i16> %B) {
; CHECK-LABEL: cmne4xi16:
; CHECK: cmeq {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmne8xi16(<8 x i16> %A, <8 x i16> %B) {
; CHECK-LABEL: cmne8xi16:
; CHECK: cmeq {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmne2xi32(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: cmne2xi32:
; CHECK: cmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmne4xi32(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: cmne4xi32:
; CHECK: cmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmne2xi64(<2 x i64> %A, <2 x i64> %B) {
; CHECK-LABEL: cmne2xi64:
; CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmgt8xi8(<8 x i8> %A, <8 x i8> %B) {
; CHECK-LABEL: cmgt8xi8:
; CHECK: cmgt {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp sgt <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmgt16xi8(<16 x i8> %A, <16 x i8> %B) {
; CHECK-LABEL: cmgt16xi8:
; CHECK: cmgt {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp sgt <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmgt4xi16(<4 x i16> %A, <4 x i16> %B) {
; CHECK-LABEL: cmgt4xi16:
; CHECK: cmgt {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp sgt <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmgt8xi16(<8 x i16> %A, <8 x i16> %B) {
; CHECK-LABEL: cmgt8xi16:
; CHECK: cmgt {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp sgt <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmgt2xi32(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: cmgt2xi32:
; CHECK: cmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp sgt <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmgt4xi32(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: cmgt4xi32:
; CHECK: cmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp sgt <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmgt2xi64(<2 x i64> %A, <2 x i64> %B) {
; CHECK-LABEL: cmgt2xi64:
; CHECK: cmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp sgt <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmlt8xi8(<8 x i8> %A, <8 x i8> %B) {
; CHECK-LABEL: cmlt8xi8:
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
; CHECK: cmgt {{v[0-9]+}}.8b, v1.8b, v0.8b
	%tmp3 = icmp slt <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmlt16xi8(<16 x i8> %A, <16 x i8> %B) {
; CHECK-LABEL: cmlt16xi8:
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
; CHECK: cmgt {{v[0-9]+}}.16b, v1.16b, v0.16b
	%tmp3 = icmp slt <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmlt4xi16(<4 x i16> %A, <4 x i16> %B) {
; CHECK-LABEL: cmlt4xi16:
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
; CHECK: cmgt {{v[0-9]+}}.4h, v1.4h, v0.4h
	%tmp3 = icmp slt <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmlt8xi16(<8 x i16> %A, <8 x i16> %B) {
; CHECK-LABEL: cmlt8xi16:
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
; CHECK: cmgt {{v[0-9]+}}.8h, v1.8h, v0.8h
	%tmp3 = icmp slt <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmlt2xi32(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: cmlt2xi32:
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
; CHECK: cmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
	%tmp3 = icmp slt <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmlt4xi32(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: cmlt4xi32:
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
; CHECK: cmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
	%tmp3 = icmp slt <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmlt2xi64(<2 x i64> %A, <2 x i64> %B) {
; CHECK-LABEL: cmlt2xi64:
; Using registers other than v0, v1 are possible, but would be odd.
; LT implemented as GT, so check reversed operands.
; CHECK: cmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
	%tmp3 = icmp slt <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmge8xi8(<8 x i8> %A, <8 x i8> %B) {
; CHECK-LABEL: cmge8xi8:
; CHECK: cmge {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp sge <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmge16xi8(<16 x i8> %A, <16 x i8> %B) {
; CHECK-LABEL: cmge16xi8:
; CHECK: cmge {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp sge <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmge4xi16(<4 x i16> %A, <4 x i16> %B) {
; CHECK-LABEL: cmge4xi16:
; CHECK: cmge {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp sge <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmge8xi16(<8 x i16> %A, <8 x i16> %B) {
; CHECK-LABEL: cmge8xi16:
; CHECK: cmge {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp sge <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmge2xi32(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: cmge2xi32:
; CHECK: cmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp sge <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmge4xi32(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: cmge4xi32:
; CHECK: cmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp sge <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmge2xi64(<2 x i64> %A, <2 x i64> %B) {
; CHECK-LABEL: cmge2xi64:
; CHECK: cmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp sge <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmle8xi8(<8 x i8> %A, <8 x i8> %B) {
; CHECK-LABEL: cmle8xi8:
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
; CHECK: cmge {{v[0-9]+}}.8b, v1.8b, v0.8b
	%tmp3 = icmp sle <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmle16xi8(<16 x i8> %A, <16 x i8> %B) {
; CHECK-LABEL: cmle16xi8:
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
; CHECK: cmge {{v[0-9]+}}.16b, v1.16b, v0.16b
	%tmp3 = icmp sle <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmle4xi16(<4 x i16> %A, <4 x i16> %B) {
; CHECK-LABEL: cmle4xi16:
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
; CHECK: cmge {{v[0-9]+}}.4h, v1.4h, v0.4h
	%tmp3 = icmp sle <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmle8xi16(<8 x i16> %A, <8 x i16> %B) {
; CHECK-LABEL: cmle8xi16:
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
; CHECK: cmge {{v[0-9]+}}.8h, v1.8h, v0.8h
	%tmp3 = icmp sle <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmle2xi32(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: cmle2xi32:
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
; CHECK: cmge {{v[0-9]+}}.2s, v1.2s, v0.2s
	%tmp3 = icmp sle <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmle4xi32(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: cmle4xi32:
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
; CHECK: cmge {{v[0-9]+}}.4s, v1.4s, v0.4s
	%tmp3 = icmp sle <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmle2xi64(<2 x i64> %A, <2 x i64> %B) {
; CHECK-LABEL: cmle2xi64:
; Using registers other than v0, v1 are possible, but would be odd.
; LE implemented as GE, so check reversed operands.
; CHECK: cmge {{v[0-9]+}}.2d, v1.2d, v0.2d
	%tmp3 = icmp sle <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmhi8xi8(<8 x i8> %A, <8 x i8> %B) {
; CHECK-LABEL: cmhi8xi8:
; CHECK: cmhi {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ugt <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmhi16xi8(<16 x i8> %A, <16 x i8> %B) {
; CHECK-LABEL: cmhi16xi8:
; CHECK: cmhi {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ugt <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmhi4xi16(<4 x i16> %A, <4 x i16> %B) {
; CHECK-LABEL: cmhi4xi16:
; CHECK: cmhi {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp ugt <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmhi8xi16(<8 x i16> %A, <8 x i16> %B) {
; CHECK-LABEL: cmhi8xi16:
; CHECK: cmhi {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp ugt <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmhi2xi32(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: cmhi2xi32:
; CHECK: cmhi {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp ugt <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmhi4xi32(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: cmhi4xi32:
; CHECK: cmhi {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp ugt <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmhi2xi64(<2 x i64> %A, <2 x i64> %B) {
; CHECK-LABEL: cmhi2xi64:
; CHECK: cmhi {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp ugt <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmlo8xi8(<8 x i8> %A, <8 x i8> %B) {
; CHECK-LABEL: cmlo8xi8:
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
; CHECK: cmhi {{v[0-9]+}}.8b, v1.8b, v0.8b
	%tmp3 = icmp ult <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmlo16xi8(<16 x i8> %A, <16 x i8> %B) {
; CHECK-LABEL: cmlo16xi8:
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
; CHECK: cmhi {{v[0-9]+}}.16b, v1.16b, v0.16b
	%tmp3 = icmp ult <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmlo4xi16(<4 x i16> %A, <4 x i16> %B) {
; CHECK-LABEL: cmlo4xi16:
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
; CHECK: cmhi {{v[0-9]+}}.4h, v1.4h, v0.4h
	%tmp3 = icmp ult <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmlo8xi16(<8 x i16> %A, <8 x i16> %B) {
; CHECK-LABEL: cmlo8xi16:
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
; CHECK: cmhi {{v[0-9]+}}.8h, v1.8h, v0.8h
	%tmp3 = icmp ult <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmlo2xi32(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: cmlo2xi32:
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
; CHECK: cmhi {{v[0-9]+}}.2s, v1.2s, v0.2s
	%tmp3 = icmp ult <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmlo4xi32(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: cmlo4xi32:
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
; CHECK: cmhi {{v[0-9]+}}.4s, v1.4s, v0.4s
	%tmp3 = icmp ult <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmlo2xi64(<2 x i64> %A, <2 x i64> %B) {
; CHECK-LABEL: cmlo2xi64:
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
; CHECK: cmhi {{v[0-9]+}}.2d, v1.2d, v0.2d
	%tmp3 = icmp ult <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmhs8xi8(<8 x i8> %A, <8 x i8> %B) {
; CHECK-LABEL: cmhs8xi8:
; CHECK: cmhs {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp uge <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmhs16xi8(<16 x i8> %A, <16 x i8> %B) {
; CHECK-LABEL: cmhs16xi8:
; CHECK: cmhs {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp uge <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmhs4xi16(<4 x i16> %A, <4 x i16> %B) {
; CHECK-LABEL: cmhs4xi16:
; CHECK: cmhs {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp uge <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmhs8xi16(<8 x i16> %A, <8 x i16> %B) {
; CHECK-LABEL: cmhs8xi16:
; CHECK: cmhs {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp uge <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmhs2xi32(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: cmhs2xi32:
; CHECK: cmhs {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp uge <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmhs4xi32(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: cmhs4xi32:
; CHECK: cmhs {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp uge <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmhs2xi64(<2 x i64> %A, <2 x i64> %B) {
; CHECK-LABEL: cmhs2xi64:
; CHECK: cmhs {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp uge <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmls8xi8(<8 x i8> %A, <8 x i8> %B) {
; CHECK-LABEL: cmls8xi8:
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
; CHECK: cmhs {{v[0-9]+}}.8b, v1.8b, v0.8b
	%tmp3 = icmp ule <8 x i8> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmls16xi8(<16 x i8> %A, <16 x i8> %B) {
; CHECK-LABEL: cmls16xi8:
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
; CHECK: cmhs {{v[0-9]+}}.16b, v1.16b, v0.16b
	%tmp3 = icmp ule <16 x i8> %A, %B;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmls4xi16(<4 x i16> %A, <4 x i16> %B) {
; CHECK-LABEL: cmls4xi16:
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
; CHECK: cmhs {{v[0-9]+}}.4h, v1.4h, v0.4h
	%tmp3 = icmp ule <4 x i16> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmls8xi16(<8 x i16> %A, <8 x i16> %B) {
; CHECK-LABEL: cmls8xi16:
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
; CHECK: cmhs {{v[0-9]+}}.8h, v1.8h, v0.8h
	%tmp3 = icmp ule <8 x i16> %A, %B;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmls2xi32(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: cmls2xi32:
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
; CHECK: cmhs {{v[0-9]+}}.2s, v1.2s, v0.2s
	%tmp3 = icmp ule <2 x i32> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmls4xi32(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: cmls4xi32:
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
; CHECK: cmhs {{v[0-9]+}}.4s, v1.4s, v0.4s
	%tmp3 = icmp ule <4 x i32> %A, %B;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmls2xi64(<2 x i64> %A, <2 x i64> %B) {
; CHECK-LABEL: cmls2xi64:
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
; CHECK: cmhs {{v[0-9]+}}.2d, v1.2d, v0.2d
	%tmp3 = icmp ule <2 x i64> %A, %B;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmtst8xi8(<8 x i8> %A, <8 x i8> %B) {
; CHECK-LABEL: cmtst8xi8:
; CHECK: cmtst {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = and <8 x i8> %A, %B
	%tmp4 = icmp ne <8 x i8> %tmp3, zeroinitializer
   %tmp5 = sext <8 x i1> %tmp4 to <8 x i8>
	ret <8 x i8> %tmp5
}

define <16 x i8> @cmtst16xi8(<16 x i8> %A, <16 x i8> %B) {
; CHECK-LABEL: cmtst16xi8:
; CHECK: cmtst {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = and <16 x i8> %A, %B
	%tmp4 = icmp ne <16 x i8> %tmp3, zeroinitializer
   %tmp5 = sext <16 x i1> %tmp4 to <16 x i8>
	ret <16 x i8> %tmp5
}

define <4 x i16> @cmtst4xi16(<4 x i16> %A, <4 x i16> %B) {
; CHECK-LABEL: cmtst4xi16:
; CHECK: cmtst {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = and <4 x i16> %A, %B
	%tmp4 = icmp ne <4 x i16> %tmp3, zeroinitializer
   %tmp5 = sext <4 x i1> %tmp4 to <4 x i16>
	ret <4 x i16> %tmp5
}

define <8 x i16> @cmtst8xi16(<8 x i16> %A, <8 x i16> %B) {
; CHECK-LABEL: cmtst8xi16:
; CHECK: cmtst {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = and <8 x i16> %A, %B
	%tmp4 = icmp ne <8 x i16> %tmp3, zeroinitializer
   %tmp5 = sext <8 x i1> %tmp4 to <8 x i16>
	ret <8 x i16> %tmp5
}

define <2 x i32> @cmtst2xi32(<2 x i32> %A, <2 x i32> %B) {
; CHECK-LABEL: cmtst2xi32:
; CHECK: cmtst {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = and <2 x i32> %A, %B
	%tmp4 = icmp ne <2 x i32> %tmp3, zeroinitializer
   %tmp5 = sext <2 x i1> %tmp4 to <2 x i32>
	ret <2 x i32> %tmp5
}

define <4 x i32> @cmtst4xi32(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: cmtst4xi32:
; CHECK: cmtst {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = and <4 x i32> %A, %B
	%tmp4 = icmp ne <4 x i32> %tmp3, zeroinitializer
   %tmp5 = sext <4 x i1> %tmp4 to <4 x i32>
	ret <4 x i32> %tmp5
}

define <2 x i64> @cmtst2xi64(<2 x i64> %A, <2 x i64> %B) {
; CHECK-LABEL: cmtst2xi64:
; CHECK: cmtst {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = and <2 x i64> %A, %B
	%tmp4 = icmp ne <2 x i64> %tmp3, zeroinitializer
   %tmp5 = sext <2 x i1> %tmp4 to <2 x i64>
	ret <2 x i64> %tmp5
}



define <8 x i8> @cmeqz8xi8(<8 x i8> %A) {
; CHECK-LABEL: cmeqz8xi8:
; CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x0|0}}
	%tmp3 = icmp eq <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmeqz16xi8(<16 x i8> %A) {
; CHECK-LABEL: cmeqz16xi8:
; CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x0|0}}
	%tmp3 = icmp eq <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmeqz4xi16(<4 x i16> %A) {
; CHECK-LABEL: cmeqz4xi16:
; CHECK: cmeq {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #{{0x0|0}}
	%tmp3 = icmp eq <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmeqz8xi16(<8 x i16> %A) {
; CHECK-LABEL: cmeqz8xi16:
; CHECK: cmeq {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #{{0x0|0}}
	%tmp3 = icmp eq <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmeqz2xi32(<2 x i32> %A) {
; CHECK-LABEL: cmeqz2xi32:
; CHECK: cmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0x0|0}}
	%tmp3 = icmp eq <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmeqz4xi32(<4 x i32> %A) {
; CHECK-LABEL: cmeqz4xi32:
; CHECK: cmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0x0|0}}
	%tmp3 = icmp eq <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmeqz2xi64(<2 x i64> %A) {
; CHECK-LABEL: cmeqz2xi64:
; CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0x0|0}}
	%tmp3 = icmp eq <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <8 x i8> @cmgez8xi8(<8 x i8> %A) {
; CHECK-LABEL: cmgez8xi8:
; CHECK: cmge {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x0|0}}
	%tmp3 = icmp sge <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmgez16xi8(<16 x i8> %A) {
; CHECK-LABEL: cmgez16xi8:
; CHECK: cmge {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x0|0}}
	%tmp3 = icmp sge <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmgez4xi16(<4 x i16> %A) {
; CHECK-LABEL: cmgez4xi16:
; CHECK: cmge {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #{{0x0|0}}
	%tmp3 = icmp sge <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmgez8xi16(<8 x i16> %A) {
; CHECK-LABEL: cmgez8xi16:
; CHECK: cmge {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #{{0x0|0}}
	%tmp3 = icmp sge <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmgez2xi32(<2 x i32> %A) {
; CHECK-LABEL: cmgez2xi32:
; CHECK: cmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0x0|0}}
	%tmp3 = icmp sge <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmgez4xi32(<4 x i32> %A) {
; CHECK-LABEL: cmgez4xi32:
; CHECK: cmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0x0|0}}
	%tmp3 = icmp sge <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmgez2xi64(<2 x i64> %A) {
; CHECK-LABEL: cmgez2xi64:
; CHECK: cmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0x0|0}}
	%tmp3 = icmp sge <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <8 x i8> @cmgez8xi8_alt(<8 x i8> %A) {
; CHECK-LABEL: cmgez8xi8_alt:
; CHECK: cmge {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x0|0}}
  %sign = ashr <8 x i8> %A, <i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>
  %not = xor <8 x i8> %sign, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  ret <8 x i8> %not
}

define <16 x i8> @cmgez16xi8_alt(<16 x i8> %A) {
; CHECK-LABEL: cmgez16xi8_alt:
; CHECK: cmge {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x0|0}}
  %sign = ashr <16 x i8> %A, <i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7, i8 7>
  %not = xor <16 x i8> %sign, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  ret <16 x i8> %not
}

define <4 x i16> @cmgez4xi16_alt(<4 x i16> %A) {
; CHECK-LABEL: cmgez4xi16_alt:
; CHECK: cmge {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #{{0x0|0}}
  %sign = ashr <4 x i16> %A, <i16 15, i16 15, i16 15, i16 15>
  %not = xor <4 x i16> %sign, <i16 -1, i16 -1, i16 -1, i16 -1>
  ret <4 x i16> %not
}

define <8 x i16> @cmgez8xi16_alt(<8 x i16> %A) {
; CHECK-LABEL: cmgez8xi16_alt:
; CHECK: cmge {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #{{0x0|0}}
  %sign = ashr <8 x i16> %A, <i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15, i16 15>
  %not = xor <8 x i16> %sign, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  ret <8 x i16> %not
}

define <2 x i32> @cmgez2xi32_alt(<2 x i32> %A) {
; CHECK-LABEL: cmgez2xi32_alt:
; CHECK: cmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0x0|0}}
  %sign = ashr <2 x i32> %A, <i32 31, i32 31>
  %not = xor <2 x i32> %sign, <i32 -1, i32 -1>
  ret <2 x i32> %not
}

define <4 x i32> @cmgez4xi32_alt(<4 x i32> %A) {
; CHECK-LABEL: cmgez4xi32_alt:
; CHECK: cmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0x0|0}}
  %sign = ashr <4 x i32> %A, <i32 31, i32 31, i32 31, i32 31>
  %not = xor <4 x i32> %sign, <i32 -1, i32 -1, i32 -1, i32 -1>
  ret <4 x i32> %not
}

define <2 x i64> @cmgez2xi64_alt(<2 x i64> %A) {
; CHECK-LABEL: cmgez2xi64_alt:
; CHECK: cmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0x0|0}}
  %sign = ashr <2 x i64> %A, <i64 63, i64 63>
  %not = xor <2 x i64> %sign, <i64 -1, i64 -1>
  ret <2 x i64> %not
}


define <8 x i8> @cmgtz8xi8(<8 x i8> %A) {
; CHECK-LABEL: cmgtz8xi8:
; CHECK: cmgt {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x0|0}}
	%tmp3 = icmp sgt <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmgtz16xi8(<16 x i8> %A) {
; CHECK-LABEL: cmgtz16xi8:
; CHECK: cmgt {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x0|0}}
	%tmp3 = icmp sgt <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmgtz4xi16(<4 x i16> %A) {
; CHECK-LABEL: cmgtz4xi16:
; CHECK: cmgt {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #{{0x0|0}}
	%tmp3 = icmp sgt <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmgtz8xi16(<8 x i16> %A) {
; CHECK-LABEL: cmgtz8xi16:
; CHECK: cmgt {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #{{0x0|0}}
	%tmp3 = icmp sgt <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmgtz2xi32(<2 x i32> %A) {
; CHECK-LABEL: cmgtz2xi32:
; CHECK: cmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0x0|0}}
	%tmp3 = icmp sgt <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmgtz4xi32(<4 x i32> %A) {
; CHECK-LABEL: cmgtz4xi32:
; CHECK: cmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0x0|0}}
	%tmp3 = icmp sgt <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmgtz2xi64(<2 x i64> %A) {
; CHECK-LABEL: cmgtz2xi64:
; CHECK: cmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0x0|0}}
	%tmp3 = icmp sgt <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmlez8xi8(<8 x i8> %A) {
; CHECK-LABEL: cmlez8xi8:
; CHECK: cmle {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x0|0}}
	%tmp3 = icmp sle <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmlez16xi8(<16 x i8> %A) {
; CHECK-LABEL: cmlez16xi8:
; CHECK: cmle {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x0|0}}
	%tmp3 = icmp sle <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmlez4xi16(<4 x i16> %A) {
; CHECK-LABEL: cmlez4xi16:
; CHECK: cmle {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #{{0x0|0}}
	%tmp3 = icmp sle <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmlez8xi16(<8 x i16> %A) {
; CHECK-LABEL: cmlez8xi16:
; CHECK: cmle {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #{{0x0|0}}
	%tmp3 = icmp sle <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmlez2xi32(<2 x i32> %A) {
; CHECK-LABEL: cmlez2xi32:
; CHECK: cmle {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0x0|0}}
	%tmp3 = icmp sle <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmlez4xi32(<4 x i32> %A) {
; CHECK-LABEL: cmlez4xi32:
; CHECK: cmle {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0x0|0}}
	%tmp3 = icmp sle <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmlez2xi64(<2 x i64> %A) {
; CHECK-LABEL: cmlez2xi64:
; CHECK: cmle {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0x0|0}}
	%tmp3 = icmp sle <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmltz8xi8(<8 x i8> %A) {
; CHECK-LABEL: cmltz8xi8:
; CHECK: cmlt {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x0|0}}
	%tmp3 = icmp slt <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmltz16xi8(<16 x i8> %A) {
; CHECK-LABEL: cmltz16xi8:
; CHECK: cmlt {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x0|0}}
	%tmp3 = icmp slt <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmltz4xi16(<4 x i16> %A) {
; CHECK-LABEL: cmltz4xi16:
; CHECK: cmlt {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #{{0x0|0}}
	%tmp3 = icmp slt <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmltz8xi16(<8 x i16> %A) {
; CHECK-LABEL: cmltz8xi16:
; CHECK: cmlt {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #{{0x0|0}}
	%tmp3 = icmp slt <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmltz2xi32(<2 x i32> %A) {
; CHECK-LABEL: cmltz2xi32:
; CHECK: cmlt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0x0|0}}
	%tmp3 = icmp slt <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmltz4xi32(<4 x i32> %A) {
; CHECK-LABEL: cmltz4xi32:
; CHECK: cmlt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0x0|0}}
	%tmp3 = icmp slt <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmltz2xi64(<2 x i64> %A) {
; CHECK-LABEL: cmltz2xi64:
; CHECK: cmlt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0x0|0}}
	%tmp3 = icmp slt <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmneqz8xi8(<8 x i8> %A) {
; CHECK-LABEL: cmneqz8xi8:
; CHECK: cmeq {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, #{{0x0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmneqz16xi8(<16 x i8> %A) {
; CHECK-LABEL: cmneqz16xi8:
; CHECK: cmeq {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, #{{0x0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmneqz4xi16(<4 x i16> %A) {
; CHECK-LABEL: cmneqz4xi16:
; CHECK: cmeq {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, #{{0x0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmneqz8xi16(<8 x i16> %A) {
; CHECK-LABEL: cmneqz8xi16:
; CHECK: cmeq {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, #{{0x0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmneqz2xi32(<2 x i32> %A) {
; CHECK-LABEL: cmneqz2xi32:
; CHECK: cmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0x0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ne <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmneqz4xi32(<4 x i32> %A) {
; CHECK-LABEL: cmneqz4xi32:
; CHECK: cmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0x0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmneqz2xi64(<2 x i64> %A) {
; CHECK-LABEL: cmneqz2xi64:
; CHECK: cmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0x0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ne <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmhsz8xi8(<8 x i8> %A) {
; CHECK-LABEL: cmhsz8xi8:
; CHECK: movi {{v[0-9]+}}.8b, #{{0x2|2}}
; CHECK-NEXT: cmhs {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp uge <8 x i8> %A, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmhsz16xi8(<16 x i8> %A) {
; CHECK-LABEL: cmhsz16xi8:
; CHECK: movi {{v[0-9]+}}.16b, #{{0x2|2}}
; CHECK-NEXT: cmhs {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp uge <16 x i8> %A, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmhsz4xi16(<4 x i16> %A) {
; CHECK-LABEL: cmhsz4xi16:
; CHECK: movi {{v[0-9]+}}.4h, #{{0x2|2}}
; CHECK-NEXT: cmhs {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp uge <4 x i16> %A, <i16 2, i16 2, i16 2, i16 2>
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmhsz8xi16(<8 x i16> %A) {
; CHECK-LABEL: cmhsz8xi16:
; CHECK: movi {{v[0-9]+}}.8h, #{{0x2|2}}
; CHECK-NEXT: cmhs {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp uge <8 x i16> %A, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmhsz2xi32(<2 x i32> %A) {
; CHECK-LABEL: cmhsz2xi32:
; CHECK: movi {{v[0-9]+}}.2s, #{{0x2|2}}
; CHECK-NEXT: cmhs {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp uge <2 x i32> %A, <i32 2, i32 2>
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmhsz4xi32(<4 x i32> %A) {
; CHECK-LABEL: cmhsz4xi32:
; CHECK: movi {{v[0-9]+}}.4s, #{{0x2|2}}
; CHECK-NEXT: cmhs {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp uge <4 x i32> %A, <i32 2, i32 2, i32 2, i32 2>
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmhsz2xi64(<2 x i64> %A) {
; CHECK-LABEL: cmhsz2xi64:
; CHECK: orr w[[TWO:[0-9]+]], wzr, #0x2
; CHECK-NEXT: {{v[0-9]+}}.2d, x[[TWO]]
; CHECK-NEXT: cmhs {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp uge <2 x i64> %A, <i64 2, i64 2>
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <8 x i8> @cmhiz8xi8(<8 x i8> %A) {
; CHECK-LABEL: cmhiz8xi8:
; CHECK: movi {{v[0-9]+}}.8b, #{{0x1|1}}
; CHECK-NEXT: cmhi {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ugt <8 x i8> %A, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmhiz16xi8(<16 x i8> %A) {
; CHECK-LABEL: cmhiz16xi8:
; CHECK: movi {{v[0-9]+}}.16b, #{{0x1|1}}
; CHECK-NEXT: cmhi {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = icmp ugt <16 x i8> %A, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmhiz4xi16(<4 x i16> %A) {
; CHECK-LABEL: cmhiz4xi16:
; CHECK: movi {{v[0-9]+}}.4h, #{{0x1|1}}
; CHECK-NEXT: cmhi {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = icmp ugt <4 x i16> %A, <i16 1, i16 1, i16 1, i16 1>
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmhiz8xi16(<8 x i16> %A) {
; CHECK-LABEL: cmhiz8xi16:
; CHECK: movi {{v[0-9]+}}.8h, #{{0x1|1}}
; CHECK-NEXT: cmhi {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = icmp ugt <8 x i16> %A, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmhiz2xi32(<2 x i32> %A) {
; CHECK-LABEL: cmhiz2xi32:
; CHECK: movi {{v[0-9]+}}.2s, #{{0x1|1}}
; CHECK-NEXT: cmhi {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = icmp ugt <2 x i32> %A, <i32 1, i32 1>
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmhiz4xi32(<4 x i32> %A) {
; CHECK-LABEL: cmhiz4xi32:
; CHECK: movi {{v[0-9]+}}.4s, #{{0x1|1}}
; CHECK-NEXT: cmhi {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = icmp ugt <4 x i32> %A, <i32 1, i32 1, i32 1, i32 1>
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmhiz2xi64(<2 x i64> %A) {
; CHECK-LABEL: cmhiz2xi64:
; CHECK: orr w[[ONE:[0-9]+]], wzr, #{{0x1|1}}
; CHECK-NEXT: dup {{v[0-9]+}}.2d, x[[ONE]]
; CHECK-NEXT: cmhi {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = icmp ugt <2 x i64> %A, <i64 1, i64 1>
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmlsz8xi8(<8 x i8> %A) {
; CHECK-LABEL: cmlsz8xi8:
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
; CHECK: movi {{v1.8b|v1.2d}}, #{{0x0|0}}
; CHECK-NEXT: cmhs {{v[0-9]+}}.8b, v1.8b, v0.8b
	%tmp3 = icmp ule <8 x i8> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmlsz16xi8(<16 x i8> %A) {
; CHECK-LABEL: cmlsz16xi8:
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
; CHECK: movi {{v1.16b|v1.2d}}, #{{0x0|0}}
; CHECK-NEXT: cmhs {{v[0-9]+}}.16b, v1.16b, v0.16b
	%tmp3 = icmp ule <16 x i8> %A, zeroinitializer;
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmlsz4xi16(<4 x i16> %A) {
; CHECK-LABEL: cmlsz4xi16:
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
; CHECK: movi {{v1.8b|v1.2d}}, #{{0x0|0}}
; CHECK-NEXT: cmhs {{v[0-9]+}}.4h, v1.4h, v0.4h
	%tmp3 = icmp ule <4 x i16> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmlsz8xi16(<8 x i16> %A) {
; CHECK-LABEL: cmlsz8xi16:
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
; CHECK: movi {{v1.16b|v1.2d}}, #{{0x0|0}}
; CHECK-NEXT: cmhs {{v[0-9]+}}.8h, v1.8h, v0.8h
	%tmp3 = icmp ule <8 x i16> %A, zeroinitializer;
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmlsz2xi32(<2 x i32> %A) {
; CHECK-LABEL: cmlsz2xi32:
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
; CHECK: movi {{v1.8b|v1.2d}}, #{{0x0|0}}
; CHECK-NEXT: cmhs {{v[0-9]+}}.2s, v1.2s, v0.2s
	%tmp3 = icmp ule <2 x i32> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmlsz4xi32(<4 x i32> %A) {
; CHECK-LABEL: cmlsz4xi32:
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
; CHECK: movi {{v1.16b|v1.2d}}, #{{0x0|0}}
; CHECK-NEXT: cmhs {{v[0-9]+}}.4s, v1.4s, v0.4s
	%tmp3 = icmp ule <4 x i32> %A, zeroinitializer;
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmlsz2xi64(<2 x i64> %A) {
; CHECK-LABEL: cmlsz2xi64:
; Using registers other than v0, v1 are possible, but would be odd.
; LS implemented as HS, so check reversed operands.
; CHECK: movi {{v1.16b|v1.2d}}, #{{0x0|0}}
; CHECK-NEXT: cmhs {{v[0-9]+}}.2d, v1.2d, v0.2d
	%tmp3 = icmp ule <2 x i64> %A, zeroinitializer;
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <8 x i8> @cmloz8xi8(<8 x i8> %A) {
; CHECK-LABEL: cmloz8xi8:
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
; CHECK: movi v1.8b, #{{0x2|2}}
; CHECK-NEXT: cmhi {{v[0-9]+}}.8b, v1.8b, {{v[0-9]+}}.8b
	%tmp3 = icmp ult <8 x i8> %A, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i8>
	ret <8 x i8> %tmp4
}

define <16 x i8> @cmloz16xi8(<16 x i8> %A) {
; CHECK-LABEL: cmloz16xi8:
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
; CHECK: movi v1.16b, #{{0x2|2}}
; CHECK-NEXT: cmhi {{v[0-9]+}}.16b, v1.16b, v0.16b
	%tmp3 = icmp ult <16 x i8> %A, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
   %tmp4 = sext <16 x i1> %tmp3 to <16 x i8>
	ret <16 x i8> %tmp4
}

define <4 x i16> @cmloz4xi16(<4 x i16> %A) {
; CHECK-LABEL: cmloz4xi16:
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
; CHECK: movi v1.4h, #{{0x2|2}}
; CHECK-NEXT: cmhi {{v[0-9]+}}.4h, v1.4h, v0.4h
	%tmp3 = icmp ult <4 x i16> %A, <i16 2, i16 2, i16 2, i16 2>
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i16>
	ret <4 x i16> %tmp4
}

define <8 x i16> @cmloz8xi16(<8 x i16> %A) {
; CHECK-LABEL: cmloz8xi16:
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
; CHECK: movi v1.8h, #{{0x2|2}}
; CHECK-NEXT: cmhi {{v[0-9]+}}.8h, v1.8h, v0.8h
	%tmp3 = icmp ult <8 x i16> %A, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
   %tmp4 = sext <8 x i1> %tmp3 to <8 x i16>
	ret <8 x i16> %tmp4
}

define <2 x i32> @cmloz2xi32(<2 x i32> %A) {
; CHECK-LABEL: cmloz2xi32:
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
; CHECK: movi v1.2s, #{{0x2|2}}
; CHECK-NEXT: cmhi {{v[0-9]+}}.2s, v1.2s, v0.2s
	%tmp3 = icmp ult <2 x i32> %A, <i32 2, i32 2>
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @cmloz4xi32(<4 x i32> %A) {
; CHECK-LABEL: cmloz4xi32:
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
; CHECK: movi v1.4s, #{{0x2|2}}
; CHECK-NEXT: cmhi {{v[0-9]+}}.4s, v1.4s, v0.4s
	%tmp3 = icmp ult <4 x i32> %A, <i32 2, i32 2, i32 2, i32 2>
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @cmloz2xi64(<2 x i64> %A) {
; CHECK-LABEL: cmloz2xi64:
; Using registers other than v0, v1 are possible, but would be odd.
; LO implemented as HI, so check reversed operands.
; CHECK: orr w[[TWO:[0-9]+]], wzr, #{{0x2|2}}
; CHECK-NEXT: dup v1.2d, x[[TWO]]
; CHECK-NEXT: cmhi {{v[0-9]+}}.2d, v1.2d, v0.2d
	%tmp3 = icmp ult <2 x i64> %A, <i64 2, i64 2>
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <2 x i32> @fcmoeq2xfloat(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: fcmoeq2xfloat:
; CHECK: fcmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
   %tmp3 = fcmp oeq <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmoeq4xfloat(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: fcmoeq4xfloat:
; CHECK: fcmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
   %tmp3 = fcmp oeq <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmoeq2xdouble(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: fcmoeq2xdouble:
; CHECK: fcmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
   %tmp3 = fcmp oeq <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmoge2xfloat(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: fcmoge2xfloat:
; CHECK: fcmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
   %tmp3 = fcmp oge <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmoge4xfloat(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: fcmoge4xfloat:
; CHECK: fcmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
   %tmp3 = fcmp oge <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmoge2xdouble(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: fcmoge2xdouble:
; CHECK: fcmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
   %tmp3 = fcmp oge <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmogt2xfloat(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: fcmogt2xfloat:
; CHECK: fcmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
   %tmp3 = fcmp ogt <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmogt4xfloat(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: fcmogt4xfloat:
; CHECK: fcmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
   %tmp3 = fcmp ogt <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmogt2xdouble(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: fcmogt2xdouble:
; CHECK: fcmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
   %tmp3 = fcmp ogt <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmole2xfloat(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: fcmole2xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; OLE implemented as OGE, so check reversed operands.
; CHECK: fcmge {{v[0-9]+}}.2s, v1.2s, v0.2s
   %tmp3 = fcmp ole <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmole4xfloat(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: fcmole4xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; OLE implemented as OGE, so check reversed operands.
; CHECK: fcmge {{v[0-9]+}}.4s, v1.4s, v0.4s
   %tmp3 = fcmp ole <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmole2xdouble(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: fcmole2xdouble:
; Using registers other than v0, v1 are possible, but would be odd.
; OLE implemented as OGE, so check reversed operands.
; CHECK: fcmge {{v[0-9]+}}.2d, v1.2d, v0.2d
   %tmp3 = fcmp ole <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmolt2xfloat(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: fcmolt2xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; OLE implemented as OGE, so check reversed operands.
; CHECK: fcmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
   %tmp3 = fcmp olt <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmolt4xfloat(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: fcmolt4xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; OLE implemented as OGE, so check reversed operands.
; CHECK: fcmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
   %tmp3 = fcmp olt <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmolt2xdouble(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: fcmolt2xdouble:
; Using registers other than v0, v1 are possible, but would be odd.
; OLE implemented as OGE, so check reversed operands.
; CHECK: fcmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
   %tmp3 = fcmp olt <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmone2xfloat(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: fcmone2xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; ONE = OGT | OLT, OLT implemented as OGT so check reversed operands
; CHECK: fcmgt {{v[0-9]+}}.2s, v0.2s, v1.2s
; CHECK-NEXT: fcmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
; CHECK-NEXT: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp one <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmone4xfloat(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: fcmone4xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; ONE = OGT | OLT, OLT implemented as OGT so check reversed operands
; CHECK: fcmgt {{v[0-9]+}}.4s, v0.4s, v1.4s
; CHECK-NEXT: fcmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp one <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmone2xdouble(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: fcmone2xdouble:
; Using registers other than v0, v1 are possible, but would be odd.
; ONE = OGT | OLT, OLT implemented as OGT so check reversed operands
; CHECK: fcmgt {{v[0-9]+}}.2d, v0.2d, v1.2d
; CHECK-NEXT: fcmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; todo check reversed operands
   %tmp3 = fcmp one <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <2 x i32> @fcmord2xfloat(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: fcmord2xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; ORD = OGE | OLT, OLT implemented as OGT, so check reversed operands.
; CHECK: fcmge {{v[0-9]+}}.2s, v0.2s, v1.2s
; CHECK-NEXT: fcmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
; CHECK-NEXT: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ord <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}


define <4 x i32> @fcmord4xfloat(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: fcmord4xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; ORD = OGE | OLT, OLT implemented as OGT, so check reversed operands.
; CHECK: fcmge {{v[0-9]+}}.4s, v0.4s, v1.4s
; CHECK-NEXT: fcmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ord <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmord2xdouble(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: fcmord2xdouble:
; Using registers other than v0, v1 are possible, but would be odd.
; ORD = OGE | OLT, OLT implemented as OGT, so check reversed operands.
; CHECK: fcmge {{v[0-9]+}}.2d, v0.2d, v1.2d
; CHECK-NEXT: fcmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ord <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <2 x i32> @fcmuno2xfloat(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: fcmuno2xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; UNO = !(OGE | OLT), OLT implemented as OGT, so check reversed operands.
; CHECK: fcmge {{v[0-9]+}}.2s, v0.2s, v1.2s
; CHECK-NEXT: fcmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
; CHECK-NEXT: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp uno <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmuno4xfloat(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: fcmuno4xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; UNO = !(OGE | OLT), OLT implemented as OGT, so check reversed operands.
; CHECK: fcmge {{v[0-9]+}}.4s, v0.4s, v1.4s
; CHECK-NEXT: fcmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp uno <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmuno2xdouble(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: fcmuno2xdouble:
; Using registers other than v0, v1 are possible, but would be odd.
; UNO = !(OGE | OLT), OLT implemented as OGT, so check reversed operands.
; CHECK: fcmge {{v[0-9]+}}.2d, v0.2d, v1.2d
; CHECK-NEXT: fcmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp uno <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmueq2xfloat(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: fcmueq2xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; UEQ = !ONE = !(OGT | OLT), OLT implemented as OGT so check reversed operands
; CHECK: fcmgt {{v[0-9]+}}.2s, v0.2s, v1.2s
; CHECK-NEXT: fcmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
; CHECK-NEXT: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ueq <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmueq4xfloat(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: fcmueq4xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; UEQ = !ONE = !(OGT | OLT), OLT implemented as OGT so check reversed operands
; CHECK: fcmgt {{v[0-9]+}}.4s, v0.4s, v1.4s
; CHECK-NEXT: fcmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ueq <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmueq2xdouble(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: fcmueq2xdouble:
; Using registers other than v0, v1 are possible, but would be odd.
; UEQ = !ONE = !(OGT | OLT), OLT implemented as OGT so check reversed operands
; CHECK: fcmgt {{v[0-9]+}}.2d, v0.2d, v1.2d
; CHECK-NEXT: fcmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ueq <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmuge2xfloat(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: fcmuge2xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; UGE = ULE with swapped operands, ULE implemented as !OGT.
; CHECK: fcmgt {{v[0-9]+}}.2s, v1.2s, v0.2s
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp uge <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmuge4xfloat(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: fcmuge4xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; UGE = ULE with swapped operands, ULE implemented as !OGT.
; CHECK: fcmgt {{v[0-9]+}}.4s, v1.4s, v0.4s
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp uge <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmuge2xdouble(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: fcmuge2xdouble:
; Using registers other than v0, v1 are possible, but would be odd.
; UGE = ULE with swapped operands, ULE implemented as !OGT.
; CHECK: fcmgt {{v[0-9]+}}.2d, v1.2d, v0.2d
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp uge <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmugt2xfloat(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: fcmugt2xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; UGT = ULT with swapped operands, ULT implemented as !OGE.
; CHECK: fcmge {{v[0-9]+}}.2s, v1.2s, v0.2s
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ugt <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmugt4xfloat(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: fcmugt4xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; UGT = ULT with swapped operands, ULT implemented as !OGE.
; CHECK: fcmge {{v[0-9]+}}.4s, v1.4s, v0.4s
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ugt <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmugt2xdouble(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: fcmugt2xdouble:
; CHECK: fcmge {{v[0-9]+}}.2d, v1.2d, v0.2d
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ugt <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmule2xfloat(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: fcmule2xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; ULE implemented as !OGT.
; CHECK: fcmgt {{v[0-9]+}}.2s, v0.2s, v1.2s
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ule <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmule4xfloat(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: fcmule4xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; ULE implemented as !OGT.
; CHECK: fcmgt {{v[0-9]+}}.4s, v0.4s, v1.4s
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ule <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmule2xdouble(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: fcmule2xdouble:
; Using registers other than v0, v1 are possible, but would be odd.
; ULE implemented as !OGT.
; CHECK: fcmgt {{v[0-9]+}}.2d, v0.2d, v1.2d
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ule <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmult2xfloat(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: fcmult2xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; ULT implemented as !OGE.
; CHECK: fcmge {{v[0-9]+}}.2s, v0.2s, v1.2s
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ult <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmult4xfloat(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: fcmult4xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; ULT implemented as !OGE.
; CHECK: fcmge {{v[0-9]+}}.4s, v0.4s, v1.4s
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ult <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmult2xdouble(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: fcmult2xdouble:
; Using registers other than v0, v1 are possible, but would be odd.
; ULT implemented as !OGE.
; CHECK: fcmge {{v[0-9]+}}.2d, v0.2d, v1.2d
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ult <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmune2xfloat(<2 x float> %A, <2 x float> %B) {
; CHECK-LABEL: fcmune2xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; UNE = !OEQ.
; CHECK: fcmeq {{v[0-9]+}}.2s, v0.2s, v1.2s
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp une <2 x float> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmune4xfloat(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: fcmune4xfloat:
; Using registers other than v0, v1 are possible, but would be odd.
; UNE = !OEQ.
; CHECK: fcmeq {{v[0-9]+}}.4s, v0.4s, v1.4s
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp une <4 x float> %A, %B
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmune2xdouble(<2 x double> %A, <2 x double> %B) {
; CHECK-LABEL: fcmune2xdouble:
; Using registers other than v0, v1 are possible, but would be odd.
; UNE = !OEQ.
; CHECK: fcmeq {{v[0-9]+}}.2d, v0.2d, v1.2d
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp une <2 x double> %A, %B
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmoeqz2xfloat(<2 x float> %A) {
; CHECK-LABEL: fcmoeqz2xfloat:
; CHECK: fcmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
   %tmp3 = fcmp oeq <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmoeqz4xfloat(<4 x float> %A) {
; CHECK-LABEL: fcmoeqz4xfloat:
; CHECK: fcmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
   %tmp3 = fcmp oeq <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmoeqz2xdouble(<2 x double> %A) {
; CHECK-LABEL: fcmoeqz2xdouble:
; CHECK: fcmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
   %tmp3 = fcmp oeq <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <2 x i32> @fcmogez2xfloat(<2 x float> %A) {
; CHECK-LABEL: fcmogez2xfloat:
; CHECK: fcmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
   %tmp3 = fcmp oge <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmogez4xfloat(<4 x float> %A) {
; CHECK-LABEL: fcmogez4xfloat:
; CHECK: fcmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
   %tmp3 = fcmp oge <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmogez2xdouble(<2 x double> %A) {
; CHECK-LABEL: fcmogez2xdouble:
; CHECK: fcmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
   %tmp3 = fcmp oge <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmogtz2xfloat(<2 x float> %A) {
; CHECK-LABEL: fcmogtz2xfloat:
; CHECK: fcmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
   %tmp3 = fcmp ogt <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmogtz4xfloat(<4 x float> %A) {
; CHECK-LABEL: fcmogtz4xfloat:
; CHECK: fcmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
   %tmp3 = fcmp ogt <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmogtz2xdouble(<2 x double> %A) {
; CHECK-LABEL: fcmogtz2xdouble:
; CHECK: fcmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
   %tmp3 = fcmp ogt <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmoltz2xfloat(<2 x float> %A) {
; CHECK-LABEL: fcmoltz2xfloat:
; CHECK: fcmlt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
   %tmp3 = fcmp olt <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmoltz4xfloat(<4 x float> %A) {
; CHECK-LABEL: fcmoltz4xfloat:
; CHECK: fcmlt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
   %tmp3 = fcmp olt <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmoltz2xdouble(<2 x double> %A) {
; CHECK-LABEL: fcmoltz2xdouble:
; CHECK: fcmlt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
   %tmp3 = fcmp olt <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmolez2xfloat(<2 x float> %A) {
; CHECK-LABEL: fcmolez2xfloat:
; CHECK: fcmle {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
   %tmp3 = fcmp ole <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmolez4xfloat(<4 x float> %A) {
; CHECK-LABEL: fcmolez4xfloat:
; CHECK: fcmle {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
   %tmp3 = fcmp ole <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmolez2xdouble(<2 x double> %A) {
; CHECK-LABEL: fcmolez2xdouble:
; CHECK: fcmle {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
   %tmp3 = fcmp ole <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmonez2xfloat(<2 x float> %A) {
; CHECK-LABEL: fcmonez2xfloat:
; ONE with zero = OLT | OGT
; CHECK: fcmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
; CHECK-NEXT: fcmlt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
; CHECK-NEXT: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp one <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmonez4xfloat(<4 x float> %A) {
; CHECK-LABEL: fcmonez4xfloat:
; ONE with zero = OLT | OGT
; CHECK: fcmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
; CHECK-NEXT: fcmlt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp one <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmonez2xdouble(<2 x double> %A) {
; CHECK-LABEL: fcmonez2xdouble:
; ONE with zero = OLT | OGT
; CHECK: fcmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
; CHECK-NEXT: fcmlt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp one <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmordz2xfloat(<2 x float> %A) {
; CHECK-LABEL: fcmordz2xfloat:
; ORD with zero = OLT | OGE
; CHECK: fcmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
; CHECK-NEXT: fcmlt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
; CHECK-NEXT: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ord <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmordz4xfloat(<4 x float> %A) {
; CHECK-LABEL: fcmordz4xfloat:
; ORD with zero = OLT | OGE
; CHECK: fcmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
; CHECK-NEXT: fcmlt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ord <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmordz2xdouble(<2 x double> %A) {
; CHECK-LABEL: fcmordz2xdouble:
; ORD with zero = OLT | OGE
; CHECK: fcmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
; CHECK-NEXT: fcmlt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ord <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmueqz2xfloat(<2 x float> %A) {
; CHECK-LABEL: fcmueqz2xfloat:
; UEQ with zero = !ONE = !(OLT |OGT)
; CHECK: fcmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
; CHECK-NEXT: fcmlt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
; CHECK-NEXT: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ueq <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmueqz4xfloat(<4 x float> %A) {
; CHECK-LABEL: fcmueqz4xfloat:
; UEQ with zero = !ONE = !(OLT |OGT)
; CHECK: fcmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
; CHECK-NEXT: fcmlt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ueq <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmueqz2xdouble(<2 x double> %A) {
; CHECK-LABEL: fcmueqz2xdouble:
; UEQ with zero = !ONE = !(OLT |OGT)
; CHECK: fcmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
; CHECK-NEXT: fcmlt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ueq <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmugez2xfloat(<2 x float> %A) {
; CHECK-LABEL: fcmugez2xfloat:
; UGE with zero = !OLT
; CHECK: fcmlt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp uge <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmugez4xfloat(<4 x float> %A) {
; CHECK-LABEL: fcmugez4xfloat:
; UGE with zero = !OLT
; CHECK: fcmlt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp uge <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmugez2xdouble(<2 x double> %A) {
; CHECK-LABEL: fcmugez2xdouble:
; UGE with zero = !OLT
; CHECK: fcmlt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp uge <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmugtz2xfloat(<2 x float> %A) {
; CHECK-LABEL: fcmugtz2xfloat:
; UGT with zero = !OLE
; CHECK: fcmle {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ugt <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmugtz4xfloat(<4 x float> %A) {
; CHECK-LABEL: fcmugtz4xfloat:
; UGT with zero = !OLE
; CHECK: fcmle {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ugt <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmugtz2xdouble(<2 x double> %A) {
; CHECK-LABEL: fcmugtz2xdouble:
; UGT with zero = !OLE
; CHECK: fcmle {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ugt <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmultz2xfloat(<2 x float> %A) {
; CHECK-LABEL: fcmultz2xfloat:
; ULT with zero = !OGE
; CHECK: fcmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ult <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmultz4xfloat(<4 x float> %A) {
; CHECK-LABEL: fcmultz4xfloat:
; CHECK: fcmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ult <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmultz2xdouble(<2 x double> %A) {
; CHECK-LABEL: fcmultz2xdouble:
; CHECK: fcmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ult <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <2 x i32> @fcmulez2xfloat(<2 x float> %A) {
; CHECK-LABEL: fcmulez2xfloat:
; ULE with zero = !OGT
; CHECK: fcmgt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp ule <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmulez4xfloat(<4 x float> %A) {
; CHECK-LABEL: fcmulez4xfloat:
; ULE with zero = !OGT
; CHECK: fcmgt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ule <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmulez2xdouble(<2 x double> %A) {
; CHECK-LABEL: fcmulez2xdouble:
; ULE with zero = !OGT
; CHECK: fcmgt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp ule <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}

define <2 x i32> @fcmunez2xfloat(<2 x float> %A) {
; CHECK-LABEL: fcmunez2xfloat:
; UNE with zero = !OEQ with zero
; CHECK: fcmeq {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp une <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmunez4xfloat(<4 x float> %A) {
; CHECK-LABEL: fcmunez4xfloat:
; UNE with zero = !OEQ with zero
; CHECK: fcmeq {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp une <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}
define <2 x i64> @fcmunez2xdouble(<2 x double> %A) {
; CHECK-LABEL: fcmunez2xdouble:
; UNE with zero = !OEQ with zero
; CHECK: fcmeq {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp une <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4
}


define <2 x i32> @fcmunoz2xfloat(<2 x float> %A) {
; CHECK-LABEL: fcmunoz2xfloat:
; UNO with zero = !ORD = !(OLT | OGE)
; CHECK: fcmge {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
; CHECK-NEXT: fcmlt {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, #{{0.0|0}}
; CHECK-NEXT: orr {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
   %tmp3 = fcmp uno <2 x float> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i32>
	ret <2 x i32> %tmp4
}

define <4 x i32> @fcmunoz4xfloat(<4 x float> %A) {
; CHECK-LABEL: fcmunoz4xfloat:
; UNO with zero = !ORD = !(OLT | OGE)
; CHECK: fcmge {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
; CHECK-NEXT: fcmlt {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, #{{0.0|0}}
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp uno <4 x float> %A, zeroinitializer
   %tmp4 = sext <4 x i1> %tmp3 to <4 x i32>
	ret <4 x i32> %tmp4
}

define <2 x i64> @fcmunoz2xdouble(<2 x double> %A) {
; CHECK-LABEL: fcmunoz2xdouble:
; UNO with zero = !ORD = !(OLT | OGE)
; CHECK: fcmge {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
; CHECK-NEXT: fcmlt {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, #{{0.0|0}}
; CHECK-NEXT: orr {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
; CHECK-NEXT: {{mvn|not}} {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
   %tmp3 = fcmp uno <2 x double> %A, zeroinitializer
   %tmp4 = sext <2 x i1> %tmp3 to <2 x i64>
	ret <2 x i64> %tmp4

}

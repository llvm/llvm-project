; RUN: llc -mtriple=arm-eabi -float-abi=soft -mattr=+neon %s -o - | FileCheck %s
; RUN: llc -mtriple=arm-eabi -float-abi=soft -mcpu=swift %s -o - | FileCheck %s --check-prefix=SWIFT

; CHECK: t1
; CHECK: vld1.64
; CHECK: vld1.64
; CHECK: vadd.i64 q
; CHECK: vst1.64
; SWIFT: t1
; SWIFT: vld1.64 {{.d[0-9]+, d[0-9]+}, \[r[0-9]+:128\]}}
; SWIFT: vld1.64 {{.d[0-9]+, d[0-9]+}, \[r[0-9]+:128\]}}
; SWIFT: vadd.i64 q
; SWIFT: vst1.64 {{.d[0-9]+, d[0-9]+}, \[r[0-9]+:128\]}}
define void @t1(ptr %r, ptr %a, ptr %b) nounwind {
entry:
	%0 = load <2 x i64>, ptr %a, align 16		; <<2 x i64>> [#uses=1]
	%1 = load <2 x i64>, ptr %b, align 16		; <<2 x i64>> [#uses=1]
	%2 = add <2 x i64> %0, %1		; <<2 x i64>> [#uses=1]
	%3 = bitcast <2 x i64> %2 to <4 x i32>		; <<4 x i32>> [#uses=1]
	store <4 x i32> %3, ptr %r, align 16
	ret void
}

; CHECK: t2
; CHECK: vld1.64
; CHECK: vld1.64
; CHECK: vsub.i64 q
; CHECK: vmov r0, r1, d
; CHECK: vmov r2, r3, d
; SWIFT: t2
; SWIFT: vld1.64 {{.d[0-9]+, d[0-9]+}, \[r[0-9]+:128\]}}
; SWIFT: vld1.64 {{.d[0-9]+, d[0-9]+}, \[r[0-9]+:128\]}}
; SWIFT: vsub.i64 q
; SWIFT: vmov r0, r1, d
; SWIFT: vmov r2, r3, d
define <4 x i32> @t2(ptr %a, ptr %b) nounwind readonly {
entry:
	%0 = load <2 x i64>, ptr %a, align 16		; <<2 x i64>> [#uses=1]
	%1 = load <2 x i64>, ptr %b, align 16		; <<2 x i64>> [#uses=1]
	%2 = sub <2 x i64> %0, %1		; <<2 x i64>> [#uses=1]
	%3 = bitcast <2 x i64> %2 to <4 x i32>		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %3
}

; Limited alignment.
; SWIFT: t3
; SWIFT: vld1.64 {{.d[0-9]+, d[0-9]+}, \[r[0-9]+}}
; SWIFT: vld1.64 {{.d[0-9]+, d[0-9]+}, \[r[0-9]+}}
; SWIFT: vadd.i64 q
; SWIFT: vst1.64 {{.d[0-9]+, d[0-9]+}, \[r[0-9]+}}
define void @t3(ptr %r, ptr %a, ptr %b) nounwind {
entry:
	%0 = load <2 x i64>, ptr %a, align 8
	%1 = load <2 x i64>, ptr %b, align 8
	%2 = add <2 x i64> %0, %1
	%3 = bitcast <2 x i64> %2 to <4 x i32>
	store <4 x i32> %3, ptr %r, align 8
	ret void
}

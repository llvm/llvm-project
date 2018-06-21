; RUN: llc < %s -mtriple=i386-apple-darwin10 -mcpu=corei7-avx -mattr=+avx | FileCheck %s

declare i32 @llvm.x86.avx.ptestz.256(<4 x i64> %p1, <4 x i64> %p2) nounwind
declare i32 @llvm.x86.avx.ptestc.256(<4 x i64> %p1, <4 x i64> %p2) nounwind

define <4 x float> @test1(<4 x i64> %a, <4 x float> %b) nounwind {
entry:
; CHECK-LABEL: test1:
; CHECK: vptest
; CHECK-NEXT:	jne
; CHECK: ret

  %res = call i32 @llvm.x86.avx.ptestz.256(<4 x i64> %a, <4 x i64> %a) nounwind 
  %one = icmp ne i32 %res, 0 
  br i1 %one, label %bb1, label %bb2

bb1:
  %c = fadd <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
  br label %return

bb2:
	%d = fdiv <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
	br label %return

return:
  %e = phi <4 x float> [%c, %bb1], [%d, %bb2]
  ret <4 x float> %e
}

define <4 x float> @test3(<4 x i64> %a, <4 x float> %b) nounwind {
entry:
; CHECK-LABEL: test3:
; CHECK: vptest
; CHECK-NEXT:	jne
; CHECK: ret

  %res = call i32 @llvm.x86.avx.ptestz.256(<4 x i64> %a, <4 x i64> %a) nounwind 
  %one = trunc i32 %res to i1 
  br i1 %one, label %bb1, label %bb2

bb1:
  %c = fadd <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
  br label %return

bb2:
	%d = fdiv <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
	br label %return

return:
  %e = phi <4 x float> [%c, %bb1], [%d, %bb2]
  ret <4 x float> %e
}

define <4 x float> @test4(<4 x i64> %a, <4 x float> %b) nounwind {
entry:
; CHECK-LABEL: test4:
; CHECK: vptest
; CHECK-NEXT:	jae
; CHECK: ret

  %res = call i32 @llvm.x86.avx.ptestc.256(<4 x i64> %a, <4 x i64> %a) nounwind 
  %one = icmp ne i32 %res, 0 
  br i1 %one, label %bb1, label %bb2

bb1:
  %c = fadd <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
  br label %return

bb2:
	%d = fdiv <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
	br label %return

return:
  %e = phi <4 x float> [%c, %bb1], [%d, %bb2]
  ret <4 x float> %e
}

define <4 x float> @test6(<4 x i64> %a, <4 x float> %b) nounwind {
entry:
; CHECK-LABEL: test6:
; CHECK: vptest
; CHECK-NEXT:	jae
; CHECK: ret

  %res = call i32 @llvm.x86.avx.ptestc.256(<4 x i64> %a, <4 x i64> %a) nounwind 
  %one = trunc i32 %res to i1 
  br i1 %one, label %bb1, label %bb2

bb1:
  %c = fadd <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
  br label %return

bb2:
	%d = fdiv <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
	br label %return

return:
  %e = phi <4 x float> [%c, %bb1], [%d, %bb2]
  ret <4 x float> %e
}

define <4 x float> @test7(<4 x i64> %a, <4 x float> %b) nounwind {
entry:
; CHECK-LABEL: test7:
; CHECK: vptest
; CHECK-NEXT:	jne
; CHECK: ret

  %res = call i32 @llvm.x86.avx.ptestz.256(<4 x i64> %a, <4 x i64> %a) nounwind 
  %one = icmp eq i32 %res, 1 
  br i1 %one, label %bb1, label %bb2

bb1:
  %c = fadd <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
  br label %return

bb2:
	%d = fdiv <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
	br label %return

return:
  %e = phi <4 x float> [%c, %bb1], [%d, %bb2]
  ret <4 x float> %e
}

define <4 x float> @test8(<4 x i64> %a, <4 x float> %b) nounwind {
entry:
; CHECK-LABEL: test8:
; CHECK: vptest
; CHECK-NEXT:	je
; CHECK: ret

  %res = call i32 @llvm.x86.avx.ptestz.256(<4 x i64> %a, <4 x i64> %a) nounwind 
  %one = icmp ne i32 %res, 1 
  br i1 %one, label %bb1, label %bb2

bb1:
  %c = fadd <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
  br label %return

bb2:
	%d = fdiv <4 x float> %b, < float 1.000000e+002, float 2.000000e+002, float 3.000000e+002, float 4.000000e+002 >
	br label %return

return:
  %e = phi <4 x float> [%c, %bb1], [%d, %bb2]
  ret <4 x float> %e
}



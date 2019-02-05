; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

define x86_fp80 @foo(x86_fp80 %x) nounwind{
entry:
	%tmp2 = call x86_fp80 @llvm.sqrt.f80( x86_fp80 %x )
	ret x86_fp80 %tmp2
        
; CHECK-LABEL: foo:
; CHECK: fldt 4(%esp)
; CHECK-NEXT: fsqrt
; CHECK-NEXT: ret
}

declare x86_fp80 @llvm.sqrt.f80(x86_fp80)

define x86_fp80 @bar(x86_fp80 %x) nounwind {
entry:
	%tmp2 = call x86_fp80 @llvm.powi.f80( x86_fp80 %x, i32 3 )
	ret x86_fp80 %tmp2
; CHECK-LABEL: bar:
; CHECK: fldt 4(%esp)
; CHECK-NEXT: fld	%st(0)
; CHECK-NEXT: fmul	%st(1)
; CHECK-NEXT: fmulp
; CHECK-NEXT: ret
}

declare x86_fp80 @llvm.powi.f80(x86_fp80, i32)

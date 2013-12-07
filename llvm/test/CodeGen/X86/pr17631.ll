; RUN: llc < %s -mcpu=core-avx-i -mtriple=i386-pc-win32 | FileCheck %s

%struct_type = type { [64 x <8 x float>], <8 x float> }

; Function Attrs: nounwind readnone
declare i32 @llvm.x86.avx.movmsk.ps.256(<8 x float>)

; Function Attrs: nounwind
define i32 @equal(<8 x i32> %A) {
allocas:
  %first_alloc  = alloca [64 x <8 x i32>]
  %second_alloc = alloca %struct_type

  %A1 = bitcast <8 x i32> %A to <8 x float>
  %A2 = call i32 @llvm.x86.avx.movmsk.ps.256(<8 x float> %A1)
  ret i32 %A2
}

; CHECK: equal
; CHECK-NOT: vzeroupper
; CHECK: _chkstk
; CHECK: ret

define <8 x float> @foo(<8 x float> %y, i64* %p, double %x) {
  %i = fptoui double %x to i64
  store i64 %i, i64* %p
  %ret = fadd <8 x float> %y, %y
  ret <8 x float> %ret
}

; CHECK: foo
; CHECK-NOT: vzeroupper
; CHECK: _ftol2
; CHECK: ret

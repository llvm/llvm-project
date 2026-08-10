; RUN: llc < %s -mtriple=x86_64-linux -mcpu=nehalem | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 -mcpu=nehalem | FileCheck %s

; Full strength reduction wouldn't reduce register pressure, so LSR should
; stick with indexing here.

; CHECK: movaps        (%{{rsi|rdx}},%rax,4), [[X3:%xmm[0-9]+]]
; CHECK: cvtdq2ps
; CHECK: orps          {{%xmm[0-9]+}}, [[X4:%xmm[0-9]+]]
; CHECK: movaps        [[X4]], (%{{rdi|rcx}},%rax,4)
; CHECK: addq  $4, %rax
; CHECK: cmpl  %eax, (%{{rdx|r8}})
; CHECK-NEXT: jg

define void @vvfloorf(ptr nocapture %y, ptr nocapture %x, ptr nocapture %n) nounwind {
entry:
  %0 = load i32, ptr %n, align 4
  %1 = icmp sgt i32 %0, 0
  br i1 %1, label %bb, label %return

bb:
  %indvar = phi i64 [ %indvar.next, %bb ], [ 0, %entry ]
  %tmp = shl i64 %indvar, 2
  %scevgep = getelementptr float, ptr %y, i64 %tmp
  %scevgep10 = getelementptr float, ptr %x, i64 %tmp
  %2 = load <4 x float>, ptr %scevgep10, align 16
  %3 = bitcast <4 x float> %2 to <4 x i32>
  %4 = and <4 x i32> %3, <i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647>
  %5 = bitcast <4 x i32> %4 to <4 x float>
  %6 = and <4 x i32> %3, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
  %7 = tail call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %5, <4 x float> <float 8.388608e+06, float 8.388608e+06, float 8.388608e+06, float 8.388608e+06>, i8 5) nounwind
  %tmp.i4 = bitcast <4 x float> %7 to <4 x i32>
  %8 = xor <4 x i32> %tmp.i4, <i32 -1, i32 -1, i32 -1, i32 -1>
  %9 = and <4 x i32> %8, <i32 1258291200, i32 1258291200, i32 1258291200, i32 1258291200>
  %10 = or <4 x i32> %9, %6
  %11 = bitcast <4 x i32> %10 to <4 x float>
  %12 = fadd <4 x float> %2, %11
  %13 = fsub <4 x float> %12, %11
  %14 = tail call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %2, <4 x float> %13, i8 1) nounwind
  %15 = bitcast <4 x float> %14 to <4 x i32>
  %16 = tail call <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32> %15) nounwind readnone
  %17 = fadd <4 x float> %13, %16
  %tmp.i = bitcast <4 x float> %17 to <4 x i32>
  %18 = or <4 x i32> %tmp.i, %6
  %19 = bitcast <4 x i32> %18 to <4 x float>
  store <4 x float> %19, ptr %scevgep, align 16
  %tmp12 = add i64 %tmp, 4
  %tmp13 = trunc i64 %tmp12 to i32
  %20 = load i32, ptr %n, align 4
  %21 = icmp sgt i32 %20, %tmp13
  %indvar.next = add i64 %indvar, 1
  br i1 %21, label %bb, label %return

return:
  ret void
}

declare <4 x float> @llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, i8) nounwind readnone

declare <4 x float> @llvm.x86.sse2.cvtdq2ps(<4 x i32>) nounwind readnone

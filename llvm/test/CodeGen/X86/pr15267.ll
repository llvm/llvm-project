; RUN: llc < %s -mtriple=x86_64-pc-linux -mcpu=corei7-avx | FileCheck %s

define <4 x i3> @test1(<4 x i3>* %in) nounwind {
  %ret = load <4 x i3>* %in, align 1
  ret <4 x i3> %ret
}
; CHECK-LABEL: test1
; CHECK: movzwl
; CHECK: shrl $3
; CHECK: andl $7
; CHECK: andl $7
; CHECK: vmovd
; CHECK: pinsrd $1
; CHECK: shrl $6
; CHECK: andl $7
; CHECK: pinsrd $2
; CHECK: shrl $9
; CHECK: andl $7
; CHECK: pinsrd $3
; CHECK: ret

define <4 x i1> @test2(<4 x i1>* %in) nounwind {
  %ret = load <4 x i1>* %in, align 1
  ret <4 x i1> %ret
}

; CHECK-LABEL: test2
; CHECK: movzbl
; CHECK: shrl
; CHECK: andl $1
; CHECK: andl $1
; CHECK: vmovd
; CHECK: pinsrd $1
; CHECK: shrl $2
; CHECK: andl $1
; CHECK: pinsrd $2
; CHECK: shrl $3
; CHECK: andl $1
; CHECK: pinsrd $3
; CHECK: ret

define <4 x i64> @test3(<4 x i1>* %in) nounwind {
  %wide.load35 = load <4 x i1>* %in, align 1
  %sext = sext <4 x i1> %wide.load35 to <4 x i64>
  ret <4 x i64> %sext
}

; CHECK-LABEL: test3
; CHECK: movzbl
; CHECK: movq
; CHECK: shlq
; CHECK: sarq
; CHECK: vmovq
; CHECK: movq
; CHECK: shlq
; CHECK: sarq
; CHECK: vmovq
; CHECK: vpunpcklqdq
; CHECK: movq
; CHECK: shlq
; CHECK: sarq
; CHECK: vmovq
; CHECK: shlq
; CHECK: sarq
; CHECK: vmovq
; CHECK: vpunpcklqdq
; CHECK: vinsertf128
; CHECK: ret

define <16 x i4> @test4(<16 x i4>* %in) nounwind {
  %ret = load <16 x i4>* %in, align 1
  ret <16 x i4> %ret
}

; CHECK-LABEL: test4
; CHECK: movl
; CHECK-NEXT: shrl
; CHECK-NEXT: andl
; CHECK-NEXT: movl
; CHECK-NEXT: andl
; CHECK-NEXT: vmovd
; CHECK-NEXT: vpinsrb
; CHECK-NEXT: movl
; CHECK-NEXT: shrl
; CHECK-NEXT: andl
; CHECK-NEXT: vpinsrb
; CHECK-NEXT: movl
; CHECK-NEXT: shrl
; CHECK-NEXT: andl
; CHECK-NEXT: vpinsrb
; CHECK-NEXT: movl
; CHECK-NEXT: shrl
; CHECK-NEXT: andl
; CHECK-NEXT: vpinsrb
; CHECK-NEXT: movl
; CHECK-NEXT: shrl
; CHECK-NEXT: andl
; CHECK-NEXT: vpinsrb
; CHECK-NEXT: movl
; CHECK-NEXT: shrl
; CHECK-NEXT: andl
; CHECK-NEXT: vpinsrb
; CHECK-NEXT: movl
; CHECK-NEXT: shrl
; CHECK-NEXT: vpinsrb
; CHECK-NEXT: movq
; CHECK-NEXT: shrq
; CHECK-NEXT: andl
; CHECK-NEXT: vpinsrb
; CHECK-NEXT: movq
; CHECK-NEXT: shrq
; CHECK-NEXT: andl
; CHECK-NEXT: vpinsrb
; CHECK-NEXT: movq
; CHECK-NEXT: shrq
; CHECK-NEXT: andl
; CHECK-NEXT: vpinsrb
; CHECK-NEXT: movq
; CHECK-NEXT: shrq
; CHECK-NEXT: andl
; CHECK-NEXT: vpinsrb
; CHECK-NEXT: movq
; CHECK-NEXT: shrq
; CHECK-NEXT: andl
; CHECK-NEXT: vpinsrb
; CHECK-NEXT: movq
; CHECK-NEXT: shrq
; CHECK-NEXT: andl
; CHECK-NEXT: vpinsrb
; CHECK-NEXT: movq
; CHECK-NEXT: shrq
; CHECK-NEXT: andl
; CHECK-NEXT: vpinsrb
; CHECK-NEXT: shrq
; CHECK-NEXT: vpinsrb
; CHECK-NEXT: retq

; RUN: llc < %s -march=x86    | FileCheck %s -check-prefix=X32
; RUN: llc < %s -march=x86-64 | FileCheck %s -check-prefix=X64

; DAGCombiner crashes during sext folding

define <2 x i256> @test_sext1() {
  %Se = sext <2 x i8> <i8 -100, i8 -99> to <2 x i256>
  %Shuff = shufflevector <2 x i256> zeroinitializer, <2 x i256> %Se, <2 x i32> <i32 1, i32 3>
  ret <2 x i256> %Shuff

  ; X64-LABEL: test_sext1
  ; X64:       movq $-1
  ; X64-NEXT:  movq $-1
  ; X64-NEXT:  movq $-1
  ; X64-NEXT:  movq $-99

  ; X32-LABEL: test_sext1
  ; X32:       movl $-1
  ; X32-NEXT:  movl $-1
  ; X32-NEXT:  movl $-1
  ; X32-NEXT:  movl $-1
  ; X32-NEXT:  movl $-1
  ; X32-NEXT:  movl $-1
  ; X32-NEXT:  movl $-1
  ; X32-NEXT:  movl $-99
}

define <2 x i256> @test_sext2() {
  %Se = sext <2 x i128> <i128 -2000, i128 -1999> to <2 x i256>
  %Shuff = shufflevector <2 x i256> zeroinitializer, <2 x i256> %Se, <2 x i32> <i32 1, i32 3>
  ret <2 x i256> %Shuff

  ; X64-LABEL: test_sext2
  ; X64:       movq $-1
  ; X64-NEXT:  movq $-1
  ; X64-NEXT:  movq $-1
  ; X64-NEXT:  movq $-1999

  ; X32-LABEL: test_sext2
  ; X32:       movl $-1
  ; X32-NEXT:  movl $-1
  ; X32-NEXT:  movl $-1
  ; X32-NEXT:  movl $-1
  ; X32-NEXT:  movl $-1
  ; X32-NEXT:  movl $-1
  ; X32-NEXT:  movl $-1
  ; X32-NEXT:  movl $-1999
}

define <2 x i256> @test_zext1() {
  %Se = zext <2 x i8> <i8 -1, i8 -2> to <2 x i256>
  %Shuff = shufflevector <2 x i256> zeroinitializer, <2 x i256> %Se, <2 x i32> <i32 1, i32 3>
  ret <2 x i256> %Shuff

  ; X64-LABEL: test_zext1
  ; X64:       movq $0
  ; X64-NEXT:  movq $0
  ; X64-NEXT:  movq $0
  ; X64-NEXT:  movq $254

  ; X32-LABEL: test_zext1
  ; X32:       movl $0
  ; X32-NEXT:  movl $0
  ; X32-NEXT:  movl $0
  ; X32-NEXT:  movl $0
  ; X32-NEXT:  movl $0
  ; X32-NEXT:  movl $0
  ; X32-NEXT:  movl $0
  ; X32-NEXT:  movl $254
}

define <2 x i256> @test_zext2() {
  %Se = zext <2 x i128> <i128 -1, i128 -2> to <2 x i256>
  %Shuff = shufflevector <2 x i256> zeroinitializer, <2 x i256> %Se, <2 x i32> <i32 1, i32 3>
  ret <2 x i256> %Shuff

  ; X64-LABEL: test_zext2
  ; X64:       movq $0
  ; X64-NEXT:  movq $0
  ; X64-NEXT:  movq $-1
  ; X64-NEXT:  movq $-2

  ; X32-LABEL: test_zext2
  ; X32:       movl $0
  ; X32-NEXT:  movl $0
  ; X32-NEXT:  movl $0
  ; X32-NEXT:  movl $0
  ; X32-NEXT:  movl $-1
  ; X32-NEXT:  movl $-1
  ; X32-NEXT:  movl $-1
  ; X32-NEXT:  movl $-2
}

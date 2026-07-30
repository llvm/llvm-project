; RUN: llc -mtriple thumbv7---elf -mattr=+neon -filetype asm -o - %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32"
target triple = "thumbv7-elf"

%bar = type { float, float, float }
%baz = type { i32, [16 x %bar], [16 x float], [16 x i32], i8 }
%foo = type { <4 x float> }
%quux = type { ptr, ptr, i32 }
%quuz = type { %quux, i32, %bar, [128 x i8], [16 x %foo], %foo, %foo, %foo }

define void @aaaa(ptr %this, ptr %block) {
entry:
  br i1 undef, label %bb.nph269, label %bb201

bb.nph269:
  br label %bb12

bb12:
  %0 = fmul <4 x float> undef, undef
  %1 = shufflevector <4 x float> %0, <4 x float> undef, <2 x i32> <i32 2, i32 3>
  %2 = shufflevector <2 x float> %1, <2 x float> undef, <4 x i32> zeroinitializer
  %3 = fadd <4 x float> undef, %2
  br i1 undef, label %bb194, label %bb186

bb186:
  br label %bb194

bb194:
  %besterror.0.0 = phi <4 x float> [ %3, %bb186 ], [ undef, %bb12 ]
  %indvar.next294 = add i32 undef, 1
  br label %bb12

bb201:
  ret void
}

; CHECK-NOT: fldmfdd


; RUN: llc < %s -mtriple=i686-pc-win32 -mattr=+sse2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE2
; RUN: llc < %s -mtriple=i686-pc-win32 -mattr=+sse2,+avx | FileCheck %s --check-prefix=CHECK --check-prefix=AVX
; RUN: llc < %s -mtriple=i686-pc-win32 -mattr=+sse2,+avx,+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: llc < %s -mtriple=i686-pc-win32 -mattr=+sse2,+avx,+avx512vl | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512VL

; While we don't support varargs with fastcall, we do support forwarding.

@asdf = internal constant [4 x i8] c"asdf"

declare void @puts(i8*)

define i32 @call_fast_thunk() {
  %r = call x86_fastcallcc i32 (...) @fast_thunk(i32 inreg 1, i32 inreg 2, i32 3)
  ret i32 %r
}

define x86_fastcallcc i32 @fast_thunk(...) {
  call void @puts(i8* getelementptr ([4 x i8], [4 x i8]* @asdf, i32 0, i32 0))
  %r = musttail call x86_fastcallcc i32 (...) bitcast (i32 (i32, i32, i32)* @fast_target to i32 (...)*) (...)
  ret i32 %r
}

; Check that we spill and fill around the call to puts.

; CHECK-LABEL: @fast_thunk@0:
; CHECK-DAG: movl %ecx, {{.*}}
; CHECK-DAG: movl %edx, {{.*}}
; CHECK: calll _puts
; CHECK-DAG: movl {{.*}}, %ecx
; CHECK-DAG: movl {{.*}}, %edx
; CHECK: jmp @fast_target@12

define x86_fastcallcc i32 @fast_target(i32 inreg %a, i32 inreg %b, i32 %c) {
  %a0 = add i32 %a, %b
  %a1 = add i32 %a0, %c
  ret i32 %a1
}

; Repeat the test for vectorcall, which has XMM registers.

define i32 @call_vector_thunk() {
  %r = call x86_vectorcallcc i32 (...) @vector_thunk(i32 inreg 1, i32 inreg 2, i32 3)
  ret i32 %r
}

define x86_vectorcallcc i32 @vector_thunk(...) {
  call void @puts(i8* getelementptr ([4 x i8], [4 x i8]* @asdf, i32 0, i32 0))
  %r = musttail call x86_vectorcallcc i32 (...) bitcast (i32 (i32, i32, i32)* @vector_target to i32 (...)*) (...)
  ret i32 %r
}

; Check that we spill and fill SSE registers around the call to puts.

; CHECK-LABEL: vector_thunk@@0:
; CHECK-DAG: movl %ecx, {{.*}}
; CHECK-DAG: movl %edx, {{.*}}

; SSE2-DAG: movups %xmm0, {{.*}}
; SSE2-DAG: movups %xmm1, {{.*}}
; SSE2-DAG: movups %xmm2, {{.*}}
; SSE2-DAG: movups %xmm3, {{.*}}
; SSE2-DAG: movups %xmm4, {{.*}}
; SSE2-DAG: movups %xmm5, {{.*}}

; AVX-DAG: vmovups %ymm0, {{.*}}
; AVX-DAG: vmovups %ymm1, {{.*}}
; AVX-DAG: vmovups %ymm2, {{.*}}
; AVX-DAG: vmovups %ymm3, {{.*}}
; AVX-DAG: vmovups %ymm4, {{.*}}
; AVX-DAG: vmovups %ymm5, {{.*}}

; AVX512-DAG: vmovups %zmm0, {{.*}}
; AVX512-DAG: vmovups %zmm1, {{.*}}
; AVX512-DAG: vmovups %zmm2, {{.*}}
; AVX512-DAG: vmovups %zmm3, {{.*}}
; AVX512-DAG: vmovups %zmm4, {{.*}}
; AVX512-DAG: vmovups %zmm5, {{.*}}

; CHECK: calll _puts

; SSE2-DAG: movups {{.*}}, %xmm0
; SSE2-DAG: movups {{.*}}, %xmm1
; SSE2-DAG: movups {{.*}}, %xmm2
; SSE2-DAG: movups {{.*}}, %xmm3
; SSE2-DAG: movups {{.*}}, %xmm4
; SSE2-DAG: movups {{.*}}, %xmm5

; AVX-DAG: vmovups {{.*}}, %ymm0
; AVX-DAG: vmovups {{.*}}, %ymm1
; AVX-DAG: vmovups {{.*}}, %ymm2
; AVX-DAG: vmovups {{.*}}, %ymm3
; AVX-DAG: vmovups {{.*}}, %ymm4
; AVX-DAG: vmovups {{.*}}, %ymm5

; AVX512-DAG: vmovups {{.*}}, %zmm0
; AVX512-DAG: vmovups {{.*}}, %zmm1
; AVX512-DAG: vmovups {{.*}}, %zmm2
; AVX512-DAG: vmovups {{.*}}, %zmm3
; AVX512-DAG: vmovups {{.*}}, %zmm4
; AVX512-DAG: vmovups {{.*}}, %zmm5

; CHECK-DAG: movl {{.*}}, %ecx
; CHECK-DAG: movl {{.*}}, %edx
; CHECK: jmp vector_target@@12

define x86_vectorcallcc i32 @vector_target(i32 inreg %a, i32 inreg %b, i32 %c) {
  %a0 = add i32 %a, %b
  %a1 = add i32 %a0, %c
  ret i32 %a1
}

; Repeat the test for vectorcall, which has XMM registers.

define i32 @call_vector_thunk_prefer256() "min-legal-vector-width"="256" "prefer-vector-width"="256" {
  %r = call x86_vectorcallcc i32 (...) @vector_thunk_prefer256(i32 inreg 1, i32 inreg 2, i32 3)
  ret i32 %r
}

define x86_vectorcallcc i32 @vector_thunk_prefer256(...) "min-legal-vector-width"="256" "prefer-vector-width"="256" {
  call void @puts(i8* getelementptr ([4 x i8], [4 x i8]* @asdf, i32 0, i32 0))
  %r = musttail call x86_vectorcallcc i32 (...) bitcast (i32 (i32, i32, i32)* @vector_target_prefer256 to i32 (...)*) (...)
  ret i32 %r
}

; Check that we spill and fill SSE registers around the call to puts.

; CHECK-LABEL: vector_thunk_prefer256@@0:
; CHECK-DAG: movl %ecx, {{.*}}
; CHECK-DAG: movl %edx, {{.*}}

; SSE2-DAG: movups %xmm0, {{.*}}
; SSE2-DAG: movups %xmm1, {{.*}}
; SSE2-DAG: movups %xmm2, {{.*}}
; SSE2-DAG: movups %xmm3, {{.*}}
; SSE2-DAG: movups %xmm4, {{.*}}
; SSE2-DAG: movups %xmm5, {{.*}}

; AVX-DAG: vmovups %ymm0, {{.*}}
; AVX-DAG: vmovups %ymm1, {{.*}}
; AVX-DAG: vmovups %ymm2, {{.*}}
; AVX-DAG: vmovups %ymm3, {{.*}}
; AVX-DAG: vmovups %ymm4, {{.*}}
; AVX-DAG: vmovups %ymm5, {{.*}}

; AVX512F-DAG: vmovups %zmm0, {{.*}}
; AVX512F-DAG: vmovups %zmm1, {{.*}}
; AVX512F-DAG: vmovups %zmm2, {{.*}}
; AVX512F-DAG: vmovups %zmm3, {{.*}}
; AVX512F-DAG: vmovups %zmm4, {{.*}}
; AVX512F-DAG: vmovups %zmm5, {{.*}}

; AVX512VL-DAG: vmovups %ymm0, {{.*}}
; AVX512VL-DAG: vmovups %ymm1, {{.*}}
; AVX512VL-DAG: vmovups %ymm2, {{.*}}
; AVX512VL-DAG: vmovups %ymm3, {{.*}}
; AVX512VL-DAG: vmovups %ymm4, {{.*}}
; AVX512VL-DAG: vmovups %ymm5, {{.*}}

; CHECK: calll _puts

; SSE2-DAG: movups {{.*}}, %xmm0
; SSE2-DAG: movups {{.*}}, %xmm1
; SSE2-DAG: movups {{.*}}, %xmm2
; SSE2-DAG: movups {{.*}}, %xmm3
; SSE2-DAG: movups {{.*}}, %xmm4
; SSE2-DAG: movups {{.*}}, %xmm5

; AVX-DAG: vmovups {{.*}}, %ymm0
; AVX-DAG: vmovups {{.*}}, %ymm1
; AVX-DAG: vmovups {{.*}}, %ymm2
; AVX-DAG: vmovups {{.*}}, %ymm3
; AVX-DAG: vmovups {{.*}}, %ymm4
; AVX-DAG: vmovups {{.*}}, %ymm5

; AVX512F-DAG: vmovups {{.*}}, %zmm0
; AVX512F-DAG: vmovups {{.*}}, %zmm1
; AVX512F-DAG: vmovups {{.*}}, %zmm2
; AVX512F-DAG: vmovups {{.*}}, %zmm3
; AVX512F-DAG: vmovups {{.*}}, %zmm4
; AVX512F-DAG: vmovups {{.*}}, %zmm5

; AVX512VL-DAG: vmovups {{.*}}, %ymm0
; AVX512VL-DAG: vmovups {{.*}}, %ymm1
; AVX512VL-DAG: vmovups {{.*}}, %ymm2
; AVX512VL-DAG: vmovups {{.*}}, %ymm3
; AVX512VL-DAG: vmovups {{.*}}, %ymm4
; AVX512VL-DAG: vmovups {{.*}}, %ymm5

; CHECK-DAG: movl {{.*}}, %ecx
; CHECK-DAG: movl {{.*}}, %edx
; CHECK: jmp vector_target_prefer256@@12

define x86_vectorcallcc i32 @vector_target_prefer256(i32 inreg %a, i32 inreg %b, i32 %c) "min-legal-vector-width"="256" "prefer-vector-width"="256" {
  %a0 = add i32 %a, %b
  %a1 = add i32 %a0, %c
  ret i32 %a1
}

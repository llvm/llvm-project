; Check that scalar FP conversions to signed and unsigned int64 are using
; reasonable sequences, across platforms and target switches.
;
; The signed case is straight forward, and the tests here basically
; ensure successful compilation (f80 with avx512 was broken at one point).
;
; For the unsigned case there are many possible sequences, so to avoid
; a fragile test we just check for the presence of a few key instructions.
; AVX512 on Intel64 can use vcvtts[ds]2usi directly for float and double.
; Otherwise the sequence will involve an FP subtract (fsub, subss or subsd),
; and a truncating conversion (cvtts[ds]2si, fisttp, or fnstcw+fist).  When
; both a subtract and fnstcw are needed, they can occur in either order.
;
; The interesting subtargets are AVX512F (vcvtts[ds]2usi), SSE3 (fisttp),
; SSE2 (cvtts[ds]2si) and vanilla X87 (fnstcw+fist, 32-bit only).
;
; RUN: llc < %s -mtriple=i386-pc-windows-msvc     -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512_32
; RUN: llc < %s -mtriple=i386-unknown-linux-gnu   -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512_32
; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc   -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512_64
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512_64
; RUN: llc < %s -mtriple=i386-pc-windows-msvc     -mattr=+sse3 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE3_32
; RUN: llc < %s -mtriple=i386-unknown-linux-gnu   -mattr=+sse3 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE3_32
; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc   -mattr=+sse3 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE3_64
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+sse3 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE3_64
; RUN: llc < %s -mtriple=i386-pc-windows-msvc     -mattr=+sse2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE2_32
; RUN: llc < %s -mtriple=i386-unknown-linux-gnu   -mattr=+sse2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE2_32
; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc   -mattr=+sse2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE2_64
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+sse2 | FileCheck %s --check-prefix=CHECK --check-prefix=SSE2_64
; RUN: llc < %s -mtriple=i386-pc-windows-msvc     -mattr=-sse  | FileCheck %s --check-prefix=CHECK --check-prefix=X87
; RUN: llc < %s -mtriple=i386-unknown-linux-gnu   -mattr=-sse  | FileCheck %s --check-prefix=CHECK --check-prefix=X87

; CHECK-LABEL: f_to_u64
; X87-DAG: fsub
; X87-DAG: fnstcw
; X87: fist
; SSE2_32-DAG: {{subss|fsub}}
; SSE2_32-DAG: fnstcw
; SSE2_32: fist
; SSE2_64: subss
; SSE2_64: cvttss2si
; SSE3_32: {{subss|fsub}}
; SSE3_32: fistt
; SSE3_64: subss
; SSE3_64: cvttss2si
; AVX512_32: {{subss|fsub}}
; AVX512_32: fistt
; AVX512_64: vcvttss2usi
; CHECK: ret
define i64 @f_to_u64(float %a) nounwind {
  %r = fptoui float %a to i64
  ret i64 %r
}

; CHECK-LABEL: f_to_s64
; X87: fnstcw
; X87: fist
; SSE2_32: fnstcw
; SSE2_32: fist
; SSE2_64: cvttss2si
; SSE3_32: fistt
; SSE3_64: cvttss2si
; AVX512_32: fistt
; AVX512_64: vcvttss2si
; CHECK: ret
define i64 @f_to_s64(float %a) nounwind {
  %r = fptosi float %a to i64
  ret i64 %r
}

; CHECK-LABEL: d_to_u64
; X87-DAG: fsub
; X87-DAG: fnstcw
; X87: fist
; SSE2_32-DAG: {{subsd|fsub}}
; SSE2_32-DAG: fnstcw
; SSE2_32: fist
; SSE2_64: subsd
; SSE2_64: cvttsd2si
; SSE3_32: {{subsd|fsub}}
; SSE3_32: fistt
; SSE3_64: subsd
; SSE3_64: cvttsd2si
; AVX512_32: {{subsd|fsub}}
; AVX512_32: fistt
; AVX512_64: vcvttsd2usi
; CHECK: ret
define i64 @d_to_u64(double %a) nounwind {
  %r = fptoui double %a to i64
  ret i64 %r
}

; CHECK-LABEL: d_to_s64
; X87: fnstcw
; X87: fist
; SSE2_32: fnstcw
; SSE2_32: fist
; SSE2_64: cvttsd2si
; SSE3_32: fistt
; SSE3_64: cvttsd2si
; AVX512_32: fistt
; AVX512_64: vcvttsd2si
; CHECK: ret
define i64 @d_to_s64(double %a) nounwind {
  %r = fptosi double %a to i64
  ret i64 %r
}

; CHECK-LABEL: x_to_u64
; CHECK-DAG: fsub
; X87-DAG: fnstcw
; SSE2_32-DAG: fnstcw
; SSE2_64-DAG: fnstcw
; CHECK: fist
; CHECK: ret
define i64 @x_to_u64(x86_fp80 %a) nounwind {
  %r = fptoui x86_fp80 %a to i64
  ret i64 %r
}

; CHECK-LABEL: x_to_s64
; X87: fnstcw
; X87: fist
; SSE2_32: fnstcw
; SSE2_32: fist
; SSE2_64: fnstcw
; SSE2_64: fist
; SSE3_32: fistt
; SSE3_64: fistt
; AVX512_32: fistt
; AVX512_64: fistt
; CHECK: ret
define i64 @x_to_s64(x86_fp80 %a) nounwind {
  %r = fptosi x86_fp80 %a to i64
  ret i64 %r
}

; CHECK-LABEL: t_to_u64
; CHECK: __fixunstfdi
; CHECK: ret
define i64 @t_to_u64(fp128 %a) nounwind {
  %r = fptoui fp128 %a to i64
  ret i64 %r
}

; CHECK-LABEL: t_to_s64
; CHECK: __fixtfdi
; CHECK: ret
define i64 @t_to_s64(fp128 %a) nounwind {
  %r = fptosi fp128 %a to i64
  ret i64 %r
}

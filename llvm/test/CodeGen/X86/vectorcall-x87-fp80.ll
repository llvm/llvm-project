; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,X64,LINUX64
; RUN: llc -mtriple=x86_64-pc-win32 < %s | FileCheck %s --check-prefixes=CHECK,X64,WIN64
; RUN: llc -mtriple=i686-unknown-linux-gnu -mattr=+sse2 < %s | FileCheck %s --check-prefixes=CHECK,X86,LINUX32
; RUN: llc -mtriple=i686-pc-win32 -mattr=+sse2 < %s | FileCheck %s --check-prefixes=CHECK,X86,WIN32

; An x87 long double is a floating-point type but not a vectorcall vector type,
; so it must not be assigned an XMM register.
; The \01 prefix keeps the vectorcall byte-count suffix out of the symbol names.

define x86_vectorcallcc i32 @"\01f80_arg"(x86_fp80 %a) {
; CHECK-LABEL: f80_arg:
; CHECK-NOT: xmm
; X64: fldt (%rcx)
; LINUX32: fldt 12(%esp)
; WIN32: fldt 8(%ebp)
; CHECK-NOT: xmm
; X64: retq
; LINUX32: retl $12
; WIN32: retl $16
  %r = fptosi x86_fp80 %a to i32
  ret i32 %r
}

define x86_vectorcallcc void @"\01f80_arg_store"(x86_fp80 %a, ptr %p) {
; CHECK-LABEL: f80_arg_store:
; CHECK-NOT: xmm
; X64: fldt (%rcx)
; X64: fstpt (%rdx)
; LINUX32-DAG: fldt 4(%esp)
; LINUX32-DAG: movl 16(%esp), %e[[P:[a-z]+]]
; WIN32-DAG: fldt 8(%ebp)
; WIN32-DAG: movl 24(%ebp), %e[[P:[a-z]+]]
; X86: fstpt (%e[[P]])
; CHECK-NOT: xmm
; X64: retq
; LINUX32: retl $16
; WIN32: retl $20
  store x86_fp80 %a, ptr %p
  ret void
}

declare x86_vectorcallcc void @"\01f80_callee"(x86_fp80)

define void @"\01f80_call"(x86_fp80 %a) {
; CHECK-LABEL: f80_call:
; CHECK-NOT: xmm
; X64: fstpt {{[0-9]*}}(%rsp)
; X86: fstpt (%esp)
; CHECK-NOT: xmm
; CHECK: call{{l|q}} f80_callee
  call x86_vectorcallcc void @"\01f80_callee"(x86_fp80 %a)
  ret void
}

; Returned in FP0; on Win64 the return is demoted to sret.
define x86_vectorcallcc x86_fp80 @"\01f80_ret"(ptr %p) {
; CHECK-LABEL: f80_ret:
; CHECK-NOT: xmm
; LINUX64: fldt (%rcx)
; LINUX64-NOT: fstp
; LINUX64: retq
; WIN64: fldt (%rdx)
; WIN64: fstpt (%r{{cx|ax}})
; WIN64: retq
; X86: fldt (%e{{[a-z]+}})
; X86-NOT: fstp
; X86: retl $4
  %v = load x86_fp80, ptr %p
  ret x86_fp80 %v
}

define x86_vectorcallcc void @"\01f80_mixed"(i32 inreg %i, double %d, x86_fp80 %a, <4 x float> %v, i32 inreg %j, x86_fp80 %b, ptr %p) {
; CHECK-LABEL: f80_mixed:
; CHECK-NOT: xmm
; X64-DAG: fldt (%r8)
; X64-DAG: fldt (%r{{[a-z0-9]+}})
; LINUX32-DAG: fldt 4(%esp)
; LINUX32-DAG: fldt 16(%esp)
; WIN32-DAG: fldt 8(%ebp)
; WIN32-DAG: fldt 24(%ebp)
; CHECK: faddp
; CHECK: fstpt (%{{[er][a-z0-9]+}})
; CHECK-NOT: xmm
; X64: retq
; LINUX32: retl $28
; WIN32: retl $36
  %s = fadd x86_fp80 %a, %b
  store x86_fp80 %s, ptr %p
  ret void
}

define x86_vectorcallcc float @"\01float_control"(float %a, float %b) {
; CHECK-LABEL: float_control:
; CHECK: movaps %xmm1, %xmm0
; CHECK-NEXT: ret{{[lq]}}
  ret float %b
}

define x86_vectorcallcc double @"\01double_control"(double %a, double %b) {
; CHECK-LABEL: double_control:
; CHECK: movaps %xmm1, %xmm0
; CHECK-NEXT: ret{{[lq]}}
  ret double %b
}

define x86_vectorcallcc <4 x float> @"\01vector_control"(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: vector_control:
; CHECK: movaps %xmm1, %xmm0
; CHECK-NEXT: ret{{[lq]}}
  ret <4 x float> %b
}

%struct.HVA2 = type { <4 x float>, <4 x float> }

define x86_vectorcallcc <4 x float> @"\01hva_control"(%struct.HVA2 inreg %h) {
; CHECK-LABEL: hva_control:
; CHECK: movaps %xmm1, %xmm0
; CHECK-NEXT: ret{{[lq]}}
  %e = extractvalue %struct.HVA2 %h, 1
  ret <4 x float> %e
}

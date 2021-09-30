; RUN: llc -mtriple=i686-pc-win32 -mattr=+sse2 < %s | FileCheck %s --check-prefix=CHECK --check-prefix=X86
; RUN: llc -mtriple=x86_64-pc-win32 < %s | FileCheck %s --check-prefix=CHECK --check-prefix=X64

; Test integer arguments.

define x86_vectorcallcc i32 @test_int_1() {
; CHECK-LABEL: {{^}}test_int_1@@0:
; CHECK: xorl %eax, %eax
  ret i32 0
}

define x86_vectorcallcc i32 @test_int_2(i32 inreg %a) {
; X86-LABEL: {{^}}test_int_2@@4:
; X64-LABEL: {{^}}test_int_2@@8:
; CHECK: movl %ecx, %eax
  ret i32 %a
}

define x86_vectorcallcc i32 @test_int_3(i64 inreg %a) {
; X86-LABEL: {{^}}test_int_3@@8:
; X64-LABEL: {{^}}test_int_3@@8:
; X86: movl %ecx, %eax
; X64: movq %rcx, %rax
  %at = trunc i64 %a to i32
  ret i32 %at
}

define x86_vectorcallcc i32 @test_int_4(i32 inreg %a, i32 inreg %b) {
; X86-LABEL: {{^}}test_int_4@@8:
; X86: leal (%ecx,%edx), %eax
; X64-LABEL: {{^}}test_int_4@@16:
; X64: leal (%rcx,%rdx), %eax
  %s = add i32 %a, %b
  ret i32 %s
}

define x86_vectorcallcc i32 @"\01test_int_5"(i32, i32) {
; CHECK-LABEL: {{^}}test_int_5:
  ret i32 0
}

define x86_vectorcallcc double @test_fp_1(double %a, double %b) {
; CHECK-LABEL: {{^}}test_fp_1@@16:
; CHECK: movaps %xmm1, %xmm0
  ret double %b
}

define x86_vectorcallcc double @test_fp_2(double, double, double, double, double, double, double %r) {
; CHECK-LABEL: {{^}}test_fp_2@@56:
; CHECK: movsd {{[0-9]+\(%[re]sp\)}}, %xmm0
  ret double %r
}

define x86_vectorcallcc {double, double, double, double} @test_fp_3() {
; CHECK-LABEL: {{^}}test_fp_3@@0:
; CHECK: xorps %xmm0
; CHECK: xorps %xmm1
; CHECK: xorps %xmm2
; CHECK: xorps %xmm3
  ret {double, double, double, double}
        { double 0.0, double 0.0, double 0.0, double 0.0 }
}

; FIXME: Returning via x87 isn't compatible, but its hard to structure the
; tablegen any other way.
define x86_vectorcallcc {double, double, double, double, double} @test_fp_4() {
; CHECK-LABEL: {{^}}test_fp_4@@0:
; CHECK: fldz
; CHECK: xorps %xmm0
; CHECK: xorps %xmm1
; CHECK: xorps %xmm2
; CHECK: xorps %xmm3
  ret {double, double, double, double, double}
        { double 0.0, double 0.0, double 0.0, double 0.0, double 0.0 }
}

define x86_vectorcallcc <16 x i8> @test_vec_1(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: {{^}}test_vec_1@@32:
; CHECK: movaps %xmm1, %xmm0
  ret <16 x i8> %b
}

define x86_vectorcallcc <16 x i8> @test_vec_2(double, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> %r) {
; CHECK-LABEL: {{^}}test_vec_2@@104:
; X64:           movq    {{[0-9]*}}(%rsp), %rax
; CHECK:         movaps (%{{rax|ecx}}), %xmm0
  ret <16 x i8> %r
}

%struct.HVA5 = type { <4 x float>, <4 x float>, <4 x float>, <4 x float>, <4 x float> }
%struct.HVA4 = type { <4 x float>, <4 x float>, <4 x float>, <4 x float> }
%struct.HVA3 = type { <4 x float>, <4 x float>, <4 x float> }
%struct.HVA2 = type { <4 x float>, <4 x float> }

define x86_vectorcallcc <4 x float> @test_mixed_1(i32 %a, %struct.HVA4 inreg %bb, i32 %c) {
; CHECK-LABEL: test_mixed_1
; CHECK:       movaps	%xmm1, 16(%{{(e|r)}}sp)
; CHECK:       movaps	%xmm1, %xmm0
; CHECK:       ret{{q|l}}
entry:
  %b = alloca %struct.HVA4, align 16
  store %struct.HVA4 %bb, %struct.HVA4* %b, align 16
  %w1 = getelementptr inbounds %struct.HVA4, %struct.HVA4* %b, i32 0, i32 1
  %0 = load <4 x float>, <4 x float>* %w1, align 16
  ret <4 x float> %0
}

define x86_vectorcallcc <4 x float> @test_mixed_2(%struct.HVA4 inreg %a, %struct.HVA4* %b, <4 x float> %c) {
; CHECK-LABEL: test_mixed_2
; X86:         movaps  %xmm0, (%esp)
; X64:         movaps  %xmm2, %xmm0
; CHECK:       ret{{[ql]}}
entry:
  %c.addr = alloca <4 x float>, align 16
  store <4 x float> %c, <4 x float>* %c.addr, align 16
  %0 = load <4 x float>, <4 x float>* %c.addr, align 16
  ret <4 x float> %0
}

define x86_vectorcallcc <4 x float> @test_mixed_3(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %d, <4 x float> %e, %struct.HVA2* %f) {
; CHECK-LABEL: test_mixed_3
; CHECK:       movaps	(%{{[re][ac]}}x), %xmm0
; CHECK:       ret{{[ql]}}
entry:
  %x = getelementptr inbounds %struct.HVA2, %struct.HVA2* %f, i32 0, i32 0
  %0 = load <4 x float>, <4 x float>* %x, align 16
  ret <4 x float> %0
}

define x86_vectorcallcc <4 x float> @test_mixed_4(%struct.HVA4 inreg %a, %struct.HVA2* %bb, <4 x float> %c) {
; CHECK-LABEL: test_mixed_4
; X86:         movaps	16(%eax), %xmm0
; X64:         movaps	16(%rdx), %xmm0
; CHECK:       ret{{[ql]}}
entry:
  %y4 = getelementptr inbounds %struct.HVA2, %struct.HVA2* %bb, i32 0, i32 1
  %0 = load <4 x float>, <4 x float>* %y4, align 16
  ret <4 x float> %0
}

define x86_vectorcallcc <4 x float> @test_mixed_5(%struct.HVA3 inreg %a, %struct.HVA3* %b, <4 x float> %c, %struct.HVA2 inreg %dd) {
; CHECK-LABEL: test_mixed_5
; CHECK-DAG:   movaps	%xmm{{[0,5]}}, 16(%{{(e|r)}}sp)
; CHECK-DAG:   movaps	%xmm5, %xmm0
; CHECK:       ret{{[ql]}}
entry:
  %d = alloca %struct.HVA2, align 16
  store %struct.HVA2 %dd, %struct.HVA2* %d, align 16
  %y5 = getelementptr inbounds %struct.HVA2, %struct.HVA2* %d, i32 0, i32 1
  %0 = load <4 x float>, <4 x float>* %y5, align 16
  ret <4 x float> %0
}

define x86_vectorcallcc %struct.HVA4 @test_mixed_6(%struct.HVA4 inreg %a, %struct.HVA4* %b) {
; CHECK-LABEL: test_mixed_6
; CHECK:       movaps	(%{{[re]}}sp), %xmm0
; CHECK:       movaps	16(%{{[re]}}sp), %xmm1
; CHECK:       movaps	32(%{{[re]}}sp), %xmm2
; CHECK:       movaps	48(%{{[re]}}sp), %xmm3
; CHECK:       ret{{[ql]}}
entry:
  %retval = alloca %struct.HVA4, align 16
  %0 = bitcast %struct.HVA4* %retval to i8*
  %1 = bitcast %struct.HVA4* %b to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 16 %0, i8* align 16 %1, i32 64, i1 false)
  %2 = load %struct.HVA4, %struct.HVA4* %retval, align 16
  ret %struct.HVA4 %2
}

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1)

define x86_vectorcallcc void @test_mixed_7(%struct.HVA5* noalias sret(%struct.HVA5) %agg.result) {
; CHECK-LABEL: test_mixed_7@@0
; X64:         mov{{[ql]}}	%rcx, %rax
; CHECK:       movaps	%xmm{{[0-9]}}, 64(%{{rcx|eax}})
; CHECK:       movaps	%xmm{{[0-9]}}, 48(%{{rcx|eax}})
; CHECK:       movaps	%xmm{{[0-9]}}, 32(%{{rcx|eax}})
; CHECK:       movaps	%xmm{{[0-9]}}, 16(%{{rcx|eax}})
; CHECK:       movaps	%xmm{{[0-9]}}, (%{{rcx|eax}})
; CHECK:       ret{{[ql]}}
entry:
  %a = alloca %struct.HVA5, align 16
  %0 = bitcast %struct.HVA5* %a to i8*
  call void @llvm.memset.p0i8.i64(i8* align 16 %0, i8 0, i64 80, i1 false)
  %1 = bitcast %struct.HVA5* %agg.result to i8*
  %2 = bitcast %struct.HVA5* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %1, i8* align 16 %2, i64 80, i1 false)
  ret void
}

define x86_vectorcallcc <4 x float> @test_mixed_8(<4 x float> %a, <4 x float> %b, <4 x float> %c, <4 x float> %d, i32 %e, <4 x float> %f) {
; CHECK-LABEL: test_mixed_8
; X86:         movaps	%xmm4, %xmm0
; X64:         movaps	%xmm5, %xmm0
; CHECK:       ret{{[ql]}}
entry:
  %f.addr = alloca <4 x float>, align 16
  store <4 x float> %f, <4 x float>* %f.addr, align 16
  %0 = load <4 x float>, <4 x float>* %f.addr, align 16
  ret <4 x float> %0
}

%struct.HFA4 = type { double, double, double, double }
declare x86_vectorcallcc double @test_mixed_9_callee(%struct.HFA4 %x, double %y)

define x86_vectorcallcc double @test_mixed_9_caller(%struct.HFA4 inreg %b) {
; CHECK-LABEL: test_mixed_9_caller
; CHECK:       movaps  %xmm3, %xmm4
; CHECK:       movaps  %xmm2, %xmm3
; CHECK:       movaps  %xmm1, %xmm2
; X32:         movasd  %xmm0, %xmm1
; X64:         movap{{d|s}}  %xmm5, %xmm1
; CHECK:       call{{l|q}}   test_mixed_9_callee@@40
; CHECK:       addsd   {{.*}}, %xmm0
; CHECK:       ret{{l|q}}
entry:
  %call = call x86_vectorcallcc double @test_mixed_9_callee(%struct.HFA4 inreg %b, double 3.000000e+00)
  %add = fadd double 1.000000e+00, %call
  ret double %add
}

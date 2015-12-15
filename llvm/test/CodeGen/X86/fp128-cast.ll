; RUN: llc < %s -O2 -mtriple=x86_64-linux-android -mattr=+mmx | FileCheck %s
; RUN: llc < %s -O2 -mtriple=x86_64-linux-gnu -mattr=+mmx | FileCheck %s

; Check soft floating point conversion function calls.

@vi32 = common global i32 0, align 4
@vi64 = common global i64 0, align 8
@vu32 = common global i32 0, align 4
@vu64 = common global i64 0, align 8
@vf32 = common global float 0.000000e+00, align 4
@vf64 = common global double 0.000000e+00, align 8
@vf128 = common global fp128 0xL00000000000000000000000000000000, align 16

define void @TestFPExtF32_F128() {
entry:
  %0 = load float, float* @vf32, align 4
  %conv = fpext float %0 to fp128
  store fp128 %conv, fp128* @vf128, align 16
  ret void
; CHECK-LABEL: TestFPExtF32_F128:
; CHECK:       movss      vf32(%rip), %xmm0
; CHECK-NEXT:  callq      __extendsftf2
; CHECK-NEXT:  movaps     %xmm0, vf128(%rip)
; CHECK:       retq
}

define void @TestFPExtF64_F128() {
entry:
  %0 = load double, double* @vf64, align 8
  %conv = fpext double %0 to fp128
  store fp128 %conv, fp128* @vf128, align 16
  ret void
; CHECK-LABEL: TestFPExtF64_F128:
; CHECK:       movsd      vf64(%rip), %xmm0
; CHECK-NEXT:  callq      __extenddftf2
; CHECK-NEXT:  movapd     %xmm0, vf128(%rip)
; CHECK:       ret
}

define void @TestFPToSIF128_I32() {
entry:
  %0 = load fp128, fp128* @vf128, align 16
  %conv = fptosi fp128 %0 to i32
  store i32 %conv, i32* @vi32, align 4
  ret void
; CHECK-LABEL: TestFPToSIF128_I32:
; CHECK:        movaps     vf128(%rip), %xmm0
; CHECK-NEXT:   callq      __fixtfsi
; CHECK-NEXT:   movl       %eax, vi32(%rip)
; CHECK:        retq
}

define void @TestFPToUIF128_U32() {
entry:
  %0 = load fp128, fp128* @vf128, align 16
  %conv = fptoui fp128 %0 to i32
  store i32 %conv, i32* @vu32, align 4
  ret void
; CHECK-LABEL: TestFPToUIF128_U32:
; CHECK:        movaps     vf128(%rip), %xmm0
; CHECK-NEXT:   callq      __fixunstfsi
; CHECK-NEXT:   movl       %eax, vu32(%rip)
; CHECK:        retq
}

define void @TestFPToSIF128_I64() {
entry:
  %0 = load fp128, fp128* @vf128, align 16
  %conv = fptosi fp128 %0 to i32
  %conv1 = sext i32 %conv to i64
  store i64 %conv1, i64* @vi64, align 8
  ret void
; CHECK-LABEL: TestFPToSIF128_I64:
; CHECK:       movaps      vf128(%rip), %xmm0
; CHECK-NEXT:  callq       __fixtfsi
; CHECK-NEXT:  cltq
; CHECK-NEXT:  movq        %rax, vi64(%rip)
; CHECK:       retq
}

define void @TestFPToUIF128_U64() {
entry:
  %0 = load fp128, fp128* @vf128, align 16
  %conv = fptoui fp128 %0 to i32
  %conv1 = zext i32 %conv to i64
  store i64 %conv1, i64* @vu64, align 8
  ret void
; CHECK-LABEL: TestFPToUIF128_U64:
; CHECK:       movaps      vf128(%rip), %xmm0
; CHECK-NEXT:  callq       __fixunstfsi
; CHECK-NEXT:  movl        %eax, %eax
; CHECK-NEXT:  movq        %rax, vu64(%rip)
; CHECK:       retq
}

define void @TestFPTruncF128_F32() {
entry:
  %0 = load fp128, fp128* @vf128, align 16
  %conv = fptrunc fp128 %0 to float
  store float %conv, float* @vf32, align 4
  ret void
; CHECK-LABEL: TestFPTruncF128_F32:
; CHECK:       movaps      vf128(%rip), %xmm0
; CHECK-NEXT:  callq       __trunctfsf2
; CHECK-NEXT:  movss       %xmm0, vf32(%rip)
; CHECK:       retq
}

define void @TestFPTruncF128_F64() {
entry:
  %0 = load fp128, fp128* @vf128, align 16
  %conv = fptrunc fp128 %0 to double
  store double %conv, double* @vf64, align 8
  ret void
; CHECK-LABEL: TestFPTruncF128_F64:
; CHECK:       movapd      vf128(%rip), %xmm0
; CHECK-NEXT:  callq       __trunctfdf2
; CHECK-NEXT:  movsd       %xmm0, vf64(%rip)
; CHECK:       retq
}

define void @TestSIToFPI32_F128() {
entry:
  %0 = load i32, i32* @vi32, align 4
  %conv = sitofp i32 %0 to fp128
  store fp128 %conv, fp128* @vf128, align 16
  ret void
; CHECK-LABEL: TestSIToFPI32_F128:
; CHECK:       movl       vi32(%rip), %edi
; CHECK-NEXT:  callq      __floatsitf
; CHECK-NEXT:  movaps     %xmm0, vf128(%rip)
; CHECK:       retq
}

define void @TestUIToFPU32_F128() #2 {
entry:
  %0 = load i32, i32* @vu32, align 4
  %conv = uitofp i32 %0 to fp128
  store fp128 %conv, fp128* @vf128, align 16
  ret void
; CHECK-LABEL: TestUIToFPU32_F128:
; CHECK:       movl       vu32(%rip), %edi
; CHECK-NEXT:  callq      __floatunsitf
; CHECK-NEXT:  movaps     %xmm0, vf128(%rip)
; CHECK:       retq
}

define void @TestSIToFPI64_F128(){
entry:
  %0 = load i64, i64* @vi64, align 8
  %conv = sitofp i64 %0 to fp128
  store fp128 %conv, fp128* @vf128, align 16
  ret void
; CHECK-LABEL: TestSIToFPI64_F128:
; CHECK:       movq       vi64(%rip), %rdi
; CHECK-NEXT:  callq      __floatditf
; CHECK-NEXT:  movaps     %xmm0, vf128(%rip)
; CHECK:       retq
}

define void @TestUIToFPU64_F128() #2 {
entry:
  %0 = load i64, i64* @vu64, align 8
  %conv = uitofp i64 %0 to fp128
  store fp128 %conv, fp128* @vf128, align 16
  ret void
; CHECK-LABEL: TestUIToFPU64_F128:
; CHECK:       movq       vu64(%rip), %rdi
; CHECK-NEXT:  callq      __floatunditf
; CHECK-NEXT:  movaps     %xmm0, vf128(%rip)
; CHECK:       retq
}

define i32 @TestConst128(fp128 %v) {
entry:
  %cmp = fcmp ogt fp128 %v, 0xL00000000000000003FFF000000000000
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: TestConst128:
; CHECK:       movaps {{.*}}, %xmm1
; CHECK-NEXT:  callq __gttf2
; CHECK-NEXT:  test
; CHECK:       retq
}

; C code:
;  struct TestBits_ieee_ext {
;    unsigned v1;
;    unsigned v2;
; };
; union TestBits_LDU {
;   FP128 ld;
;   struct TestBits_ieee_ext bits;
; };
; int TestBits128(FP128 ld) {
;   union TestBits_LDU u;
;   u.ld = ld * ld;
;   return ((u.bits.v1 | u.bits.v2)  == 0);
; }
define i32 @TestBits128(fp128 %ld) {
entry:
  %mul = fmul fp128 %ld, %ld
  %0 = bitcast fp128 %mul to i128
  %shift = lshr i128 %0, 32
  %or5 = or i128 %shift, %0
  %or = trunc i128 %or5 to i32
  %cmp = icmp eq i32 %or, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: TestBits128:
; CHECK:       movaps %xmm0, %xmm1
; CHECK-NEXT:  callq __multf3
; CHECK-NEXT:  movaps %xmm0, (%rsp)
; CHECK-NEXT:  movq (%rsp),
; CHECK-NEXT:  movq %
; CHECK-NEXT:  shrq $32,
; CHECK:       orl
; CHECK-NEXT:  sete %al
; CHECK-NEXT:  movzbl %al, %eax
; CHECK:       retq
;
; If TestBits128 fails due to any llvm or clang change,
; please make sure the original simplified C code will
; be compiled into correct IL and assembly code, not
; just this TestBits128 test case. Better yet, try to
; test the whole libm and its test cases.
}

; C code: (compiled with -target x86_64-linux-android)
; typedef long double __float128;
; __float128 TestPair128(unsigned long a, unsigned long b) {
;   unsigned __int128 n;
;   unsigned __int128 v1 = ((unsigned __int128)a << 64);
;   unsigned __int128 v2 = (unsigned __int128)b;
;   n = (v1 | v2) + 3;
;   return *(__float128*)&n;
; }
define fp128 @TestPair128(i64 %a, i64 %b) {
entry:
  %conv = zext i64 %a to i128
  %shl = shl nuw i128 %conv, 64
  %conv1 = zext i64 %b to i128
  %or = or i128 %shl, %conv1
  %add = add i128 %or, 3
  %0 = bitcast i128 %add to fp128
  ret fp128 %0
; CHECK-LABEL: TestPair128:
; CHECK:       addq $3, %rsi
; CHECK:       movq %rsi, -24(%rsp)
; CHECK:       movq %rdi, -16(%rsp)
; CHECK:       movaps -24(%rsp), %xmm0
; CHECK-NEXT:  retq
}

define fp128 @TestTruncCopysign(fp128 %x, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 50000
  br i1 %cmp, label %if.then, label %cleanup

if.then:                                          ; preds = %entry
  %conv = fptrunc fp128 %x to double
  %call = tail call double @copysign(double 0x7FF0000000000000, double %conv) #2
  %conv1 = fpext double %call to fp128
  br label %cleanup

cleanup:                                          ; preds = %entry, %if.then
  %retval.0 = phi fp128 [ %conv1, %if.then ], [ %x, %entry ]
  ret fp128 %retval.0
; CHECK-LABEL: TestTruncCopysign:
; CHECK:       callq __trunctfdf2
; CHECK-NEXT:  andpd {{.*}}, %xmm0
; CHECK-NEXT:  orpd {{.*}}, %xmm0
; CHECK-NEXT:  callq __extenddftf2
; CHECK:       retq
}

declare double @copysign(double, double) #1

attributes #2 = { nounwind readnone }

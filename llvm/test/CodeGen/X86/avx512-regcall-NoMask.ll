; RUN: llc < %s -mtriple=i386-pc-win32       -mattr=+avx512f -mattr=+avx512vl -mattr=+avx512bw -mattr=+avx512dq  | FileCheck --check-prefix=X32 %s
; RUN: llc < %s -mtriple=x86_64-win32        -mattr=+avx512f -mattr=+avx512vl -mattr=+avx512bw -mattr=+avx512dq  | FileCheck --check-prefix=WIN64 %s
; RUN: llc < %s -mtriple=x86_64-linux-gnu    -mattr=+avx512f -mattr=+avx512vl -mattr=+avx512bw -mattr=+avx512dq  | FileCheck --check-prefix=LINUXOSX64 %s 

; X32-LABEL:  test_argReti1:
; X32:        kmov{{.*}}  %eax, %k{{[0-7]}}
; X32:        kmov{{.*}}  %k{{[0-7]}}, %eax
; X32:        ret{{.*}}

; WIN64-LABEL:  test_argReti1:
; WIN64:        kmov{{.*}}  %eax, %k{{[0-7]}}
; WIN64:        kmov{{.*}}  %k{{[0-7]}}, %eax
; WIN64:        ret{{.*}}

; Test regcall when receiving/returning i1
define x86_regcallcc i1 @test_argReti1(i1 %a)  {
  %add = add i1 %a, 1
  ret i1 %add
}

; X32-LABEL:  test_CallargReti1:
; X32:        kmov{{.*}}  %k{{[0-7]}}, %eax
; X32:        call{{.*}}   {{.*}}test_argReti1
; X32:        kmov{{.*}}  %eax, %k{{[0-7]}}
; X32:        ret{{.*}}

; WIN64-LABEL:  test_CallargReti1:
; WIN64:        kmov{{.*}}  %k{{[0-7]}}, %eax
; WIN64:        call{{.*}}   {{.*}}test_argReti1
; WIN64:        kmov{{.*}}  %eax, %k{{[0-7]}}
; WIN64:        ret{{.*}}

; Test regcall when passing/retrieving i1
define x86_regcallcc i1 @test_CallargReti1(i1 %a)  {
  %b = add i1 %a, 1
  %c = call x86_regcallcc i1 @test_argReti1(i1 %b)
  %d = add i1 %c, 1
  ret i1 %d
}

; X32-LABEL:  test_argReti8:
; X32:        incb  %al
; X32:        ret{{.*}}

; WIN64-LABEL:  test_argReti8:
; WIN64:        incb %al
; WIN64:        ret{{.*}}

; Test regcall when receiving/returning i8
define x86_regcallcc i8 @test_argReti8(i8 %a)  {
  %add = add i8 %a, 1
  ret i8 %add
}

; X32-LABEL:  test_CallargReti8:
; X32:        incb %al
; X32:        call{{.*}}   {{.*}}test_argReti8
; X32:        incb %al
; X32:        ret{{.*}}

; WIN64-LABEL:  test_CallargReti8:
; WIN64:        incb %al
; WIN64:        call{{.*}}   {{.*}}test_argReti8
; WIN64:        incb %al
; WIN64:        ret{{.*}}

; Test regcall when passing/retrieving i8
define x86_regcallcc i8 @test_CallargReti8(i8 %a)  {
  %b = add i8 %a, 1
  %c = call x86_regcallcc i8 @test_argReti8(i8 %b)
  %d = add i8 %c, 1
  ret i8 %d
}

; X32-LABEL:  test_argReti16:
; X32:        incl %eax
; X32:        ret{{.*}}

; WIN64-LABEL:  test_argReti16:
; WIN64:        incl %eax
; WIN64:        ret{{.*}}

; Test regcall when receiving/returning i16
define x86_regcallcc i16 @test_argReti16(i16 %a)  {
  %add = add i16 %a, 1
  ret i16 %add
}

; X32-LABEL:  test_CallargReti16:
; X32:        incl %eax
; X32:        call{{.*}}   {{.*}}test_argReti16
; X32:        incl %eax
; X32:        ret{{.*}}

; WIN64-LABEL:  test_CallargReti16:
; WIN64:        incl %eax
; WIN64:        call{{.*}}   {{.*}}test_argReti16
; WIN64:        incl %eax
; WIN64:        ret{{.*}}

; Test regcall when passing/retrieving i16
define x86_regcallcc i16 @test_CallargReti16(i16 %a)  {
  %b = add i16 %a, 1
  %c = call x86_regcallcc i16 @test_argReti16(i16 %b)
  %d = add i16 %c, 1
  ret i16 %d
}

; X32-LABEL:  test_argReti32:
; X32:        incl %eax
; X32:        ret{{.*}}

; WIN64-LABEL:  test_argReti32:
; WIN64:        incl %eax
; WIN64:        ret{{.*}}

; Test regcall when receiving/returning i32
define x86_regcallcc i32 @test_argReti32(i32 %a)  {
  %add = add i32 %a, 1
  ret i32 %add
}

; X32-LABEL:  test_CallargReti32:
; X32:        incl %eax
; X32:        call{{.*}}   {{.*}}test_argReti32
; X32:        incl %eax
; X32:        ret{{.*}}

; WIN64-LABEL:  test_CallargReti32:
; WIN64:        incl %eax
; WIN64:        call{{.*}}   {{.*}}test_argReti32
; WIN64:        incl %eax
; WIN64:        ret{{.*}}

; Test regcall when passing/retrieving i32
define x86_regcallcc i32 @test_CallargReti32(i32 %a)  {
  %b = add i32 %a, 1
  %c = call x86_regcallcc i32 @test_argReti32(i32 %b)
  %d = add i32 %c, 1
  ret i32 %d
}

; X32-LABEL:  test_argReti64:
; X32:        addl $3, %eax
; X32:        adcl $1, %ecx
; X32:        ret{{.*}}

; WIN64-LABEL:  test_argReti64:
; WIN64:        movabsq $4294967299, %r{{.*}}
; WIN64:        addq %r{{.*}}, %rax
; WIN64:        ret{{.*}}

; Test regcall when receiving/returning i64
define x86_regcallcc i64 @test_argReti64(i64 %a)  {
  %add = add i64 %a, 4294967299
  ret i64 %add
}

; X32-LABEL:  test_CallargReti64:
; X32:        add{{.*}}  $1, %eax
; X32:        adcl   $0, {{%e(cx|dx|si|di|bx|bp)}}
; X32:        call{{.*}}   {{.*}}test_argReti64
; X32:        add{{.*}}  $1, %eax
; X32:        adcl   $0, {{%e(cx|dx|si|di|bx|bp)}}
; X32:        ret{{.*}}

; WIN64-LABEL:  test_CallargReti64:
; WIN64:        incq %rax
; WIN64:        call{{.*}}   {{.*}}test_argReti64
; WIN64:        incq %rax
; WIN64:        ret{{.*}}

; Test regcall when passing/retrieving i64
define x86_regcallcc i64 @test_CallargReti64(i64 %a)  {
  %b = add i64 %a, 1
  %c = call x86_regcallcc i64 @test_argReti64(i64 %b)
  %d = add i64 %c, 1
  ret i64 %d
}

; X32-LABEL:  test_argRetFloat:
; X32:        vadd{{.*}}  {{.*}}, %xmm0
; X32:        ret{{.*}}

; WIN64-LABEL:  test_argRetFloat:
; WIN64:        vadd{{.*}}  {{.*}}, %xmm0
; WIN64:        ret{{.*}}

; Test regcall when receiving/returning float
define x86_regcallcc float @test_argRetFloat(float %a)  {
  %add = fadd float 1.0, %a
  ret float %add
}

; X32-LABEL:  test_CallargRetFloat:
; X32:        vadd{{.*}}  {{%xmm([0-7])}}, %xmm0, %xmm0
; X32:        call{{.*}}   {{.*}}test_argRetFloat
; X32:        vadd{{.*}}  {{%xmm([0-7])}}, %xmm0, %xmm0
; X32:        ret{{.*}}

; WIN64-LABEL:  test_CallargRetFloat:
; WIN64:        vadd{{.*}}  {{%xmm([0-9]+)}}, %xmm0, %xmm0
; WIN64:        call{{.*}}   {{.*}}test_argRetFloat
; WIN64:        vadd{{.*}}  {{%xmm([0-9]+)}}, %xmm0, %xmm0
; WIN64:        ret{{.*}}

; Test regcall when passing/retrieving float
define x86_regcallcc float @test_CallargRetFloat(float %a)  {
  %b = fadd float 1.0, %a
  %c = call x86_regcallcc float @test_argRetFloat(float %b)
  %d = fadd float 1.0, %c
  ret float %d
}

; X32-LABEL:  test_argRetDouble:
; X32:        vadd{{.*}}  {{.*}}, %xmm0
; X32:        ret{{.*}}

; WIN64-LABEL:  test_argRetDouble:
; WIN64:        vadd{{.*}}  {{.*}}, %xmm0
; WIN64:        ret{{.*}}

; Test regcall when receiving/returning double
define x86_regcallcc double @test_argRetDouble(double %a)  {
  %add = fadd double %a, 1.0
  ret double %add
}

; X32-LABEL:  test_CallargRetDouble:
; X32:        vadd{{.*}}  {{%xmm([0-7])}}, %xmm0, %xmm0
; X32:        call{{.*}}   {{.*}}test_argRetDouble
; X32:        vadd{{.*}}  {{%xmm([0-7])}}, %xmm0, %xmm0
; X32:        ret{{.*}}

; WIN64-LABEL:  test_CallargRetDouble:
; WIN64:        vadd{{.*}}  {{%xmm([0-9]+)}}, %xmm0, %xmm0
; WIN64:        call{{.*}}   {{.*}}test_argRetDouble
; WIN64:        vadd{{.*}}  {{%xmm([0-9]+)}}, %xmm0, %xmm0
; WIN64:        ret{{.*}}

; Test regcall when passing/retrieving double
define x86_regcallcc double @test_CallargRetDouble(double %a)  {
  %b = fadd double 1.0, %a
  %c = call x86_regcallcc double @test_argRetDouble(double %b)
  %d = fadd double 1.0, %c
  ret double %d
}

; X32: test_argRetf80
; X32-NOT: fldt
; X32: fadd	%st(0), %st(0)
; X32: retl

; WIN64: test_argRetf80
; WIN64-NOT: fldt
; WIN64: fadd	%st(0), %st(0)
; WIN64: retq

; Test regcall when receiving/returning long double
define x86_regcallcc x86_fp80 @test_argRetf80(x86_fp80 %a0) nounwind {
  %r0 = fadd x86_fp80 %a0, %a0
  ret x86_fp80 %r0
}

; X32: test_CallargRetf80
; X32-NOT: fldt
; X32: fadd	%st({{[0-7]}}), %st({{[0-7]}})
; X32: call{{.*}}   {{.*}}test_argRetf80
; X32: fadd{{.*}}	%st({{[0-7]}})
; X32: retl

; WIN64: test_CallargRetf80
; WIN64-NOT: fldt
; WIN64: fadd	%st({{[0-7]}}), %st({{[0-7]}})
; WIN64: call{{.*}}   {{.*}}test_argRetf80
; WIN64: fadd{{.*}}	%st({{[0-7]}})
; WIN64: retq

; Test regcall when passing/retrieving long double
define x86_regcallcc x86_fp80 @test_CallargRetf80(x86_fp80 %a)  {
  %b = fadd x86_fp80 %a, %a
  %c = call x86_regcallcc x86_fp80 @test_argRetf80(x86_fp80 %b)
  %d = fadd x86_fp80 %c, %c
  ret x86_fp80 %d
}

; X32-LABEL:  test_argRetPointer:
; X32:        incl %eax
; X32:        ret{{.*}}

; WIN64-LABEL:  test_argRetPointer:
; WIN64:        incl %eax
; WIN64:        ret{{.*}}

; Test regcall when receiving/returning pointer
define x86_regcallcc [4 x i32]* @test_argRetPointer([4 x i32]* %a)  {
  %b = ptrtoint [4 x i32]* %a to i32
  %c = add i32 %b, 1
  %d = inttoptr i32 %c to [4 x i32]*
  ret [4 x i32]* %d
}

; X32-LABEL:  test_CallargRetPointer:
; X32:        incl %eax
; X32:        call{{.*}}   {{.*}}test_argRetPointer
; X32:        incl %eax
; X32:        ret{{.*}}

; WIN64-LABEL:  test_CallargRetPointer:
; WIN64:        incl %eax
; WIN64:        call{{.*}}   {{.*}}test_argRetPointer
; WIN64:        incl %eax
; WIN64:        ret{{.*}}

; Test regcall when passing/retrieving pointer
define x86_regcallcc [4 x i32]* @test_CallargRetPointer([4 x i32]* %a)  {
  %b = ptrtoint [4 x i32]* %a to i32
  %c = add i32 %b, 1
  %d = inttoptr i32 %c to [4 x i32]*
  %e = call x86_regcallcc [4 x i32]* @test_argRetPointer([4 x i32]* %d)
  %f = ptrtoint [4 x i32]* %e to i32
  %g = add i32 %f, 1
  %h = inttoptr i32 %g to [4 x i32]*
  ret [4 x i32]* %h
}

; X32-LABEL:  test_argRet128Vector:
; X32:        vpblend{{.*}}  %xmm0, %xmm1, %xmm0
; X32:        ret{{.*}}

; WIN64-LABEL:  test_argRet128Vector:
; WIN64:        vpblend{{.*}}  %xmm0, %xmm1, %xmm0
; WIN64:        ret{{.*}}

; Test regcall when receiving/returning 128 bit vector
define x86_regcallcc <4 x i32> @test_argRet128Vector(<4 x i32> %a, <4 x i32> %b)  {
  %d = select <4 x i1> undef , <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %d
}

; X32-LABEL:  test_CallargRet128Vector:
; X32:        vmov{{.*}}  %xmm0, {{%xmm([0-7])}}
; X32:        call{{.*}}   {{.*}}test_argRet128Vector
; X32:        vmovdqa{{.*}}  {{%xmm([0-7])}}, %xmm0
; X32:        ret{{.*}}

; WIN64-LABEL:  test_CallargRet128Vector:
; WIN64:        vmov{{.*}}  %xmm0, {{%xmm([0-9]+)}}
; WIN64:        call{{.*}}   {{.*}}test_argRet128Vector
; WIN64:        vmovdqa{{.*}}  {{%xmm([0-9]+)}}, %xmm0
; WIN64:        ret{{.*}}

; Test regcall when passing/retrieving 128 bit vector
define x86_regcallcc <4 x i32> @test_CallargRet128Vector(<4 x i32> %a)  {
  %b = call x86_regcallcc <4 x i32> @test_argRet128Vector(<4 x i32> %a, <4 x i32> %a)
  %c = select <4 x i1> undef , <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %c
}

; X32-LABEL:  test_argRet256Vector:
; X32:        vpblend{{.*}}  %ymm0, %ymm1, %ymm0
; X32:        ret{{.*}}

; WIN64-LABEL:  test_argRet256Vector:
; WIN64:        vpblend{{.*}}  %ymm0, %ymm1, %ymm0
; WIN64:        ret{{.*}}

; Test regcall when receiving/returning 256 bit vector
define x86_regcallcc <8 x i32> @test_argRet256Vector(<8 x i32> %a, <8 x i32> %b)  {
  %d = select <8 x i1> undef , <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %d
}

; X32-LABEL:  test_CallargRet256Vector:
; X32:        vmov{{.*}}  %ymm0, %ymm1
; X32:        call{{.*}}   {{.*}}test_argRet256Vector
; X32:        vmovdqa{{.*}}  %ymm1, %ymm0
; X32:        ret{{.*}}

; WIN64-LABEL:  test_CallargRet256Vector:
; WIN64:        vmov{{.*}}  %ymm0, %ymm1
; WIN64:        call{{.*}}   {{.*}}test_argRet256Vector
; WIN64:        vmovdqa{{.*}}  %ymm1, %ymm0
; WIN64:        ret{{.*}}

; Test regcall when passing/retrieving 256 bit vector
define x86_regcallcc <8 x i32> @test_CallargRet256Vector(<8 x i32> %a)  {
  %b = call x86_regcallcc <8 x i32> @test_argRet256Vector(<8 x i32> %a, <8 x i32> %a)
  %c = select <8 x i1> undef , <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %c
}

; X32-LABEL:  test_argRet512Vector:
; X32:        vpblend{{.*}}  %zmm0, %zmm1, %zmm0
; X32:        ret{{.*}}

; WIN64-LABEL:  test_argRet512Vector:
; WIN64:        vpblend{{.*}}  %zmm0, %zmm1, %zmm0
; WIN64:        ret{{.*}}

; Test regcall when receiving/returning 512 bit vector
define x86_regcallcc <16 x i32> @test_argRet512Vector(<16 x i32> %a, <16 x i32> %b)  {
  %d = select <16 x i1> undef , <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %d
}

; X32-LABEL:  test_CallargRet512Vector:
; X32:        vmov{{.*}}  %zmm0, %zmm1
; X32:        call{{.*}}   {{.*}}test_argRet512Vector
; X32:        movdqa{{.*}}  %zmm1, %zmm0
; X32:        ret{{.*}}

; WIN64-LABEL:  test_CallargRet512Vector:
; WIN64:        vmov{{.*}}  %zmm0, %zmm1
; WIN64:        call{{.*}}   {{.*}}test_argRet512Vector
; WIN64:        vmovdqa{{.*}}  %zmm1, %zmm0
; WIN64:        ret{{.*}}

; Test regcall when passing/retrieving 512 bit vector
define x86_regcallcc <16 x i32> @test_CallargRet512Vector(<16 x i32> %a)  {
  %b = call x86_regcallcc <16 x i32> @test_argRet512Vector(<16 x i32> %a, <16 x i32> %a)
  %c = select <16 x i1> undef , <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %c
}

; WIN64-LABEL: testf32_inp
; WIN64: {{.*}} {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}
; WIN64: {{.*}} {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}
; WIN64: {{.*}} {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}
; WIN64: {{.*}} {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}
; WIN64: {{.*}} {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}
; WIN64: {{.*}} {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}
; WIN64: {{.*}} {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}
; WIN64: {{.*}} {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}
; WIN64: retq

; X32-LABEL: testf32_inp
; X32: vmovups {{%xmm([0-7])}}, {{.*(%esp).*}}  {{#+}} 16-byte Spill
; X32: vmovups {{%xmm([0-7])}}, {{.*(%esp).*}}  {{#+}} 16-byte Spill
; X32: {{.*}} {{%zmm[0-7]}}, {{%zmm[0-7]}}, {{%zmm[0-7]}}
; X32: {{.*}} {{%zmm[0-7]}}, {{%zmm[0-7]}}, {{%zmm[0-7]}}
; X32: {{.*}} {{%zmm[0-7]}}, {{%zmm[0-7]}}, {{%zmm[0-7]}}
; X32: {{.*}} {{%zmm[0-7]}}, {{%zmm[0-7]}}, {{%zmm[0-7]}}
; X32: {{.*}} {{%zmm[0-7]}}, {{%zmm[0-7]}}, {{%zmm[0-7]}}
; X32: {{.*}} {{%zmm[0-7]}}, {{%zmm[0-7]}}, {{%zmm[0-7]}}
; X32: {{.*}} {{%zmm[0-7]}}, {{%zmm[0-7]}}, {{%zmm[0-7]}}
; X32: {{.*}} {{%zmm[0-7]}}, {{%zmm[0-7]}}, {{%zmm[0-7]}}
; X32: vmovups {{.*(%esp).*}}, {{%xmm([0-7])}}  {{#+}} 16-byte Reload
; X32: vmovups {{.*(%esp).*}}, {{%xmm([0-7])}}  {{#+}} 16-byte Reload
; X32: retl

; LINUXOSX64-LABEL: testf32_inp
; LINUXOSX64: {{.*}} {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}
; LINUXOSX64: {{.*}} {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}
; LINUXOSX64: {{.*}} {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}
; LINUXOSX64: {{.*}} {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}
; LINUXOSX64: {{.*}} {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}
; LINUXOSX64: {{.*}} {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}
; LINUXOSX64: {{.*}} {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}
; LINUXOSX64: {{.*}} {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}, {{%zmm([0-9]|1[0-1])}}
; LINUXOSX64: retq

; Test regcall when running multiple input parameters - callee saved XMMs
define x86_regcallcc <32 x float> @testf32_inp(<32 x float> %a, <32 x float> %b, <32 x float> %c) nounwind {
  %x1 = fadd <32 x float> %a, %b
  %x2 = fmul <32 x float> %a, %b
  %x3 = fsub <32 x float> %x1, %x2
  %x4 = fadd <32 x float> %x3, %c
  ret <32 x float> %x4
}

; X32-LABEL: pushl {{%e(si|di|bx|bp)}}
; X32: pushl {{%e(si|di|bx|bp)}}
; X32: pushl {{%e(si|di|bx|bp)}}
; X32: pushl {{%e(si|di|bx|bp)}}
; X32: popl {{%e(si|di|bx|bp)}}
; X32: popl {{%e(si|di|bx|bp)}}
; X32: popl {{%e(si|di|bx|bp)}}
; X32: popl {{%e(si|di|bx|bp)}}
; X32: retl

; WIN64-LABEL: pushq	{{%r(bp|bx|1[0-5])}}
; WIN64: pushq	{{%r(bp|bx|1[0-5])}}
; WIN64: pushq	{{%r(bp|bx|1[0-5])}}
; WIN64: pushq	{{%r(bp|bx|1[0-5])}}
; WIN64: popq	{{%r(bp|bx|1[0-5])}}
; WIN64: popq	{{%r(bp|bx|1[0-5])}}
; WIN64: popq	{{%r(bp|bx|1[0-5])}}
; WIN64: popq	{{%r(bp|bx|1[0-5])}}
; WIN64: retq

; LINUXOSX64-LABEL: pushq	{{%r(bp|bx|1[2-5])}}
; LINUXOSX64: pushq	{{%r(bp|bx|1[2-5])}}
; LINUXOSX64: pushq	{{%r(bp|bx|1[2-5])}}
; LINUXOSX64: popq	{{%r(bp|bx|1[2-5])}}
; LINUXOSX64: popq	{{%r(bp|bx|1[2-5])}}
; LINUXOSX64: popq	{{%r(bp|bx|1[2-5])}}
; LINUXOSX64: retq

; Test regcall when running multiple input parameters - callee saved GPRs
define x86_regcallcc i32 @testi32_inp(i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6,
                                      i32 %b1, i32 %b2, i32 %b3, i32 %b4, i32 %b5, i32 %b6) nounwind {
  %x1 = sub i32 %a1, %a2
  %x2 = sub i32 %a3, %a4
  %x3 = sub i32 %a5, %a6
  %y1 = sub i32 %b1, %b2
  %y2 = sub i32 %b3, %b4
  %y3 = sub i32 %b5, %b6
  %v1 = add i32 %a1, %a2
  %v2 = add i32 %a3, %a4
  %v3 = add i32 %a5, %a6
  %w1 = add i32 %b1, %b2
  %w2 = add i32 %b3, %b4
  %w3 = add i32 %b5, %b6
  %s1 = mul i32 %x1, %y1
  %s2 = mul i32 %x2, %y2
  %s3 = mul i32 %x3, %y3
  %t1 = mul i32 %v1, %w1
  %t2 = mul i32 %v2, %w2
  %t3 = mul i32 %v3, %w3
  %m1 = add i32 %s1, %s2
  %m2 = add i32 %m1, %s3
  %n1 = add i32 %t1, %t2
  %n2 = add i32 %n1, %t3
  %r1 = add i32 %m2, %n2
  ret i32 %r1
}

; X32-LABEL: testf32_stack
; X32: vaddps {{%zmm([0-7])}}, {{%zmm([0-7])}}, {{%zmm([0-7])}}
; X32: vaddps {{%zmm([0-7])}}, {{%zmm([0-7])}}, {{%zmm([0-7])}}
; X32: vaddps {{%zmm([0-7])}}, {{%zmm([0-7])}}, {{%zmm([0-7])}}
; X32: vaddps {{%zmm([0-7])}}, {{%zmm([0-7])}}, {{%zmm([0-7])}}
; X32: vaddps {{%zmm([0-7])}}, {{%zmm([0-7])}}, {{%zmm([0-7])}}
; X32: vaddps {{%zmm([0-7])}}, {{%zmm([0-7])}}, {{%zmm([0-7])}}
; X32: vaddps {{([0-9])+}}(%ebp), {{%zmm([0-7])}}, {{%zmm([0-7])}}
; X32: vaddps {{([0-9])+}}(%ebp), {{%zmm([0-7])}}, {{%zmm([0-7])}}
; X32: vaddps {{([0-9])+}}(%ebp), {{%zmm([0-7])}}, {{%zmm([0-7])}}
; X32: vaddps {{([0-9])+}}(%ebp), {{%zmm([0-7])}}, {{%zmm([0-7])}}
; X32: vaddps {{([0-9])+}}(%ebp), {{%zmm([0-7])}}, {{%zmm([0-7])}}
; X32: vaddps {{([0-9])+}}(%ebp), {{%zmm([0-7])}}, {{%zmm([0-7])}}
; X32: vaddps {{([0-9])+}}(%ebp), {{%zmm([0-7])}}, {{%zmm([0-7])}}
; X32: vaddps {{([0-9])+}}(%ebp), {{%zmm([0-7])}}, {{%zmm([0-7])}}
; X32: vaddps {{([0-9])+}}(%ebp), {{%zmm([0-7])}}, {{%zmm([0-1])}}
; X32: vaddps {{([0-9])+}}(%ebp), {{%zmm([0-7])}}, {{%zmm([0-1])}}
; X32: retl

; LINUXOSX64-LABEL: testf32_stack
; LINUXOSX64: vaddps {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}
; LINUXOSX64: vaddps {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}
; LINUXOSX64: vaddps {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}
; LINUXOSX64: vaddps {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}
; LINUXOSX64: vaddps {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}
; LINUXOSX64: vaddps {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}
; LINUXOSX64: vaddps {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}
; LINUXOSX64: vaddps {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}
; LINUXOSX64: vaddps {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}
; LINUXOSX64: vaddps {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}
; LINUXOSX64: vaddps {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}
; LINUXOSX64: vaddps {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}
; LINUXOSX64: vaddps {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}
; LINUXOSX64: vaddps {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}, {{%zmm([0-9]+)}}
; LINUXOSX64: vaddps {{([0-9])+}}(%rbp), {{%zmm([0-9]+)}}, {{%zmm([0-1])}}
; LINUXOSX64: vaddps {{([0-9])+}}(%rbp), {{%zmm([0-9]+)}}, {{%zmm([0-1])}}
; LINUXOSX64: retq

; Test that parameters, overflowing register capacity, are passed through the stack
define x86_regcallcc <32 x float> @testf32_stack(<32 x float> %a0, <32 x float> %b0, <32 x float> %c0, 
                                               <32 x float> %a1, <32 x float> %b1, <32 x float> %c1,
                                               <32 x float> %a2, <32 x float> %b2, <32 x float> %c2) nounwind {
  %x1 = fadd <32 x float> %a0, %b0
  %x2 = fadd <32 x float> %c0, %x1
  %x3 = fadd <32 x float> %a1, %x2
  %x4 = fadd <32 x float> %b1, %x3
  %x5 = fadd <32 x float> %c1, %x4
  %x6 = fadd <32 x float> %a2, %x5
  %x7 = fadd <32 x float> %b2, %x6
  %x8 = fadd <32 x float> %c2, %x7
  ret <32 x float> %x8
}

; X32-LABEL: vmovd   %edx, {{%xmm([0-9])}}
; X32:       vcvtsi2sdl      %eax, {{%xmm([0-9])}}, {{%xmm([0-9])}}
; X32:       vcvtsi2sdl      %ecx, {{%xmm([0-9])}}, {{%xmm([0-9])}}
; X32:       vcvtsi2sdl      %esi, {{%xmm([0-9])}}, {{%xmm([0-9])}}
; X32:       vaddsd  %xmm1, %xmm0, %xmm0
; X32:       vcvttsd2si      %xmm0, %eax
; X32:       retl

; LINUXOSX64-LABEL: test_argRetMixTypes
; LINUXOSX64:       vcvtss2sd       %xmm1, %xmm1, %xmm1
; LINUXOSX64:       vcvtsi2sdl      %eax, {{%xmm([0-9])}}, {{%xmm([0-9])}}
; LINUXOSX64:       vcvtsi2sdl      %ecx, {{%xmm([0-9])}}, {{%xmm([0-9])}}
; LINUXOSX64:       vcvtsi2sdq      %rdx, {{%xmm([0-9])}}, {{%xmm([0-9])}}
; LINUXOSX64:       vcvtsi2sdl      %edi, {{%xmm([0-9])}}, {{%xmm([0-9])}}
; LINUXOSX64:       vcvtsi2sdl      (%rsi), {{%xmm([0-9])}}, {{%xmm([0-9])}}
; LINUXOSX64:       vcvttsd2si      {{%xmm([0-9])}}, %eax

; Test regcall when passing/retrieving mixed types
define x86_regcallcc i32 @test_argRetMixTypes(double, float, i8 signext, i32, i64, i16 signext, i32*) #0 {
  %8 = fpext float %1 to double
  %9 = fadd double %8, %0
  %10 = sitofp i8 %2 to double
  %11 = fadd double %9, %10
  %12 = sitofp i32 %3 to double
  %13 = fadd double %11, %12
  %14 = sitofp i64 %4 to double
  %15 = fadd double %13, %14
  %16 = sitofp i16 %5 to double
  %17 = fadd double %15, %16
  %18 = load i32, i32* %6, align 4
  %19 = sitofp i32 %18 to double
  %20 = fadd double %17, %19
  %21 = fptosi double %20 to i32
  ret i32 %21
}

%struct.complex = type { float, double, i32, i8, i64}


; X32-LABEL: test_argMultiRet    
; X32:       vaddsd {{.*}}, %xmm1, %xmm1
; X32:       movl    $4, %eax
; X32:       movb    $7, %cl
; X32:       movl    $999, %edx
; X32:       xorl    %edi, %edi
; X32:       retl

; LINUXOSX64-LABEL: test_argMultiRet 
; LINUXOSX64:       vaddsd  {{.*}}, %xmm1, %xmm1
; LINUXOSX64:       movl    $4, %eax
; LINUXOSX64:       movb    $7, %cl
; LINUXOSX64:       movl    $999, %edx
; LINUXOSX64:       retq
        
define x86_regcallcc %struct.complex @test_argMultiRet(float, double, i32, i8, i64) local_unnamed_addr #0 {
  %6 = fadd double %1, 5.000000e+00
  %7 = insertvalue %struct.complex undef, float %0, 0
  %8 = insertvalue %struct.complex %7, double %6, 1
  %9 = insertvalue %struct.complex %8, i32 4, 2
  %10 = insertvalue %struct.complex %9, i8 7, 3
  %11 = insertvalue %struct.complex %10, i64 999, 4
  ret %struct.complex %11
}


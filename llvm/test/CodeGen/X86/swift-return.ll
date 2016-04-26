; RUN: llc -verify-machineinstrs < %s -mtriple=x86_64-unknown-unknown | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=x86_64-unknown-unknown -O0 | FileCheck --check-prefix=CHECK-O0 %s

@var = global i32 0

; Test how llvm handles return type of {i16, i8}. The return value will be
; passed in %eax and %dl.
; CHECK-LABEL: test:
; CHECK: movl %edi
; CHECK: callq gen
; CHECK: movsbl %dl
; CHECK: addl %{{.*}}, %eax
; CHECK-O0-LABEL: test
; CHECK-O0: movl %edi
; CHECK-O0: callq gen
; CHECK-O0: movswl %ax
; CHECK-O0: movsbl %dl
; CHECK-O0: addl
; CHECK-O0: movw %{{.*}}, %ax
define i16 @test(i32 %key) {
entry:
  %key.addr = alloca i32, align 4
  store i32 %key, i32* %key.addr, align 4
  %0 = load i32, i32* %key.addr, align 4
  %call = call swiftcc { i16, i8 } @gen(i32 %0)
  %v3 = extractvalue { i16, i8 } %call, 0
  %v1 = sext i16 %v3 to i32
  %v5 = extractvalue { i16, i8 } %call, 1
  %v2 = sext i8 %v5 to i32
  %add = add nsw i32 %v1, %v2
  %conv = trunc i32 %add to i16
  ret i16 %conv
}

declare swiftcc { i16, i8 } @gen(i32)

; If we can't pass every return value in register, we will pass everything
; in memroy. The caller provides space for the return value and passes
; the address in %rax. The first input argument will be in %rdi.
; CHECK-LABEL: test2:
; CHECK: leaq (%rsp), %rax
; CHECK: callq gen2
; CHECK: movl (%rsp)
; CHECK-DAG: addl 4(%rsp)
; CHECK-DAG: addl 8(%rsp)
; CHECK-DAG: addl 12(%rsp)
; CHECK-DAG: addl 16(%rsp)
; CHECK-O0-LABEL: test2:
; CHECK-O0-DAG: leaq (%rsp), %rax
; CHECK-O0: callq gen2
; CHECK-O0-DAG: movl (%rsp)
; CHECK-O0-DAG: movl 4(%rsp)
; CHECK-O0-DAG: movl 8(%rsp)
; CHECK-O0-DAG: movl 12(%rsp)
; CHECK-O0-DAG: movl 16(%rsp)
; CHECK-O0: addl
; CHECK-O0: addl
; CHECK-O0: addl
; CHECK-O0: addl
; CHECK-O0: movl %{{.*}}, %eax
define i32 @test2(i32 %key) #0 {
entry:
  %key.addr = alloca i32, align 4
  store i32 %key, i32* %key.addr, align 4
  %0 = load i32, i32* %key.addr, align 4
  %call = call swiftcc { i32, i32, i32, i32, i32 } @gen2(i32 %0)

  %v3 = extractvalue { i32, i32, i32, i32, i32 } %call, 0
  %v5 = extractvalue { i32, i32, i32, i32, i32 } %call, 1
  %v6 = extractvalue { i32, i32, i32, i32, i32 } %call, 2
  %v7 = extractvalue { i32, i32, i32, i32, i32 } %call, 3
  %v8 = extractvalue { i32, i32, i32, i32, i32 } %call, 4

  %add = add nsw i32 %v3, %v5
  %add1 = add nsw i32 %add, %v6
  %add2 = add nsw i32 %add1, %v7
  %add3 = add nsw i32 %add2, %v8
  ret i32 %add3
}

; The address of the return value is passed in %rax.
; On return, we don't keep the address in %rax.
; CHECK-LABEL: gen2:
; CHECK: movl %edi, 16(%rax)
; CHECK: movl %edi, 12(%rax)
; CHECK: movl %edi, 8(%rax)
; CHECK: movl %edi, 4(%rax)
; CHECK: movl %edi, (%rax)
; CHECK-O0-LABEL: gen2:
; CHECK-O0-DAG: movl %edi, 16(%rax)
; CHECK-O0-DAG: movl %edi, 12(%rax)
; CHECK-O0-DAG: movl %edi, 8(%rax)
; CHECK-O0-DAG: movl %edi, 4(%rax)
; CHECK-O0-DAG: movl %edi, (%rax)
define swiftcc { i32, i32, i32, i32, i32 } @gen2(i32 %key) {
  %Y = insertvalue { i32, i32, i32, i32, i32 } undef, i32 %key, 0
  %Z = insertvalue { i32, i32, i32, i32, i32 } %Y, i32 %key, 1
  %Z2 = insertvalue { i32, i32, i32, i32, i32 } %Z, i32 %key, 2
  %Z3 = insertvalue { i32, i32, i32, i32, i32 } %Z2, i32 %key, 3
  %Z4 = insertvalue { i32, i32, i32, i32, i32 } %Z3, i32 %key, 4
  ret { i32, i32, i32, i32, i32 } %Z4
}

; The return value {i32, i32, i32, i32} will be returned via registers %eax,
; %edx, %ecx, %r8d.
; CHECK-LABEL: test3:
; CHECK: callq gen3
; CHECK: addl %edx, %eax
; CHECK: addl %ecx, %eax
; CHECK: addl %r8d, %eax
; CHECK-O0-LABEL: test3:
; CHECK-O0: callq gen3
; CHECK-O0: addl %edx, %eax
; CHECK-O0: addl %ecx, %eax
; CHECK-O0: addl %r8d, %eax
define i32 @test3(i32 %key) #0 {
entry:
  %key.addr = alloca i32, align 4
  store i32 %key, i32* %key.addr, align 4
  %0 = load i32, i32* %key.addr, align 4
  %call = call swiftcc { i32, i32, i32, i32 } @gen3(i32 %0)

  %v3 = extractvalue { i32, i32, i32, i32 } %call, 0
  %v5 = extractvalue { i32, i32, i32, i32 } %call, 1
  %v6 = extractvalue { i32, i32, i32, i32 } %call, 2
  %v7 = extractvalue { i32, i32, i32, i32 } %call, 3

  %add = add nsw i32 %v3, %v5
  %add1 = add nsw i32 %add, %v6
  %add2 = add nsw i32 %add1, %v7
  ret i32 %add2
}

declare swiftcc { i32, i32, i32, i32 } @gen3(i32 %key)

; The return value {float, float, float, float} will be returned via registers
; %xmm0, %xmm1, %xmm2, %xmm3.
; CHECK-LABEL: test4:
; CHECK: callq gen4
; CHECK: addss %xmm1, %xmm0
; CHECK: addss %xmm2, %xmm0
; CHECK: addss %xmm3, %xmm0
; CHECK-O0-LABEL: test4:
; CHECK-O0: callq gen4
; CHECK-O0: addss %xmm1, %xmm0
; CHECK-O0: addss %xmm2, %xmm0
; CHECK-O0: addss %xmm3, %xmm0
define float @test4(float %key) #0 {
entry:
  %key.addr = alloca float, align 4
  store float %key, float* %key.addr, align 4
  %0 = load float, float* %key.addr, align 4
  %call = call swiftcc { float, float, float, float } @gen4(float %0)

  %v3 = extractvalue { float, float, float, float } %call, 0
  %v5 = extractvalue { float, float, float, float } %call, 1
  %v6 = extractvalue { float, float, float, float } %call, 2
  %v7 = extractvalue { float, float, float, float } %call, 3

  %add = fadd float %v3, %v5
  %add1 = fadd float %add, %v6
  %add2 = fadd float %add1, %v7
  ret float %add2
}

declare swiftcc { float, float, float, float } @gen4(float %key)

; CHECK-LABEL: consume_i1_ret:
; CHECK: callq produce_i1_ret
; CHECK: andb $1, %al
; CHECK: andb $1, %dl
; CHECK: andb $1, %cl
; CHECK: andb $1, %r8b
; CHECK-O0-LABEL: consume_i1_ret:
; CHECK-O0: callq produce_i1_ret
; CHECK-O0: andb $1, %al
; CHECK-O0: andb $1, %dl
; CHECK-O0: andb $1, %cl
; CHECK-O0: andb $1, %r8b
define void @consume_i1_ret() {
  %call = call swiftcc { i1, i1, i1, i1 } @produce_i1_ret()
  %v3 = extractvalue { i1, i1, i1, i1 } %call, 0
  %v5 = extractvalue { i1, i1, i1, i1 } %call, 1
  %v6 = extractvalue { i1, i1, i1, i1 } %call, 2
  %v7 = extractvalue { i1, i1, i1, i1 } %call, 3
  %val = zext i1 %v3 to i32
  store i32 %val, i32* @var
  %val2 = zext i1 %v5 to i32
  store i32 %val2, i32* @var
  %val3 = zext i1 %v6 to i32
  store i32 %val3, i32* @var
  %val4 = zext i1 %v7 to i32
  store i32 %val4, i32* @var
  ret void
}

declare swiftcc { i1, i1, i1, i1 } @produce_i1_ret()

; CHECK-LABEL: foo:
; CHECK: movq %rdi, (%rax)
; CHECK-O0-LABEL: foo:
; CHECK-O0: movq %rdi, (%rax)
define swiftcc void @foo(i64* sret %agg.result, i64 %val) {
  store i64 %val, i64* %agg.result
  ret void
}

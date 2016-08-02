; RUN: llc -verify-machineinstrs < %s -mtriple=x86_64-unknown-unknown | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=x86_64-unknown-unknown -O0 | FileCheck --check-prefix=CHECK-O0 %s

; Test how llvm handles return type of {i16, i8}. The return value will be passed in %eax and %dl.
; clang actually returns i32 instead of {i16, i8}.
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

; We can't pass every return value in register, instead, pass everything in memroy.
; The caller provides space for the return value and passes the address in %rdi.
; The first input argument will be in %rsi.
; CHECK-LABEL: test2:
; CHECK: leaq (%rsp), %rdi
; CHECK: movl %{{.*}}, %esi
; CHECK: callq gen2
; CHECK: movl (%rsp)
; CHECK-DAG: addl 4(%rsp)
; CHECK-DAG: addl 8(%rsp)
; CHECK-DAG: addl 12(%rsp)
; CHECK-DAG: addl 16(%rsp)
; CHECK-O0-LABEL: test2:
; CHECK-O0-DAG: leaq (%rsp), %rdi
; CHECK-O0-DAG: movl {{.*}}, %esi
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

; The address of the return value is passed in %rdi.
; On return, %rax will contain the adddress that has been passed in by the caller in %rdi.
; CHECK-LABEL: gen2:
; CHECK: movl %esi, 16(%rdi)
; CHECK: movl %esi, 12(%rdi)
; CHECK: movl %esi, 8(%rdi)
; CHECK: movl %esi, 4(%rdi)
; CHECK: movl %esi, (%rdi)
; CHECK: movq %rdi, %rax
; CHECK-O0-LABEL: gen2:
; CHECK-O0-DAG: movl %esi, 16(%rdi)
; CHECK-O0-DAG: movl %esi, 12(%rdi)
; CHECK-O0-DAG: movl %esi, 8(%rdi)
; CHECK-O0-DAG: movl %esi, 4(%rdi)
; CHECK-O0-DAG: movl %esi, (%rdi)
; CHECK-O0-DAG: movq %rdi, %rax
define swiftcc { i32, i32, i32, i32, i32 } @gen2(i32 %key) {
  %Y = insertvalue { i32, i32, i32, i32, i32 } undef, i32 %key, 0
  %Z = insertvalue { i32, i32, i32, i32, i32 } %Y, i32 %key, 1
  %Z2 = insertvalue { i32, i32, i32, i32, i32 } %Z, i32 %key, 2
  %Z3 = insertvalue { i32, i32, i32, i32, i32 } %Z2, i32 %key, 3
  %Z4 = insertvalue { i32, i32, i32, i32, i32 } %Z3, i32 %key, 4
  ret { i32, i32, i32, i32, i32 } %Z4
}

; The return value {i32, i32, i32, i32} will be returned via registers %eax, %edx, %ecx, %r8d.
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

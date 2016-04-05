; RUN: llc -mtriple=armv7k-apple-ios8.0 -mcpu=cortex-a7 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=armv7k-apple-ios8.0 -mcpu=cortex-a7 -verify-machineinstrs < %s -O0 | FileCheck --check-prefix=CHECK-O0 %s

; RUN: llc -mtriple=armv7-apple-ios -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=armv7-apple-ios -verify-machineinstrs < %s -O0 | FileCheck --check-prefix=CHECK-O0 %s

; Test how llvm handles return type of {i16, i8}. The return value will be
; passed in %r0 and %r1.
; CHECK-LABEL: test:
; CHECK: bl {{.*}}gen
; CHECK: sxth {{.*}}, r0
; CHECK: sxtab r0, {{.*}}, r1
; CHECK-O0-LABEL: test:
; CHECK-O0: bl {{.*}}gen
; CHECK-O0: sxth r0, r0
; CHECK-O0: sxtb r1, r1
; CHECK-O0: add r0, r0, r1
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

; We can't pass every return value in register, instead, pass everything in
; memroy.
; The caller provides space for the return value and passes the address in %r0.
; The first input argument will be in %r1.
; CHECK-LABEL: test2:
; CHECK: mov r1, r0
; CHECK: mov r0, sp
; CHECK: bl {{.*}}gen2
; CHECK-DAG: add
; CHECK-DAG: ldr {{.*}}, [sp, #16]
; CHECK-DAG: add
; CHECK-DAG: add
; CHECK-DAG: add
; CHECK-O0-LABEL: test2:
; CHECK-O0: str r0
; CHECK-O0: mov r0, sp
; CHECK-O0: bl {{.*}}gen2
; CHECK-O0-DAG: ldr {{.*}}, [sp]
; CHECK-O0-DAG: ldr {{.*}}, [sp, #4]
; CHECK-O0-DAG: ldr {{.*}}, [sp, #8]
; CHECK-O0-DAG: ldr {{.*}}, [sp, #12]
; CHECK-O0-DAG: ldr {{.*}}, [sp, #16]
; CHECK-O0-DAG: add
; CHECK-O0-DAG: add
; CHECK-O0-DAG: add
; CHECK-O0-DAG: add
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

; The address of the return value is passed in %r0.
; CHECK-LABEL: gen2:
; CHECK-DAG: str r1, [r0]
; CHECK-DAG: str r1, [r0, #4]
; CHECK-DAG: str r1, [r0, #8]
; CHECK-DAG: str r1, [r0, #12]
; CHECK-DAG: str r1, [r0, #16]
; CHECK-O0-LABEL: gen2:
; CHECK-O0-DAG: str r1, [r0]
; CHECK-O0-DAG: str r1, [r0, #4]
; CHECK-O0-DAG: str r1, [r0, #8]
; CHECK-O0-DAG: str r1, [r0, #12]
; CHECK-O0-DAG: str r1, [r0, #16]
define swiftcc { i32, i32, i32, i32, i32 } @gen2(i32 %key) {
  %Y = insertvalue { i32, i32, i32, i32, i32 } undef, i32 %key, 0
  %Z = insertvalue { i32, i32, i32, i32, i32 } %Y, i32 %key, 1
  %Z2 = insertvalue { i32, i32, i32, i32, i32 } %Z, i32 %key, 2
  %Z3 = insertvalue { i32, i32, i32, i32, i32 } %Z2, i32 %key, 3
  %Z4 = insertvalue { i32, i32, i32, i32, i32 } %Z3, i32 %key, 4
  ret { i32, i32, i32, i32, i32 } %Z4
}

; The return value {i32, i32, i32, i32} will be returned via registers %r0, %r1,
; %r2, %r3.
; CHECK-LABEL: test3:
; CHECK: bl {{.*}}gen3
; CHECK: add r0, r0, r1
; CHECK: add r0, r0, r2
; CHECK: add r0, r0, r3
; CHECK-O0-LABEL: test3:
; CHECK-O0: bl {{.*}}gen3
; CHECK-O0: add r0, r0, r1
; CHECK-O0: add r0, r0, r2
; CHECK-O0: add r0, r0, r3
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

; RUN: llc < %s -mtriple=s390x-linux-gnu -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -O0 -verify-machineinstrs | FileCheck --check-prefix=CHECK-O0 %s

@var = global i32 0

; Test how llvm handles return type of {i16, i8}. The return value will be
; passed in %r2 and %r3.
; CHECK-LABEL: test:
; CHECK: st %r2
; CHECK: brasl %r14, gen
; CHECK-DAG: lhr %{{r[0,2]+}}, %r2
; CHECK-DAG: lbr %{{r[0,2]+}}, %r3
; CHECK: ar %r2, %r0
; CHECK-O0-LABEL: test
; CHECK-O0: st %r2
; CHECK-O0: brasl %r14, gen
; CHECK-O0-DAG: lhr %r2, %r2
; CHECK-O0-DAG: lbr %[[REG2:r[0-9]+]], %r3
; CHECK-O0: ar %r2, %[[REG2]]
define i16 @test(i32 %key) {
entry:
  %key.addr = alloca i32, align 4
  store i32 %key, ptr %key.addr, align 4
  %0 = load i32, ptr %key.addr, align 4
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

; If we can't pass every return value in registers, we will pass everything
; in memroy. The caller provides space for the return value and passes
; the address in %r2. The first input argument will be in %r3.
; CHECK-LABEL: test2:
; CHECK: lr %r3, %r2
; CHECK-DAG: la %r2, 160(%r15)
; CHECK: brasl %r14, gen2
; CHECK: l %r2, 160(%r15)
; CHECK: a %r2, 164(%r15)
; CHECK: a %r2, 168(%r15)
; CHECK: a %r2, 172(%r15)
; CHECK: a %r2, 176(%r15)
; CHECK-O0-LABEL: test2:
; CHECK-O0: st %r2, [[SPILL1:[0-9]+]](%r15)
; CHECK-O0: l %r3, [[SPILL1]](%r15)
; CHECK-O0: la %r2, 160(%r15)
; CHECK-O0: brasl %r14, gen2
; CHECK-O0-DAG: l %r{{.*}}, 176(%r15)
; CHECK-O0-DAG: l %r{{.*}}, 172(%r15)
; CHECK-O0-DAG: l %r{{.*}}, 168(%r15)
; CHECK-O0-DAG: l %r{{.*}}, 164(%r15)
; CHECK-O0-DAG: l %r{{.*}}, 160(%r15)
; CHECK-O0: ar
; CHECK-O0: ar
; CHECK-O0: ar
; CHECK-O0: ar
define i32 @test2(i32 %key) #0 {
entry:
  %key.addr = alloca i32, align 4
  store i32 %key, ptr %key.addr, align 4
  %0 = load i32, ptr %key.addr, align 4
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

; The address of the return value is passed in %r2.
; On return, %r2 will contain the adddress that has been passed in by the caller in %r2.
; CHECK-LABEL: gen2:
; CHECK: st %r3, 16(%r2)
; CHECK: st %r3, 12(%r2)
; CHECK: st %r3, 8(%r2)
; CHECK: st %r3, 4(%r2)
; CHECK: st %r3, 0(%r2)
; CHECK-O0-LABEL: gen2:
; CHECK-O0-DAG: st %r3, 16(%r2)
; CHECK-O0-DAG: st %r3, 12(%r2)
; CHECK-O0-DAG: st %r3, 8(%r2)
; CHECK-O0-DAG: st %r3, 4(%r2)
; CHECK-O0-DAG: st %r3, 0(%r2)
define swiftcc { i32, i32, i32, i32, i32 } @gen2(i32 %key) {
  %Y = insertvalue { i32, i32, i32, i32, i32 } undef, i32 %key, 0
  %Z = insertvalue { i32, i32, i32, i32, i32 } %Y, i32 %key, 1
  %Z2 = insertvalue { i32, i32, i32, i32, i32 } %Z, i32 %key, 2
  %Z3 = insertvalue { i32, i32, i32, i32, i32 } %Z2, i32 %key, 3
  %Z4 = insertvalue { i32, i32, i32, i32, i32 } %Z3, i32 %key, 4
  ret { i32, i32, i32, i32, i32 } %Z4
}

; The return value {i32, i32, i32, i32} will be returned via registers
; %r2, %r3, %r4, %r5.
; CHECK-LABEL: test3:
; CHECK: brasl %r14, gen3
; CHECK: ar %r2, %r3
; CHECK: ar %r2, %r4
; CHECK: ar %r2, %r5
; CHECK-O0-LABEL: test3:
; CHECK-O0: brasl %r14, gen3
; CHECK-O0: ar %r2, %r3
; CHECK-O0: ar %r2, %r4
; CHECK-O0: ar %r2, %r5
define i32 @test3(i32 %key) #0 {
entry:
  %key.addr = alloca i32, align 4
  store i32 %key, ptr %key.addr, align 4
  %0 = load i32, ptr %key.addr, align 4
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
; %f0, %f2, %f4, %f6.
; CHECK-LABEL: test4:
; CHECK: brasl %r14, gen4
; CHECK: aebr %f0, %f2
; CHECK: aebr %f0, %f4
; CHECK: aebr %f0, %f6
; CHECK-O0-LABEL: test4:
; CHECK-O0: brasl %r14, gen4
; CHECK-O0: aebr %f0, %f2
; CHECK-O0: aebr %f0, %f4
; CHECK-O0: aebr %f0, %f6
define float @test4(float %key) #0 {
entry:
  %key.addr = alloca float, align 4
  store float %key, ptr %key.addr, align 4
  %0 = load float, ptr %key.addr, align 4
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
; CHECK: brasl %r14, produce_i1_ret
; CHECK: nilf %r2, 1
; CHECK: nilf %r3, 1
; CHECK: nilf %r4, 1
; CHECK: nilf %r5, 1
; CHECK-O0-LABEL: consume_i1_ret:
; CHECK-O0: brasl %r14, produce_i1_ret
; CHECK-O0: nilf %r2, 1
; CHECK-O0: nilf %r3, 1
; CHECK-O0: nilf %r4, 1
; CHECK-O0: nilf %r5, 1
define void @consume_i1_ret() {
  %call = call swiftcc { i1, i1, i1, i1 } @produce_i1_ret()
  %v3 = extractvalue { i1, i1, i1, i1 } %call, 0
  %v5 = extractvalue { i1, i1, i1, i1 } %call, 1
  %v6 = extractvalue { i1, i1, i1, i1 } %call, 2
  %v7 = extractvalue { i1, i1, i1, i1 } %call, 3
  %val = zext i1 %v3 to i32
  store volatile i32 %val, ptr @var
  %val2 = zext i1 %v5 to i32
  store volatile i32 %val2, ptr @var
  %val3 = zext i1 %v6 to i32
  store volatile i32 %val3, ptr @var
  %val4 = zext i1 %v7 to i32
  store i32 %val4, ptr @var
  ret void
}

declare swiftcc { i1, i1, i1, i1 } @produce_i1_ret()

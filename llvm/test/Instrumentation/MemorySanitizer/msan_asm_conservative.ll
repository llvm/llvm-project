; Test for handling of asm constraints in MSan instrumentation.
; RUN: opt < %s -msan-check-access-address=0 -msan-handle-asm-conservative=0 -S -passes=msan 2>&1 | FileCheck %s
; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck --check-prefixes=CHECK,USER-CONS %s
; RUN: opt < %s -msan-kernel=1 -msan-check-access-address=0 -msan-handle-asm-conservative=0 -S -passes=msan 2>&1 | FileCheck --check-prefixes=CHECK,KMSAN %s
; RUN: opt < %s -msan-kernel=1 -msan-check-access-address=0 -msan-handle-asm-conservative=1 -S -passes=msan 2>&1 | FileCheck --check-prefixes=CHECK,KMSAN,CHECK-CONS %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.pair = type { i32, i32 }
%struct.large = type { [33 x i8] }

@id1 = common dso_local global i32 0, align 4
@is1 = common dso_local global i32 0, align 4
@id2 = common dso_local global i32 0, align 4
@is2 = common dso_local global i32 0, align 4
@id3 = common dso_local global i32 0, align 4
@pair2 = common dso_local global %struct.pair zeroinitializer, align 4
@pair1 = common dso_local global %struct.pair zeroinitializer, align 4
@c2 = common dso_local global i8 0, align 1
@c1 = common dso_local global i8 0, align 1
@memcpy_d1 = common dso_local global ptr null, align 8
@memcpy_d2 = common dso_local global ptr null, align 8
@memcpy_s1 = common dso_local global ptr null, align 8
@memcpy_s2 = common dso_local global ptr null, align 8
@large = common dso_local global %struct.pair zeroinitializer, align 4

; The functions below were generated from a C source that contains declarations like follows:
;   void f1() {
;     asm("" : "=r" (id1) : "r" (is1));
;   }
; with corresponding input/output constraints.
; Note that the assembly statement is always empty, as MSan doesn't look at it anyway.

; One input register, one output register:
;   asm("" : "=r" (id1) : "r" (is1));
define dso_local void @f_1i_1o_reg() sanitize_memory {
entry:
  %0 = load i32, ptr @is1, align 4
  %1 = call i32 asm "", "=r,r,~{dirflag},~{fpsr},~{flags}"(i32 %0)
  store i32 %1, ptr @id1, align 4
  ret void
}

; CHECK-LABEL: @f_1i_1o_reg
; CHECK: [[IS1_F1:%.*]] = load i32, ptr @is1, align 4
; CHECK: call void @__msan_warning
; CHECK: call i32 asm "",{{.*}}(i32 [[IS1_F1]])
; KMSAN: [[PACK1_F1:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id1{{.*}})
; KMSAN: [[EXT1_F1:%.*]] = extractvalue { ptr, ptr } [[PACK1_F1]], 0
; KMSAN: store i32 0, ptr [[EXT1_F1]]


; Two input registers, two output registers:
;   asm("" : "=r" (id1), "=r" (id2) : "r" (is1), "r"(is2));
define dso_local void @f_2i_2o_reg() sanitize_memory {
entry:
  %0 = load i32, ptr @is1, align 4
  %1 = load i32, ptr @is2, align 4
  %2 = call { i32, i32 } asm "", "=r,=r,r,r,~{dirflag},~{fpsr},~{flags}"(i32 %0, i32 %1)
  %asmresult = extractvalue { i32, i32 } %2, 0
  %asmresult1 = extractvalue { i32, i32 } %2, 1
  store i32 %asmresult, ptr @id1, align 4
  store i32 %asmresult1, ptr @id2, align 4
  ret void
}

; CHECK-LABEL: @f_2i_2o_reg
; CHECK: [[IS1_F2:%.*]] = load i32, ptr @is1, align 4
; CHECK: [[IS2_F2:%.*]] = load i32, ptr @is2, align 4
; CHECK: call void @__msan_warning
; KMSAN: call void @__msan_warning
; CHECK: call { i32, i32 } asm "",{{.*}}(i32 [[IS1_F2]], i32 [[IS2_F2]])
; KMSAN: [[PACK1_F2:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id1{{.*}})
; KMSAN: [[EXT1_F2:%.*]] = extractvalue { ptr, ptr } [[PACK1_F2]], 0
; KMSAN: store i32 0, ptr [[EXT1_F2]]
; KMSAN: [[PACK2_F2:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id2{{.*}})
; KMSAN: [[EXT2_F2:%.*]] = extractvalue { ptr, ptr } [[PACK2_F2]], 0
; KMSAN: store i32 0, ptr [[EXT2_F2]]

; Input same as output, used twice:
;   asm("" : "=r" (id1), "=r" (id2) : "r" (id1), "r" (id2));
define dso_local void @f_2i_2o_reuse2_reg() sanitize_memory {
entry:
  %0 = load i32, ptr @id1, align 4
  %1 = load i32, ptr @id2, align 4
  %2 = call { i32, i32 } asm "", "=r,=r,r,r,~{dirflag},~{fpsr},~{flags}"(i32 %0, i32 %1)
  %asmresult = extractvalue { i32, i32 } %2, 0
  %asmresult1 = extractvalue { i32, i32 } %2, 1
  store i32 %asmresult, ptr @id1, align 4
  store i32 %asmresult1, ptr @id2, align 4
  ret void
}

; CHECK-LABEL: @f_2i_2o_reuse2_reg
; CHECK: [[ID1_F3:%.*]] = load i32, ptr @id1, align 4
; CHECK: [[ID2_F3:%.*]] = load i32, ptr @id2, align 4
; CHECK: call void @__msan_warning
; KMSAN: call void @__msan_warning
; CHECK: call { i32, i32 } asm "",{{.*}}(i32 [[ID1_F3]], i32 [[ID2_F3]])
; KMSAN: [[PACK1_F3:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id1{{.*}})
; KMSAN: [[EXT1_F3:%.*]] = extractvalue { ptr, ptr } [[PACK1_F3]], 0
; KMSAN: store i32 0, ptr [[EXT1_F3]]
; KMSAN: [[PACK2_F3:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id2{{.*}})
; KMSAN: [[EXT2_F3:%.*]] = extractvalue { ptr, ptr } [[PACK2_F3]], 0
; KMSAN: store i32 0, ptr [[EXT2_F3]]


; One of the input registers is also an output:
;   asm("" : "=r" (id1), "=r" (id2) : "r" (id1), "r"(is1));
define dso_local void @f_2i_2o_reuse1_reg() sanitize_memory {
entry:
  %0 = load i32, ptr @id1, align 4
  %1 = load i32, ptr @is1, align 4
  %2 = call { i32, i32 } asm "", "=r,=r,r,r,~{dirflag},~{fpsr},~{flags}"(i32 %0, i32 %1)
  %asmresult = extractvalue { i32, i32 } %2, 0
  %asmresult1 = extractvalue { i32, i32 } %2, 1
  store i32 %asmresult, ptr @id1, align 4
  store i32 %asmresult1, ptr @id2, align 4
  ret void
}

; CHECK-LABEL: @f_2i_2o_reuse1_reg
; CHECK: [[ID1_F4:%.*]] = load i32, ptr @id1, align 4
; CHECK: [[IS1_F4:%.*]] = load i32, ptr @is1, align 4
; CHECK: call void @__msan_warning
; KMSAN: call void @__msan_warning
; CHECK: call { i32, i32 } asm "",{{.*}}(i32 [[ID1_F4]], i32 [[IS1_F4]])
; KMSAN: [[PACK1_F4:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id1{{.*}})
; KMSAN: [[EXT1_F4:%.*]] = extractvalue { ptr, ptr } [[PACK1_F4]], 0
; KMSAN: store i32 0, ptr [[EXT1_F4]]
; KMSAN: [[PACK2_F4:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id2{{.*}})
; KMSAN: [[EXT2_F4:%.*]] = extractvalue { ptr, ptr } [[PACK2_F4]], 0
; KMSAN: store i32 0, ptr [[EXT2_F4]]


; One input register, three output registers:
;   asm("" : "=r" (id1), "=r" (id2), "=r" (id3) : "r" (is1));
define dso_local void @f_1i_3o_reg() sanitize_memory {
entry:
  %0 = load i32, ptr @is1, align 4
  %1 = call { i32, i32, i32 } asm "", "=r,=r,=r,r,~{dirflag},~{fpsr},~{flags}"(i32 %0)
  %asmresult = extractvalue { i32, i32, i32 } %1, 0
  %asmresult1 = extractvalue { i32, i32, i32 } %1, 1
  %asmresult2 = extractvalue { i32, i32, i32 } %1, 2
  store i32 %asmresult, ptr @id1, align 4
  store i32 %asmresult1, ptr @id2, align 4
  store i32 %asmresult2, ptr @id3, align 4
  ret void
}

; CHECK-LABEL: @f_1i_3o_reg
; CHECK: [[IS1_F5:%.*]] = load i32, ptr @is1, align 4
; CHECK: call void @__msan_warning
; CHECK: call { i32, i32, i32 } asm "",{{.*}}(i32 [[IS1_F5]])
; KMSAN: [[PACK1_F5:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id1{{.*}})
; KMSAN: [[EXT1_F5:%.*]] = extractvalue { ptr, ptr } [[PACK1_F5]], 0
; KMSAN: store i32 0, ptr [[EXT1_F5]]
; KMSAN: [[PACK2_F5:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id2{{.*}})
; KMSAN: [[EXT2_F5:%.*]] = extractvalue { ptr, ptr } [[PACK2_F5]], 0
; KMSAN: store i32 0, ptr [[EXT2_F5]]
; KMSAN: [[PACK3_F5:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id3{{.*}})
; KMSAN: [[EXT3_F5:%.*]] = extractvalue { ptr, ptr } [[PACK3_F5]], 0
; KMSAN: store i32 0, ptr [[EXT3_F5]]


; 2 input memory args, 2 output memory args:
;  asm("" : "=m" (id1), "=m" (id2) : "m" (is1), "m"(is2))
define dso_local void @f_2i_2o_mem() sanitize_memory {
entry:
  call void asm "", "=*m,=*m,*m,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) @id1, ptr elementtype(i32) @id2, ptr elementtype(i32) @is1, ptr elementtype(i32) @is2)
  ret void
}

; CHECK-LABEL: @f_2i_2o_mem
; USER-CONS:  store i32 0, ptr inttoptr (i64 xor (i64 ptrtoint (ptr @id1 to i64), i64 87960930222080) to ptr), align 1
; USER-CONS:  store i32 0, ptr inttoptr (i64 xor (i64 ptrtoint (ptr @id2 to i64), i64 87960930222080) to ptr), align 1
; CHECK-CONS: call void @__msan_instrument_asm_store({{.*}}@id1{{.*}}, i64 4)
; CHECK-CONS: call void @__msan_instrument_asm_store({{.*}}@id2{{.*}}, i64 4)
; CHECK: call void asm "", "=*m,=*m,*m,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) @id1, ptr elementtype(i32) @id2, ptr elementtype(i32) @is1, ptr elementtype(i32) @is2)


; Same input and output passed as both memory and register:
;  asm("" : "=r" (id1), "=m"(id1) : "r"(is1), "m"(is1));
define dso_local void @f_1i_1o_memreg() sanitize_memory {
entry:
  %0 = load i32, ptr @is1, align 4
  %1 = call i32 asm "", "=r,=*m,r,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) @id1, i32 %0, ptr elementtype(i32) @is1)
  store i32 %1, ptr @id1, align 4
  ret void
}

; CHECK-LABEL: @f_1i_1o_memreg
; CHECK: [[IS1_F7:%.*]] = load i32, ptr @is1, align 4
; USER-CONS:  store i32 0, ptr inttoptr (i64 xor (i64 ptrtoint (ptr @id1 to i64), i64 87960930222080) to ptr), align 1
; CHECK-CONS: call void @__msan_instrument_asm_store({{.*}}@id1{{.*}}, i64 4)
; CHECK: call void @__msan_warning
; CHECK: call i32 asm "", "=r,=*m,r,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) @id1, i32 [[IS1_F7]], ptr elementtype(i32) @is1)

; Three outputs, first and last returned via regs, second via mem:
;  asm("" : "=r" (id1), "=m"(id2), "=r" (id3):);
define dso_local void @f_3o_reg_mem_reg() sanitize_memory {
entry:
  %0 = call { i32, i32 } asm "", "=r,=*m,=r,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) @id2)
  %asmresult = extractvalue { i32, i32 } %0, 0
  %asmresult1 = extractvalue { i32, i32 } %0, 1
  store i32 %asmresult, ptr @id1, align 4
  store i32 %asmresult1, ptr @id3, align 4
  ret void
}

; CHECK-LABEL: @f_3o_reg_mem_reg
; USER-CONS:  store i32 0, ptr inttoptr (i64 xor (i64 ptrtoint (ptr @id2 to i64), i64 87960930222080) to ptr), align 1
; CHECK-CONS: call void @__msan_instrument_asm_store(ptr @id2, i64 4)
; CHECK: call { i32, i32 } asm "", "=r,=*m,=r,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) @id2)


; Three inputs and three outputs of different types: a pair, a char, a function pointer.
; Everything is meant to be passed in registers, but LLVM chooses to return the integer pair by pointer:
;  asm("" : "=r" (pair2), "=r" (c2), "=r" (memcpy_d1) : "r"(pair1), "r"(c1), "r"(memcpy_s1));
define dso_local void @f_3i_3o_complex_reg() sanitize_memory {
entry:
  %0 = load i64, ptr @pair1, align 4
  %1 = load i8, ptr @c1, align 1
  %2 = load ptr, ptr @memcpy_s1, align 8
  %3 = call { i8, ptr } asm "", "=*r,=r,=r,r,r,r,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(%struct.pair) @pair2, i64 %0, i8 %1, ptr %2)
  %asmresult = extractvalue { i8, ptr } %3, 0
  %asmresult1 = extractvalue { i8, ptr } %3, 1
  store i8 %asmresult, ptr @c2, align 1
  store ptr %asmresult1, ptr @memcpy_d1, align 8
  ret void
}

; CHECK-LABEL: @f_3i_3o_complex_reg
; CHECK: [[PAIR1_F9:%.*]] = load {{.*}} @pair1
; CHECK: [[C1_F9:%.*]] = load {{.*}} @c1
; CHECK: [[MEMCPY_S1_F9:%.*]] = load {{.*}} @memcpy_s1
; USER-CONS:  store { i32, i32 } zeroinitializer, ptr inttoptr (i64 xor (i64 ptrtoint (ptr @pair2 to i64), i64 87960930222080) to ptr), align 1
; CHECK-CONS: call void @__msan_instrument_asm_store({{.*}}@pair2{{.*}}, i64 8)
; CHECK: call void @__msan_warning
; KMSAN: call void @__msan_warning
; KMSAN: call void @__msan_warning
; CHECK: call { i8, ptr } asm "", "=*r,=r,=r,r,r,r,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(%struct.pair) @pair2, {{.*}}[[PAIR1_F9]], i8 [[C1_F9]], {{.*}} [[MEMCPY_S1_F9]])

; Three inputs and three outputs of different types: a pair, a char, a function pointer.
; Everything is passed in memory:
;  asm("" : "=m" (pair2), "=m" (c2), "=m" (memcpy_d1) : "m"(pair1), "m"(c1), "m"(memcpy_s1));
define dso_local void @f_3i_3o_complex_mem() sanitize_memory {
entry:
  call void asm "", "=*m,=*m,=*m,*m,*m,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(%struct.pair) @pair2, ptr elementtype(i8) @c2, ptr elementtype(ptr) @memcpy_d1, ptr elementtype(%struct.pair) @pair1, ptr elementtype(i8) @c1, ptr elementtype(ptr) @memcpy_s1)
  ret void
}

; CHECK-LABEL: @f_3i_3o_complex_mem
; USER-CONS:       store { i32, i32 } zeroinitializer, ptr inttoptr (i64 xor (i64 ptrtoint (ptr @pair2 to i64), i64 87960930222080) to ptr), align 1
; USER-CONS-NEXT:  store i8 0, ptr inttoptr (i64 xor (i64 ptrtoint (ptr @c2 to i64), i64 87960930222080) to ptr), align 1
; USER-CONS-NEXT:  store i64 0, ptr inttoptr (i64 xor (i64 ptrtoint (ptr @memcpy_d1 to i64), i64 87960930222080) to ptr), align 1
; CHECK-CONS:      call void @__msan_instrument_asm_store({{.*}}@pair2{{.*}}, i64 8)
; CHECK-CONS:      call void @__msan_instrument_asm_store({{.*}}@c2{{.*}}, i64 1)
; CHECK-CONS:      call void @__msan_instrument_asm_store({{.*}}@memcpy_d1{{.*}}, i64 8)
; CHECK: call void asm "", "=*m,=*m,=*m,*m,*m,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(%struct.pair) @pair2, ptr elementtype(i8) @c2, ptr elementtype(ptr) @memcpy_d1, ptr elementtype(%struct.pair) @pair1, ptr elementtype(i8) @c1, ptr elementtype(ptr) @memcpy_s1)

; Use memset when the size is larger.
define dso_local void @f_1i_1o_mem_large() sanitize_memory {
entry:
  %0 = call i32 asm "", "=r,=*m,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(%struct.large) @large, ptr elementtype(%struct.large) @large)
  store i32 %0, ptr @id1, align 4
  ret void
}
; CHECK-LABEL: @f_1i_1o_mem_large(
; USER-CONS:  call void @llvm.memset.p0.i64(ptr align 1 inttoptr (i64 xor (i64 ptrtoint (ptr @large to i64), i64 87960930222080) to ptr), i8 0, i64 33, i1 false)
; CHECK-CONS: call void @__msan_instrument_asm_store({{.*}}@large{{.*}}, i64 33)
; CHECK: call i32 asm "", "=r,=*m,*m,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(%struct.large) @large, ptr elementtype(%struct.large) @large)

; A simple asm goto construct to check that callbr is handled correctly:
;  int asm_goto(int n) {
;    int v = 1;
;    asm goto("cmp %0, %1; jnz %l2;" :: "r"(n), "r"(v)::skip_label);
;    return 0;
;  skip_label:
;    return 1;
;  }
; asm goto statements can't have outputs, so just make sure we check the input
; and the compiler doesn't crash.
define dso_local i32 @asm_goto(i32 %n) sanitize_memory {
entry:
  callbr void asm sideeffect "cmp $0, $1; jnz ${2:l}", "r,r,!i,~{dirflag},~{fpsr},~{flags}"(i32 %n, i32 1)
          to label %cleanup [label %skip_label]

skip_label:                                       ; preds = %entry
  br label %cleanup

cleanup:                                          ; preds = %entry, %skip_label
  %retval.0 = phi i32 [ 2, %skip_label ], [ 1, %entry ]
  ret i32 %retval.0
}

; CHECK-LABEL: @asm_goto
; KMSAN: [[LOAD_ARG:%.*]] = load {{.*}} %_msarg
; KMSAN: [[CMP:%.*]] = icmp ne {{.*}} [[LOAD_ARG]], 0
; KMSAN: br {{.*}} [[CMP]], label %[[LABEL:.*]], label
; KMSAN: [[LABEL]]:
; KMSAN-NEXT: call void @__msan_warning

; Test for handling of asm constraints in MSan instrumentation.
; RUN: opt < %s -msan-kernel=1 -msan-check-access-address=0                    \
; RUN: -msan-handle-asm-conservative=0 -S -passes=msan 2>&1 | FileCheck        \
; RUN: "-check-prefixes=CHECK,CHECK-NONCONS" %s
; RUN: opt < %s -msan -msan-kernel=1 -msan-check-access-address=0 -msan-handle-asm-conservative=0 -S | FileCheck -check-prefixes=CHECK,CHECK-NONCONS %s
; RUN: opt < %s -msan-kernel=1 -msan-check-access-address=0                    \
; RUN: -msan-handle-asm-conservative=1 -S -passes=msan 2>&1 | FileCheck        \
; RUN: "-check-prefixes=CHECK,CHECK-CONS" %s
; RUN: opt < %s -msan -msan-kernel=1 -msan-check-access-address=0 -msan-handle-asm-conservative=1 -S | FileCheck -check-prefixes=CHECK,CHECK-CONS %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.pair = type { i32, i32 }

@id1 = common dso_local global i32 0, align 4
@is1 = common dso_local global i32 0, align 4
@id2 = common dso_local global i32 0, align 4
@is2 = common dso_local global i32 0, align 4
@id3 = common dso_local global i32 0, align 4
@pair2 = common dso_local global %struct.pair zeroinitializer, align 4
@pair1 = common dso_local global %struct.pair zeroinitializer, align 4
@c2 = common dso_local global i8 0, align 1
@c1 = common dso_local global i8 0, align 1
@memcpy_d1 = common dso_local global i8* (i8*, i8*, i32)* null, align 8
@memcpy_d2 = common dso_local global i8* (i8*, i8*, i32)* null, align 8
@memcpy_s1 = common dso_local global i8* (i8*, i8*, i32)* null, align 8
@memcpy_s2 = common dso_local global i8* (i8*, i8*, i32)* null, align 8

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
  %0 = load i32, i32* @is1, align 4
  %1 = call i32 asm "", "=r,r,~{dirflag},~{fpsr},~{flags}"(i32 %0)
  store i32 %1, i32* @id1, align 4
  ret void
}

; CHECK-LABEL: @f_1i_1o_reg
; CHECK: [[IS1_F1:%.*]] = load i32, i32* @is1, align 4
; CHECK: call void @__msan_warning
; CHECK: call i32 asm "",{{.*}}(i32 [[IS1_F1]])
; CHECK: [[PACK1_F1:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id1{{.*}})
; CHECK: [[EXT1_F1:%.*]] = extractvalue { i8*, i32* } [[PACK1_F1]], 0
; CHECK: [[CAST1_F1:%.*]] = bitcast i8* [[EXT1_F1]] to i32*
; CHECK: store i32 0, i32* [[CAST1_F1]]


; Two input registers, two output registers:
;   asm("" : "=r" (id1), "=r" (id2) : "r" (is1), "r"(is2));
define dso_local void @f_2i_2o_reg() sanitize_memory {
entry:
  %0 = load i32, i32* @is1, align 4
  %1 = load i32, i32* @is2, align 4
  %2 = call { i32, i32 } asm "", "=r,=r,r,r,~{dirflag},~{fpsr},~{flags}"(i32 %0, i32 %1)
  %asmresult = extractvalue { i32, i32 } %2, 0
  %asmresult1 = extractvalue { i32, i32 } %2, 1
  store i32 %asmresult, i32* @id1, align 4
  store i32 %asmresult1, i32* @id2, align 4
  ret void
}

; CHECK-LABEL: @f_2i_2o_reg
; CHECK: [[IS1_F2:%.*]] = load i32, i32* @is1, align 4
; CHECK: [[IS2_F2:%.*]] = load i32, i32* @is2, align 4
; CHECK: call void @__msan_warning
; CHECK: call void @__msan_warning
; CHECK: call { i32, i32 } asm "",{{.*}}(i32 [[IS1_F2]], i32 [[IS2_F2]])
; CHECK: [[PACK1_F2:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id1{{.*}})
; CHECK: [[EXT1_F2:%.*]] = extractvalue { i8*, i32* } [[PACK1_F2]], 0
; CHECK: [[CAST1_F2:%.*]] = bitcast i8* [[EXT1_F2]] to i32*
; CHECK: store i32 0, i32* [[CAST1_F2]]
; CHECK: [[PACK2_F2:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id2{{.*}})
; CHECK: [[EXT2_F2:%.*]] = extractvalue { i8*, i32* } [[PACK2_F2]], 0
; CHECK: [[CAST2_F2:%.*]] = bitcast i8* [[EXT2_F2]] to i32*
; CHECK: store i32 0, i32* [[CAST2_F2]]

; Input same as output, used twice:
;   asm("" : "=r" (id1), "=r" (id2) : "r" (id1), "r" (id2));
define dso_local void @f_2i_2o_reuse2_reg() sanitize_memory {
entry:
  %0 = load i32, i32* @id1, align 4
  %1 = load i32, i32* @id2, align 4
  %2 = call { i32, i32 } asm "", "=r,=r,r,r,~{dirflag},~{fpsr},~{flags}"(i32 %0, i32 %1)
  %asmresult = extractvalue { i32, i32 } %2, 0
  %asmresult1 = extractvalue { i32, i32 } %2, 1
  store i32 %asmresult, i32* @id1, align 4
  store i32 %asmresult1, i32* @id2, align 4
  ret void
}

; CHECK-LABEL: @f_2i_2o_reuse2_reg
; CHECK: [[ID1_F3:%.*]] = load i32, i32* @id1, align 4
; CHECK: [[ID2_F3:%.*]] = load i32, i32* @id2, align 4
; CHECK: call void @__msan_warning
; CHECK: call void @__msan_warning
; CHECK: call { i32, i32 } asm "",{{.*}}(i32 [[ID1_F3]], i32 [[ID2_F3]])
; CHECK: [[PACK1_F3:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id1{{.*}})
; CHECK: [[EXT1_F3:%.*]] = extractvalue { i8*, i32* } [[PACK1_F3]], 0
; CHECK: [[CAST1_F3:%.*]] = bitcast i8* [[EXT1_F3]] to i32*
; CHECK: store i32 0, i32* [[CAST1_F3]]
; CHECK: [[PACK2_F3:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id2{{.*}})
; CHECK: [[EXT2_F3:%.*]] = extractvalue { i8*, i32* } [[PACK2_F3]], 0
; CHECK: [[CAST2_F3:%.*]] = bitcast i8* [[EXT2_F3]] to i32*
; CHECK: store i32 0, i32* [[CAST2_F3]]


; One of the input registers is also an output:
;   asm("" : "=r" (id1), "=r" (id2) : "r" (id1), "r"(is1));
define dso_local void @f_2i_2o_reuse1_reg() sanitize_memory {
entry:
  %0 = load i32, i32* @id1, align 4
  %1 = load i32, i32* @is1, align 4
  %2 = call { i32, i32 } asm "", "=r,=r,r,r,~{dirflag},~{fpsr},~{flags}"(i32 %0, i32 %1)
  %asmresult = extractvalue { i32, i32 } %2, 0
  %asmresult1 = extractvalue { i32, i32 } %2, 1
  store i32 %asmresult, i32* @id1, align 4
  store i32 %asmresult1, i32* @id2, align 4
  ret void
}

; CHECK-LABEL: @f_2i_2o_reuse1_reg
; CHECK: [[ID1_F4:%.*]] = load i32, i32* @id1, align 4
; CHECK: [[IS1_F4:%.*]] = load i32, i32* @is1, align 4
; CHECK: call void @__msan_warning
; CHECK: call void @__msan_warning
; CHECK: call { i32, i32 } asm "",{{.*}}(i32 [[ID1_F4]], i32 [[IS1_F4]])
; CHECK: [[PACK1_F4:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id1{{.*}})
; CHECK: [[EXT1_F4:%.*]] = extractvalue { i8*, i32* } [[PACK1_F4]], 0
; CHECK: [[CAST1_F4:%.*]] = bitcast i8* [[EXT1_F4]] to i32*
; CHECK: store i32 0, i32* [[CAST1_F4]]
; CHECK: [[PACK2_F4:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id2{{.*}})
; CHECK: [[EXT2_F4:%.*]] = extractvalue { i8*, i32* } [[PACK2_F4]], 0
; CHECK: [[CAST2_F4:%.*]] = bitcast i8* [[EXT2_F4]] to i32*
; CHECK: store i32 0, i32* [[CAST2_F4]]


; One input register, three output registers:
;   asm("" : "=r" (id1), "=r" (id2), "=r" (id3) : "r" (is1));
define dso_local void @f_1i_3o_reg() sanitize_memory {
entry:
  %0 = load i32, i32* @is1, align 4
  %1 = call { i32, i32, i32 } asm "", "=r,=r,=r,r,~{dirflag},~{fpsr},~{flags}"(i32 %0)
  %asmresult = extractvalue { i32, i32, i32 } %1, 0
  %asmresult1 = extractvalue { i32, i32, i32 } %1, 1
  %asmresult2 = extractvalue { i32, i32, i32 } %1, 2
  store i32 %asmresult, i32* @id1, align 4
  store i32 %asmresult1, i32* @id2, align 4
  store i32 %asmresult2, i32* @id3, align 4
  ret void
}

; CHECK-LABEL: @f_1i_3o_reg
; CHECK: [[IS1_F5:%.*]] = load i32, i32* @is1, align 4
; CHECK: call void @__msan_warning
; CHECK: call { i32, i32, i32 } asm "",{{.*}}(i32 [[IS1_F5]])
; CHECK: [[PACK1_F5:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id1{{.*}})
; CHECK: [[EXT1_F5:%.*]] = extractvalue { i8*, i32* } [[PACK1_F5]], 0
; CHECK: [[CAST1_F5:%.*]] = bitcast i8* [[EXT1_F5]] to i32*
; CHECK: store i32 0, i32* [[CAST1_F5]]
; CHECK: [[PACK2_F5:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id2{{.*}})
; CHECK: [[EXT2_F5:%.*]] = extractvalue { i8*, i32* } [[PACK2_F5]], 0
; CHECK: [[CAST2_F5:%.*]] = bitcast i8* [[EXT2_F5]] to i32*
; CHECK: store i32 0, i32* [[CAST2_F5]]
; CHECK: [[PACK3_F5:%.*]] = call {{.*}} @__msan_metadata_ptr_for_store_4({{.*}}@id3{{.*}})
; CHECK: [[EXT3_F5:%.*]] = extractvalue { i8*, i32* } [[PACK3_F5]], 0
; CHECK: [[CAST3_F5:%.*]] = bitcast i8* [[EXT3_F5]] to i32*
; CHECK: store i32 0, i32* [[CAST3_F5]]


; 2 input memory args, 2 output memory args:
;  asm("" : "=m" (id1), "=m" (id2) : "m" (is1), "m"(is2))
define dso_local void @f_2i_2o_mem() sanitize_memory {
entry:
  call void asm "", "=*m,=*m,*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* @id1, i32* @id2, i32* @is1, i32* @is2)
  ret void
}

; CHECK-LABEL: @f_2i_2o_mem
; CHECK-CONS: call void @__msan_instrument_asm_store({{.*}}@id1{{.*}}, i64 4)
; CHECK-CONS: call void @__msan_instrument_asm_store({{.*}}@id2{{.*}}, i64 4)
; CHECK: call void asm "", "=*m,=*m,*m,*m,~{dirflag},~{fpsr},~{flags}"(i32* @id1, i32* @id2, i32* @is1, i32* @is2)


; Same input and output passed as both memory and register:
;  asm("" : "=r" (id1), "=m"(id1) : "r"(is1), "m"(is1));
define dso_local void @f_1i_1o_memreg() sanitize_memory {
entry:
  %0 = load i32, i32* @is1, align 4
  %1 = call i32 asm "", "=r,=*m,r,*m,~{dirflag},~{fpsr},~{flags}"(i32* @id1, i32 %0, i32* @is1)
  store i32 %1, i32* @id1, align 4
  ret void
}

; CHECK-LABEL: @f_1i_1o_memreg
; CHECK: [[IS1_F7:%.*]] = load i32, i32* @is1, align 4
; CHECK-CONS: call void @__msan_instrument_asm_store({{.*}}@id1{{.*}}, i64 4)
; CHECK: call void @__msan_warning
; CHECK: call i32 asm "", "=r,=*m,r,*m,~{dirflag},~{fpsr},~{flags}"(i32* @id1, i32 [[IS1_F7]], i32* @is1)


; Three outputs, first and last returned via regs, second via mem:
;  asm("" : "=r" (id1), "=m"(id2), "=r" (id3):);
define dso_local void @f_3o_reg_mem_reg() sanitize_memory {
entry:
  %0 = call { i32, i32 } asm "", "=r,=*m,=r,~{dirflag},~{fpsr},~{flags}"(i32* @id2)
  %asmresult = extractvalue { i32, i32 } %0, 0
  %asmresult1 = extractvalue { i32, i32 } %0, 1
  store i32 %asmresult, i32* @id1, align 4
  store i32 %asmresult1, i32* @id3, align 4
  ret void
}

; CHECK-LABEL: @f_3o_reg_mem_reg
; CHECK-CONS: call void @__msan_instrument_asm_store({{.*}}@id2{{.*}}), i64 4)
; CHECK: call { i32, i32 } asm "", "=r,=*m,=r,~{dirflag},~{fpsr},~{flags}"(i32* @id2)


; Three inputs and three outputs of different types: a pair, a char, a function pointer.
; Everything is meant to be passed in registers, but LLVM chooses to return the integer pair by pointer:
;  asm("" : "=r" (pair2), "=r" (c2), "=r" (memcpy_d1) : "r"(pair1), "r"(c1), "r"(memcpy_s1));
define dso_local void @f_3i_3o_complex_reg() sanitize_memory {
entry:
  %0 = load i64, i64* bitcast (%struct.pair* @pair1 to i64*), align 4
  %1 = load i8, i8* @c1, align 1
  %2 = load i8* (i8*, i8*, i32)*, i8* (i8*, i8*, i32)** @memcpy_s1, align 8
  %3 = call { i8, i8* (i8*, i8*, i32)* } asm "", "=*r,=r,=r,r,r,r,~{dirflag},~{fpsr},~{flags}"(%struct.pair* @pair2, i64 %0, i8 %1, i8* (i8*, i8*, i32)* %2)
  %asmresult = extractvalue { i8, i8* (i8*, i8*, i32)* } %3, 0
  %asmresult1 = extractvalue { i8, i8* (i8*, i8*, i32)* } %3, 1
  store i8 %asmresult, i8* @c2, align 1
  store i8* (i8*, i8*, i32)* %asmresult1, i8* (i8*, i8*, i32)** @memcpy_d1, align 8
  ret void
}

; CHECK-LABEL: @f_3i_3o_complex_reg
; CHECK: [[PAIR1_F9:%.*]] = load {{.*}} @pair1
; CHECK: [[C1_F9:%.*]] = load {{.*}} @c1
; CHECK: [[MEMCPY_S1_F9:%.*]] = load {{.*}} @memcpy_s1
; CHECK-CONS: call void @__msan_instrument_asm_store({{.*}}@pair2{{.*}}, i64 8)
; CHECK: call void @__msan_warning
; CHECK: call void @__msan_warning
; CHECK: call void @__msan_warning
; CHECK: call { i8, i8* (i8*, i8*, i32)* } asm "", "=*r,=r,=r,r,r,r,~{dirflag},~{fpsr},~{flags}"(%struct.pair* @pair2, {{.*}}[[PAIR1_F9]], i8 [[C1_F9]], {{.*}} [[MEMCPY_S1_F9]])

; Three inputs and three outputs of different types: a pair, a char, a function pointer.
; Everything is passed in memory:
;  asm("" : "=m" (pair2), "=m" (c2), "=m" (memcpy_d1) : "m"(pair1), "m"(c1), "m"(memcpy_s1));
define dso_local void @f_3i_3o_complex_mem() sanitize_memory {
entry:
  call void asm "", "=*m,=*m,=*m,*m,*m,*m,~{dirflag},~{fpsr},~{flags}"(%struct.pair* @pair2, i8* @c2, i8* (i8*, i8*, i32)** @memcpy_d1, %struct.pair* @pair1, i8* @c1, i8* (i8*, i8*, i32)** @memcpy_s1)
  ret void
}

; CHECK-LABEL: @f_3i_3o_complex_mem
; CHECK-CONS: call void @__msan_instrument_asm_store({{.*}}@pair2{{.*}}, i64 8)
; CHECK-CONS: call void @__msan_instrument_asm_store({{.*}}@c2{{.*}}, i64 1)
; CHECK-CONS: call void @__msan_instrument_asm_store({{.*}}@memcpy_d1{{.*}}, i64 8)
; CHECK: call void asm "", "=*m,=*m,=*m,*m,*m,*m,~{dirflag},~{fpsr},~{flags}"(%struct.pair* @pair2, i8* @c2, i8* (i8*, i8*, i32)** @memcpy_d1, %struct.pair* @pair1, i8* @c1, i8* (i8*, i8*, i32)** @memcpy_s1)

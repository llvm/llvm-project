; RUN: llc -mtriple=aarch64-linux-gnu < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-macosx < %s | FileCheck %s

; Test case for TBL optimization of scalar indexed loads
; The pattern is: BUILD_VECTOR of scalar loads where each load uses
; an index extracted from the same vector, accessing the same base table.

; This should be optimized to use TBL when the table is known to be small (<=64 bytes)

;------------------------------------------------------------------------------
; Test 1: Basic pattern - 4 indexed byte loads from same base, known small table
;------------------------------------------------------------------------------

@small_table = constant [16 x i8] c"\00\01\02\03\04\05\06\07\08\09\0A\0B\0C\0D\0E\0F", align 16

; CHECK-LABEL: test_tbl_from_constant:
; CHECK: xtn
; CHECK: uzp1
; CHECK: ldr q
; CHECK: tbl
define <4 x i8> @test_tbl_from_constant(<4 x i32> %indices) {
entry:
  %idx0 = extractelement <4 x i32> %indices, i64 0
  %idx1 = extractelement <4 x i32> %indices, i64 1
  %idx2 = extractelement <4 x i32> %indices, i64 2
  %idx3 = extractelement <4 x i32> %indices, i64 3

  %sext0 = sext i32 %idx0 to i64
  %sext1 = sext i32 %idx1 to i64
  %sext2 = sext i32 %idx2 to i64
  %sext3 = sext i32 %idx3 to i64

  %gep0 = getelementptr inbounds i8, ptr @small_table, i64 %sext0
  %gep1 = getelementptr inbounds i8, ptr @small_table, i64 %sext1
  %gep2 = getelementptr inbounds i8, ptr @small_table, i64 %sext2
  %gep3 = getelementptr inbounds i8, ptr @small_table, i64 %sext3

  %load0 = load i8, ptr %gep0, align 1
  %load1 = load i8, ptr %gep1, align 1
  %load2 = load i8, ptr %gep2, align 1
  %load3 = load i8, ptr %gep3, align 1

  %v0 = insertelement <4 x i8> poison, i8 %load0, i64 0
  %v1 = insertelement <4 x i8> %v0, i8 %load1, i64 1
  %v2 = insertelement <4 x i8> %v1, i8 %load2, i64 2
  %v3 = insertelement <4 x i8> %v2, i8 %load3, i64 3

  ret <4 x i8> %v3
}

;------------------------------------------------------------------------------
; Test 2: With uitofp - common pattern from SPEC benchmark
;------------------------------------------------------------------------------

; CHECK-LABEL: test_tbl_uitofp:
; CHECK: xtn
; CHECK: uzp1
; CHECK: ldr q
; CHECK: tbl
; CHECK: ucvtf
define <4 x float> @test_tbl_uitofp(<4 x i32> %indices) {
entry:
  %idx0 = extractelement <4 x i32> %indices, i64 0
  %idx1 = extractelement <4 x i32> %indices, i64 1
  %idx2 = extractelement <4 x i32> %indices, i64 2
  %idx3 = extractelement <4 x i32> %indices, i64 3

  %sext0 = sext i32 %idx0 to i64
  %sext1 = sext i32 %idx1 to i64
  %sext2 = sext i32 %idx2 to i64
  %sext3 = sext i32 %idx3 to i64

  %gep0 = getelementptr inbounds i8, ptr @small_table, i64 %sext0
  %gep1 = getelementptr inbounds i8, ptr @small_table, i64 %sext1
  %gep2 = getelementptr inbounds i8, ptr @small_table, i64 %sext2
  %gep3 = getelementptr inbounds i8, ptr @small_table, i64 %sext3

  %load0 = load i8, ptr %gep0, align 1
  %load1 = load i8, ptr %gep1, align 1
  %load2 = load i8, ptr %gep2, align 1
  %load3 = load i8, ptr %gep3, align 1

  %f0 = uitofp i8 %load0 to float
  %f1 = uitofp i8 %load1 to float
  %f2 = uitofp i8 %load2 to float
  %f3 = uitofp i8 %load3 to float

  %v0 = insertelement <4 x float> poison, float %f0, i64 0
  %v1 = insertelement <4 x float> %v0, float %f1, i64 1
  %v2 = insertelement <4 x float> %v1, float %f2, i64 2
  %v3 = insertelement <4 x float> %v2, float %f3, i64 3

  ret <4 x float> %v3
}

;------------------------------------------------------------------------------
; Test 3: 32-byte table (TBL2)
;------------------------------------------------------------------------------

@table_32 = constant [32 x i8] zeroinitializer, align 16

; CHECK-LABEL: test_tbl2:
; CHECK: xtn
; CHECK: uzp1
; CHECK: ldp q
; CHECK: tbl
define <4 x i8> @test_tbl2(<4 x i32> %indices) {
entry:
  %idx0 = extractelement <4 x i32> %indices, i64 0
  %idx1 = extractelement <4 x i32> %indices, i64 1
  %idx2 = extractelement <4 x i32> %indices, i64 2
  %idx3 = extractelement <4 x i32> %indices, i64 3

  %sext0 = sext i32 %idx0 to i64
  %sext1 = sext i32 %idx1 to i64
  %sext2 = sext i32 %idx2 to i64
  %sext3 = sext i32 %idx3 to i64

  %gep0 = getelementptr inbounds i8, ptr @table_32, i64 %sext0
  %gep1 = getelementptr inbounds i8, ptr @table_32, i64 %sext1
  %gep2 = getelementptr inbounds i8, ptr @table_32, i64 %sext2
  %gep3 = getelementptr inbounds i8, ptr @table_32, i64 %sext3

  %load0 = load i8, ptr %gep0, align 1
  %load1 = load i8, ptr %gep1, align 1
  %load2 = load i8, ptr %gep2, align 1
  %load3 = load i8, ptr %gep3, align 1

  %v0 = insertelement <4 x i8> poison, i8 %load0, i64 0
  %v1 = insertelement <4 x i8> %v0, i8 %load1, i64 1
  %v2 = insertelement <4 x i8> %v1, i8 %load2, i64 2
  %v3 = insertelement <4 x i8> %v2, i8 %load3, i64 3

  ret <4 x i8> %v3
}

;------------------------------------------------------------------------------
; Test 4: External global with struct field access (SPEC pattern)
;------------------------------------------------------------------------------

%struct.quant_and_transfer_table = type { [32 x i8], [32 x i8], [32 x i8], [65 x i16] }
@quant_and_xfer_tables = external local_unnamed_addr global [12 x %struct.quant_and_transfer_table], align 2

; CHECK-LABEL: test_external_struct_field:
; CHECK: xtn
; CHECK: uzp1
; CHECK: ldp q
; CHECK: tbl
define <4 x i8> @test_external_struct_field(<4 x i32> %indices) {
entry:
  %base = getelementptr inbounds [12 x %struct.quant_and_transfer_table], ptr @quant_and_xfer_tables, i64 0, i64 0, i32 0

  %idx0 = extractelement <4 x i32> %indices, i64 0
  %idx1 = extractelement <4 x i32> %indices, i64 1
  %idx2 = extractelement <4 x i32> %indices, i64 2
  %idx3 = extractelement <4 x i32> %indices, i64 3

  %sext0 = sext i32 %idx0 to i64
  %sext1 = sext i32 %idx1 to i64
  %sext2 = sext i32 %idx2 to i64
  %sext3 = sext i32 %idx3 to i64

  %gep0 = getelementptr inbounds [32 x i8], ptr %base, i64 0, i64 %sext0
  %gep1 = getelementptr inbounds [32 x i8], ptr %base, i64 0, i64 %sext1
  %gep2 = getelementptr inbounds [32 x i8], ptr %base, i64 0, i64 %sext2
  %gep3 = getelementptr inbounds [32 x i8], ptr %base, i64 0, i64 %sext3

  %load0 = load i8, ptr %gep0, align 1
  %load1 = load i8, ptr %gep1, align 1
  %load2 = load i8, ptr %gep2, align 1
  %load3 = load i8, ptr %gep3, align 1

  %v0 = insertelement <4 x i8> poison, i8 %load0, i64 0
  %v1 = insertelement <4 x i8> %v0, i8 %load1, i64 1
  %v2 = insertelement <4 x i8> %v1, i8 %load2, i64 2
  %v3 = insertelement <4 x i8> %v2, i8 %load3, i64 3

  ret <4 x i8> %v3
}

;------------------------------------------------------------------------------
; Test 5: Dynamic base pointer (cannot use TBL without more info)
;------------------------------------------------------------------------------

; CHECK-LABEL: test_dynamic_base:
; Without TBL optimization, we see FPR->GPR extracts and scalar loads
; CHECK-NOT: tbl
; CHECK: ld1
define <4 x i8> @test_dynamic_base(ptr %table, <4 x i32> %indices) {
entry:
  %idx0 = extractelement <4 x i32> %indices, i64 0
  %idx1 = extractelement <4 x i32> %indices, i64 1
  %idx2 = extractelement <4 x i32> %indices, i64 2
  %idx3 = extractelement <4 x i32> %indices, i64 3

  %sext0 = sext i32 %idx0 to i64
  %sext1 = sext i32 %idx1 to i64
  %sext2 = sext i32 %idx2 to i64
  %sext3 = sext i32 %idx3 to i64

  %gep0 = getelementptr inbounds i8, ptr %table, i64 %sext0
  %gep1 = getelementptr inbounds i8, ptr %table, i64 %sext1
  %gep2 = getelementptr inbounds i8, ptr %table, i64 %sext2
  %gep3 = getelementptr inbounds i8, ptr %table, i64 %sext3

  %load0 = load i8, ptr %gep0, align 1
  %load1 = load i8, ptr %gep1, align 1
  %load2 = load i8, ptr %gep2, align 1
  %load3 = load i8, ptr %gep3, align 1

  %v0 = insertelement <4 x i8> poison, i8 %load0, i64 0
  %v1 = insertelement <4 x i8> %v0, i8 %load1, i64 1
  %v2 = insertelement <4 x i8> %v1, i8 %load2, i64 2
  %v3 = insertelement <4 x i8> %v2, i8 %load3, i64 3

  ret <4 x i8> %v3
}

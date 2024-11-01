; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpName [[EQ:%.*]] "test_eq"
; CHECK-DAG: OpName [[NE:%.*]] "test_ne"
; CHECK-DAG: OpName [[ULT:%.*]] "test_ult"
; CHECK-DAG: OpName [[SLT:%.*]] "test_slt"
; CHECK-DAG: OpName [[ULE:%.*]] "test_ule"
; CHECK-DAG: OpName [[SLE:%.*]] "test_sle"
; CHECK-DAG: OpName [[UGT:%.*]] "test_ugt"
; CHECK-DAG: OpName [[SGT:%.*]] "test_sgt"
; CHECK-DAG: OpName [[UGE:%.*]] "test_uge"
; CHECK-DAG: OpName [[SGE:%.*]] "test_sge"

; CHECK-DAG: OpName [[v3EQ:%.*]] "test_v3_eq"
; CHECK-DAG: OpName [[v3NE:%.*]] "test_v3_ne"
; CHECK-DAG: OpName [[v3ULT:%.*]] "test_v3_ult"
; CHECK-DAG: OpName [[v3SLT:%.*]] "test_v3_slt"
; CHECK-DAG: OpName [[v3ULE:%.*]] "test_v3_ule"
; CHECK-DAG: OpName [[v3SLE:%.*]] "test_v3_sle"
; CHECK-DAG: OpName [[v3UGT:%.*]] "test_v3_ugt"
; CHECK-DAG: OpName [[v3SGT:%.*]] "test_v3_sgt"
; CHECK-DAG: OpName [[v3UGE:%.*]] "test_v3_uge"
; CHECK-DAG: OpName [[v3SGE:%.*]] "test_v3_sge"

; CHECK:      [[EQ]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpIEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_eq(i32 %a, i32 %b) {
  %r = icmp eq i32 %a, %b
  ret i1 %r
}

; CHECK:      [[NE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpINotEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_ne(i32 %a, i32 %b) {
  %r = icmp ne i32 %a, %b
  ret i1 %r
}

; CHECK:      [[SLT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpSLessThan {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_slt(i32 %a, i32 %b) {
  %r = icmp slt i32 %a, %b
  ret i1 %r
}

; CHECK:      [[ULT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpULessThan {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_ult(i32 %a, i32 %b) {
  %r = icmp ult i32 %a, %b
  ret i1 %r
}

; CHECK:      [[ULE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpULessThanEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_ule(i32 %a, i32 %b) {
  %r = icmp ule i32 %a, %b
  ret i1 %r
}

; CHECK:      [[SLE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpSLessThanEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_sle(i32 %a, i32 %b) {
  %r = icmp sle i32 %a, %b
  ret i1 %r
}

; CHECK:      [[UGT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpUGreaterThan {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_ugt(i32 %a, i32 %b) {
  %r = icmp ugt i32 %a, %b
  ret i1 %r
}

; CHECK:      [[SGT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpSGreaterThan {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_sgt(i32 %a, i32 %b) {
  %r = icmp sgt i32 %a, %b
  ret i1 %r
}

; CHECK:      [[UGE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpUGreaterThanEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_uge(i32 %a, i32 %b) {
  %r = icmp uge i32 %a, %b
  ret i1 %r
}

; CHECK:      [[SGE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpSGreaterThanEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_sge(i32 %a, i32 %b) {
  %r = icmp sge i32 %a, %b
  ret i1 %r
}

; CHECK:      [[v3EQ]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpIEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_eq(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp eq <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3NE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpINotEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_ne(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp ne <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3SLT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpSLessThan {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_slt(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp slt <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3ULT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpULessThan {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_ult(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp ult <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3ULE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpULessThanEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_ule(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp ule <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3SLE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpSLessThanEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_sle(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp sle <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3UGT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpUGreaterThan {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_ugt(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp ugt <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3SGT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpSGreaterThan {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_sgt(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp sgt <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3UGE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpUGreaterThanEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_uge(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp uge <3 x i32> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3SGE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpSGreaterThanEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_sge(<3 x i32> %a, <3 x i32> %b) {
  %r = icmp sge <3 x i32> %a, %b
  ret <3 x i1> %r
}

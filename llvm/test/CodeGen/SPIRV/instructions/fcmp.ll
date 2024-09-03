; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpName [[UEQ:%.*]] "test_ueq"
; CHECK-DAG: OpName [[OEQ:%.*]] "test_oeq"
; CHECK-DAG: OpName [[UNE:%.*]] "test_une"
; CHECK-DAG: OpName [[ONE:%.*]] "test_one"
; CHECK-DAG: OpName [[ULT:%.*]] "test_ult"
; CHECK-DAG: OpName [[OLT:%.*]] "test_olt"
; CHECK-DAG: OpName [[ULE:%.*]] "test_ule"
; CHECK-DAG: OpName [[OLE:%.*]] "test_ole"
; CHECK-DAG: OpName [[UGT:%.*]] "test_ugt"
; CHECK-DAG: OpName [[OGT:%.*]] "test_ogt"
; CHECK-DAG: OpName [[UGE:%.*]] "test_uge"
; CHECK-DAG: OpName [[OGE:%.*]] "test_oge"
; CHECK-DAG: OpName [[UNO:%.*]] "test_uno"
; CHECK-DAG: OpName [[ORD:%.*]] "test_ord"

; CHECK-DAG: OpName [[v3UEQ:%.*]] "test_v3_ueq"
; CHECK-DAG: OpName [[v3OEQ:%.*]] "test_v3_oeq"
; CHECK-DAG: OpName [[v3UNE:%.*]] "test_v3_une"
; CHECK-DAG: OpName [[v3ONE:%.*]] "test_v3_one"
; CHECK-DAG: OpName [[v3ULT:%.*]] "test_v3_ult"
; CHECK-DAG: OpName [[v3OLT:%.*]] "test_v3_olt"
; CHECK-DAG: OpName [[v3ULE:%.*]] "test_v3_ule"
; CHECK-DAG: OpName [[v3OLE:%.*]] "test_v3_ole"
; CHECK-DAG: OpName [[v3UGT:%.*]] "test_v3_ugt"
; CHECK-DAG: OpName [[v3OGT:%.*]] "test_v3_ogt"
; CHECK-DAG: OpName [[v3UGE:%.*]] "test_v3_uge"
; CHECK-DAG: OpName [[v3OGE:%.*]] "test_v3_oge"
; CHECK-DAG: OpName [[v3UNO:%.*]] "test_v3_uno"
; CHECK-DAG: OpName [[v3ORD:%.*]] "test_v3_ord"

; CHECK:      [[UEQ]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFUnordEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_ueq(float %a, float %b) {
  %r = fcmp ueq float %a, %b
  ret i1 %r
}

; CHECK:      [[OEQ]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFOrdEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_oeq(float %a, float %b) {
  %r = fcmp oeq float %a, %b
  ret i1 %r
}

; CHECK:      [[UNE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFUnordNotEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_une(float %a, float %b) {
  %r = fcmp une float %a, %b
  ret i1 %r
}

; CHECK:      [[ONE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFOrdNotEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_one(float %a, float %b) {
  %r = fcmp one float %a, %b
  ret i1 %r
}

; CHECK:      [[ULT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFUnordLessThan {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_ult(float %a, float %b) {
  %r = fcmp ult float %a, %b
  ret i1 %r
}

; CHECK:      [[OLT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFOrdLessThan {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_olt(float %a, float %b) {
  %r = fcmp olt float %a, %b
  ret i1 %r
}

; CHECK:      [[ULE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFUnordLessThanEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_ule(float %a, float %b) {
  %r = fcmp ule float %a, %b
  ret i1 %r
}

; CHECK:      [[OLE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFOrdLessThanEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_ole(float %a, float %b) {
  %r = fcmp ole float %a, %b
  ret i1 %r
}

; CHECK:      [[UGT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFUnordGreaterThan {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_ugt(float %a, float %b) {
  %r = fcmp ugt float %a, %b
  ret i1 %r
}

; CHECK:      [[OGT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFOrdGreaterThan {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_ogt(float %a, float %b) {
  %r = fcmp ogt float %a, %b
  ret i1 %r
}

; CHECK:      [[UGE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFUnordGreaterThanEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_uge(float %a, float %b) {
  %r = fcmp uge float %a, %b
  ret i1 %r
}

; CHECK:      [[OGE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFOrdGreaterThanEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_oge(float %a, float %b) {
  %r = fcmp oge float %a, %b
  ret i1 %r
}

; CHECK:      [[ORD]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpOrdered {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_ord(float %a, float %b) {
  %r = fcmp ord float %a, %b
  ret i1 %r
}

; CHECK:      [[UNO]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpUnordered {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_uno(float %a, float %b) {
  %r = fcmp uno float %a, %b
  ret i1 %r
}

; CHECK:      [[v3UEQ]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFUnordEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_ueq(<3 x float> %a, <3 x float> %b) {
  %r = fcmp ueq <3 x float> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3OEQ]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFOrdEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_oeq(<3 x float> %a, <3 x float> %b) {
  %r = fcmp oeq <3 x float> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3UNE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFUnordNotEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_une(<3 x float> %a, <3 x float> %b) {
  %r = fcmp une <3 x float> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3ONE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFOrdNotEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_one(<3 x float> %a, <3 x float> %b) {
  %r = fcmp one <3 x float> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3ULT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFUnordLessThan {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_ult(<3 x float> %a, <3 x float> %b) {
  %r = fcmp ult <3 x float> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3OLT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFOrdLessThan {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_olt(<3 x float> %a, <3 x float> %b) {
  %r = fcmp olt <3 x float> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3ULE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFUnordLessThanEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_ule(<3 x float> %a, <3 x float> %b) {
  %r = fcmp ule <3 x float> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3OLE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFOrdLessThanEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_ole(<3 x float> %a, <3 x float> %b) {
  %r = fcmp ole <3 x float> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3UGT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFUnordGreaterThan {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_ugt(<3 x float> %a, <3 x float> %b) {
  %r = fcmp ugt <3 x float> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3OGT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFOrdGreaterThan {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_ogt(<3 x float> %a, <3 x float> %b) {
  %r = fcmp ogt <3 x float> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3UGE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFUnordGreaterThanEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_uge(<3 x float> %a, <3 x float> %b) {
  %r = fcmp uge <3 x float> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3OGE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpFOrdGreaterThanEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_oge(<3 x float> %a, <3 x float> %b) {
  %r = fcmp oge <3 x float> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3ORD]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpOrdered {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_ord(<3 x float> %a, <3 x float> %b) {
  %r = fcmp ord <3 x float> %a, %b
  ret <3 x i1> %r
}

; CHECK:      [[v3UNO]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpUnordered {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define <3 x i1> @test_v3_uno(<3 x float> %a, <3 x float> %b) {
  %r = fcmp uno <3 x float> %a, %b
  ret <3 x i1> %r
}

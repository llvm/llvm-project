; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s --translator-compatibility-mode -o - | FileCheck %s --check-prefix=CHECK-COMPAT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s --translator-compatibility-mode -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName [[EQ:%.*]] "test_eq"
; CHECK-DAG: OpName [[NE:%.*]] "test_ne"
; CHECK-COMPAT-DAG: OpName [[EQ:%.*]] "test_eq"
; CHECK-COMPAT-DAG: OpName [[NE:%.*]] "test_ne"
; CHECK-DAG: OpName [[ULT:%.*]] "test_ult"
; CHECK-DAG: OpName [[SLT:%.*]] "test_slt"
; CHECK-DAG: OpName [[ULE:%.*]] "test_ule"
; CHECK-DAG: OpName [[SLE:%.*]] "test_sle"
; CHECK-DAG: OpName [[UGT:%.*]] "test_ugt"
; CHECK-DAG: OpName [[SGT:%.*]] "test_sgt"
; CHECK-DAG: OpName [[UGE:%.*]] "test_uge"
; CHECK-DAG: OpName [[SGE:%.*]] "test_sge"

;; FIXME: Translator uses OpIEqual/OpINotEqual for test_eq/test_ne cases
; CHECK:      [[EQ]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpPtrEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
; CHECK-COMPAT: [[EQ]] = OpFunction
; CHECK-COMPAT-NOT: OpPtrEqual
; CHECK-COMPAT: OpFunctionEnd
define i1 @test_eq(ptr %a, ptr %b) {
  %r = icmp eq ptr %a, %b
  ret i1 %r
}

; CHECK:      [[NE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpPtrNotEqual {{%.+}} [[A]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
; CHECK-COMPAT: [[NE]] = OpFunction
; CHECK-COMPAT-NOT: OpPtrNotEqual
; CHECK-COMPAT: OpFunctionEnd
define i1 @test_ne(ptr %a, ptr %b) {
  %r = icmp ne ptr %a, %b
  ret i1 %r
}

; CHECK:      [[SLT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[AI:%.*]] = OpConvertPtrToU {{%.+}} [[A]]
; CHECK-NEXT: [[BI:%.*]] = OpConvertPtrToU {{%.+}} [[B]]
; CHECK:      [[R:%.*]] = OpSLessThan {{%.+}} [[AI]] [[BI]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_slt(ptr %a, ptr %b) {
  %r = icmp slt ptr %a, %b
  ret i1 %r
}

; CHECK:      [[ULT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[AI:%.*]] = OpConvertPtrToU {{%.+}} [[A]]
; CHECK-NEXT: [[BI:%.*]] = OpConvertPtrToU {{%.+}} [[B]]
; CHECK:      [[R:%.*]] = OpULessThan {{%.+}} [[AI]] [[BI]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_ult(ptr %a, ptr %b) {
  %r = icmp ult ptr %a, %b
  ret i1 %r
}

; CHECK:      [[ULE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[AI:%.*]] = OpConvertPtrToU {{%.+}} [[A]]
; CHECK-NEXT: [[BI:%.*]] = OpConvertPtrToU {{%.+}} [[B]]
; CHECK:      [[R:%.*]] = OpULessThanEqual {{%.+}} [[AI]] [[BI]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_ule(ptr %a, ptr %b) {
  %r = icmp ule ptr %a, %b
  ret i1 %r
}

; CHECK:      [[SLE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[AI:%.*]] = OpConvertPtrToU {{%.+}} [[A]]
; CHECK-NEXT: [[BI:%.*]] = OpConvertPtrToU {{%.+}} [[B]]
; CHECK:      [[R:%.*]] = OpSLessThanEqual {{%.+}} [[AI]] [[BI]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_sle(ptr %a, ptr %b) {
  %r = icmp sle ptr %a, %b
  ret i1 %r
}

; CHECK:      [[UGT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[AI:%.*]] = OpConvertPtrToU {{%.+}} [[A]]
; CHECK-NEXT: [[BI:%.*]] = OpConvertPtrToU {{%.+}} [[B]]
; CHECK:      [[R:%.*]] = OpUGreaterThan {{%.+}} [[AI]] [[BI]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_ugt(ptr %a, ptr %b) {
  %r = icmp ugt ptr %a, %b
  ret i1 %r
}

; CHECK:      [[SGT]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[AI:%.*]] = OpConvertPtrToU {{%.+}} [[A]]
; CHECK-NEXT: [[BI:%.*]] = OpConvertPtrToU {{%.+}} [[B]]
; CHECK:      [[R:%.*]] = OpSGreaterThan {{%.+}} [[AI]] [[BI]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_sgt(ptr %a, ptr %b) {
  %r = icmp sgt ptr %a, %b
  ret i1 %r
}

; CHECK:      [[UGE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[AI:%.*]] = OpConvertPtrToU {{%.+}} [[A]]
; CHECK-NEXT: [[BI:%.*]] = OpConvertPtrToU {{%.+}} [[B]]
; CHECK:      [[R:%.*]] = OpUGreaterThanEqual {{%.+}} [[AI]] [[BI]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_uge(ptr %a, ptr %b) {
  %r = icmp uge ptr %a, %b
  ret i1 %r
}

; CHECK:      [[SGE]] = OpFunction
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[AI:%.*]] = OpConvertPtrToU {{%.+}} [[A]]
; CHECK-NEXT: [[BI:%.*]] = OpConvertPtrToU {{%.+}} [[B]]
; CHECK:      [[R:%.*]] = OpSGreaterThanEqual {{%.+}} [[AI]] [[BI]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i1 @test_sge(ptr %a, ptr %b) {
  %r = icmp sge ptr %a, %b
  ret i1 %r
}

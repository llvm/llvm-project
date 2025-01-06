; This test shows how value attributes are being passed during different translation steps.
; See also test/CodeGen/SPIRV/optimizations/add-check-overflow.ll.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -print-after=prepare-functions 2>&1 | FileCheck %s  --check-prefix=CHECK-PREPARE
; Intrinsics with aggregate return type are not substituted/removed.
; CHECK-PREPARE: @llvm.uadd.with.overflow.i32

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -print-after=emit-intrinsics 2>&1 | FileCheck %s  --check-prefix=CHECK-IR
; Aggregate data are wrapped into @llvm.fake.use(),
; and their attributes are packed into a metadata for @llvm.spv.value.md().
; CHECK-IR: %[[R1:.*]] = call { i32, i1 } @llvm.uadd.with.overflow.i32
; CHECK-IR: call void @llvm.spv.value.md(metadata !0)
; CHECK-IR: call void (...) @llvm.fake.use({ i32, i1 } %[[R1]])
; CHECK-IR: %math = extractvalue { i32, i1 } %[[R1]], 0
; CHECK-IR: %ov = extractvalue { i32, i1 } %[[R1]], 1
; Type/Name attributes of the value.
; CHECK-IR: !0 = !{{[{]}}!1, !""{{[}]}}
; Origin data type of the value.
; CHECK-IR: !1 = !{{[{]}}{{[{]}} i32, i1 {{[}]}} poison{{[}]}}

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -print-after=irtranslator 2>&1 | FileCheck %s  --check-prefix=CHECK-GMIR
; Required info succeeded to get through IRTranslator.
; CHECK-GMIR: %[[phires:.*]]:_(s32) = G_PHI
; CHECK-GMIR: %[[math:.*]]:id(s32), %[[ov:.*]]:_(s1) = G_UADDO %[[phires]]:_, %[[#]]:_
; CHECK-GMIR: G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.spv.value.md), !0
; CHECK-GMIR: FAKE_USE %[[math]]:id(s32), %[[ov]]:_(s1)

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -print-after=spirv-prelegalizer 2>&1 | FileCheck %s  --check-prefix=CHECK-PRE
; Internal service instructions are consumed.
; CHECK-PRE: G_UADDO
; CHECK-PRE-NO: llvm.spv.value.md
; CHECK-PRE-NO: FAKE_USE

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -print-after=instruction-select 2>&1 | FileCheck %s  --check-prefix=CHECK-ISEL
; Names and types are restored and correctly encoded. Correct instruction selection is completed.
; CHECK-ISEL-DAG: %[[int32:.*]]:type = OpTypeInt 32, 0
; CHECK-ISEL-DAG: %[[struct:.*]]:type = OpTypeStruct %[[int32]]:type, %[[int32]]:type
; CHECK-ISEL-DAG: %[[bool:.*]]:type = OpTypeBool
; CHECK-ISEL-DAG: %[[zero32:.*]]:iid = OpConstantNull %[[int32]]:type
; CHECK-ISEL-DAG: %[[res:.*]]:iid = OpIAddCarryS %[[struct]]:type
; CHECK-ISEL-DAG: %[[math:.*]]:id = OpCompositeExtract %[[int32]]:type, %[[res]]:iid, 0
; CHECK-ISEL-DAG: %[[ov32:.*]]:iid = OpCompositeExtract %[[int32]]:type, %[[res]]:iid, 1
; CHECK-ISEL-DAG: %[[ov:.*]]:iid = OpINotEqual %[[bool]]:type, %[[ov32]]:iid, %[[zero32:.*]]:iid
; CHECK-ISEL-DAG: OpName %[[math]]:id, 1752457581, 0
; CHECK-ISEL-DAG: OpName %[[ov]]:iid, 30319

define spir_func i32 @foo(i32 %a, ptr addrspace(4) %p) {
entry:
  br label %l1

l1:                                               ; preds = %body, %entry
  %e = phi i32 [ %a, %entry ], [ %math, %body ]
  %0 = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %e, i32 1)
  %math = extractvalue { i32, i1 } %0, 0
  %ov = extractvalue { i32, i1 } %0, 1
  br i1 %ov, label %exit, label %body

body:                                             ; preds = %l1
  store i8 42, ptr addrspace(4) %p, align 1
  br label %l1

exit:                                             ; preds = %l1
  ret i32 %math
}

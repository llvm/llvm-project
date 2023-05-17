; RUN: opt -S -passes=wholeprogramdevirt -wholeprogramdevirt-summary-action=import -wholeprogramdevirt-read-summary=%S/Inputs/import-single-impl.yaml < %s | FileCheck --check-prefixes=CHECK,SINGLE-IMPL %s
; RUN: opt -S -passes=wholeprogramdevirt -wholeprogramdevirt-summary-action=import -wholeprogramdevirt-read-summary=%S/Inputs/import-uniform-ret-val.yaml < %s | FileCheck --check-prefixes=CHECK,INDIR,UNIFORM-RET-VAL %s
; RUN: opt -S -passes=wholeprogramdevirt -wholeprogramdevirt-summary-action=import -wholeprogramdevirt-read-summary=%S/Inputs/import-unique-ret-val0.yaml < %s | FileCheck --check-prefixes=CHECK,INDIR,UNIQUE-RET-VAL0 %s
; RUN: opt -S -passes=wholeprogramdevirt -wholeprogramdevirt-summary-action=import -wholeprogramdevirt-read-summary=%S/Inputs/import-unique-ret-val1.yaml < %s | FileCheck --check-prefixes=CHECK,INDIR,UNIQUE-RET-VAL1 %s
; RUN: opt -S -passes=wholeprogramdevirt -wholeprogramdevirt-summary-action=import -wholeprogramdevirt-read-summary=%S/Inputs/import-vcp.yaml < %s | FileCheck --check-prefixes=CHECK,VCP,VCP-X86,VCP64,INDIR %s
; RUN: opt -S -passes=wholeprogramdevirt -wholeprogramdevirt-summary-action=import -wholeprogramdevirt-read-summary=%S/Inputs/import-vcp.yaml -mtriple=i686-unknown-linux -data-layout=e-p:32:32 < %s | FileCheck --check-prefixes=CHECK,VCP,VCP-X86,VCP32 %s
; RUN: opt -S -passes=wholeprogramdevirt -wholeprogramdevirt-summary-action=import -wholeprogramdevirt-read-summary=%S/Inputs/import-vcp.yaml -mtriple=armv7-unknown-linux -data-layout=e-p:32:32 < %s | FileCheck --check-prefixes=CHECK,VCP,VCP-ARM %s
; RUN: opt -S -passes=wholeprogramdevirt -wholeprogramdevirt-summary-action=import -wholeprogramdevirt-read-summary=%S/Inputs/import-vcp-branch-funnel.yaml < %s | FileCheck --check-prefixes=CHECK,VCP,VCP-X86,VCP64,BRANCH-FUNNEL %s
; RUN: opt -S -passes=wholeprogramdevirt -wholeprogramdevirt-summary-action=import -wholeprogramdevirt-read-summary=%S/Inputs/import-branch-funnel.yaml < %s | FileCheck --check-prefixes=CHECK,BRANCH-FUNNEL,BRANCH-FUNNEL-NOVCP %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; VCP-X86: @__typeid_typeid1_0_1_byte = external hidden global [0 x i8], !absolute_symbol !0
; VCP-X86: @__typeid_typeid1_0_1_bit = external hidden global [0 x i8], !absolute_symbol !1
; VCP-X86: @__typeid_typeid2_8_3_byte = external hidden global [0 x i8], !absolute_symbol !0
; VCP-X86: @__typeid_typeid2_8_3_bit = external hidden global [0 x i8], !absolute_symbol !1

; Test cases where the argument values are known and we can apply virtual
; constant propagation.

; CHECK: define i32 @call1
define i32 @call1(ptr %obj) #0 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; SINGLE-IMPL: call i32 @singleimpl1
  %result = call i32 %fptr(ptr %obj, i32 1)
  ; UNIFORM-RET-VAL: ret i32 42
  ; VCP-X86: [[GEP1:%.*]] = getelementptr i8, ptr %vtable, i32 ptrtoint (ptr @__typeid_typeid1_0_1_byte to i32)
  ; VCP-ARM: [[GEP1:%.*]] = getelementptr i8, ptr %vtable, i32 42
  ; VCP: [[LOAD1:%.*]] = load i32, ptr [[GEP1]]
  ; VCP: ret i32 [[LOAD1]]
  ; BRANCH-FUNNEL-NOVCP: call i32 @__typeid_typeid1_0_branch_funnel(ptr nest %vtable, ptr %obj, i32 1)
  ret i32 %result
}

; Test cases where the argument values are unknown, so we cannot apply virtual
; constant propagation.

; CHECK: define i1 @call2
define i1 @call2(ptr %obj) #0 {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 8, metadata !"typeid2")
  %fptr = extractvalue {ptr, i1} %pair, 0
  %p = extractvalue {ptr, i1} %pair, 1
  ; SINGLE-IMPL: br i1 true,
  br i1 %p, label %cont, label %trap

cont:
  ; SINGLE-IMPL: call i1 @singleimpl2
  ; INDIR: call i1 %
  ; BRANCH-FUNNEL: call i1 @__typeid_typeid2_8_branch_funnel(ptr nest %vtable, ptr %obj, i32 undef)
  %result = call i1 %fptr(ptr %obj, i32 undef)
  ret i1 %result

trap:
  call void @llvm.trap()
  unreachable
}

; CHECK: define i1 @call3
define i1 @call3(ptr %obj) #0 {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 8, metadata !"typeid2")
  %fptr = extractvalue {ptr, i1} %pair, 0
  %p = extractvalue {ptr, i1} %pair, 1
  br i1 %p, label %cont, label %trap

cont:
  %result = call i1 %fptr(ptr %obj, i32 3)
  ; UNIQUE-RET-VAL0: icmp ne ptr %vtable, @__typeid_typeid2_8_3_unique_member
  ; UNIQUE-RET-VAL1: icmp eq ptr %vtable, @__typeid_typeid2_8_3_unique_member
  ; VCP-X86: [[GEP2:%.*]] = getelementptr i8, ptr %vtable, i32 ptrtoint (ptr @__typeid_typeid2_8_3_byte to i32)
  ; VCP-ARM: [[GEP2:%.*]] = getelementptr i8, ptr %vtable, i32 43
  ; VCP: [[LOAD2:%.*]] = load i8, ptr [[GEP2]]
  ; VCP-X86: [[AND2:%.*]] = and i8 [[LOAD2]], ptrtoint (ptr @__typeid_typeid2_8_3_bit to i8)
  ; VCP-ARM: [[AND2:%.*]] = and i8 [[LOAD2]], -128
  ; VCP: [[ICMP2:%.*]] = icmp ne i8 [[AND2]], 0
  ; VCP: ret i1 [[ICMP2]]
  ; BRANCH-FUNNEL-NOVCP: call i1 @__typeid_typeid2_8_branch_funnel(ptr nest %vtable, ptr %obj, i32 3)
  ret i1 %result

trap:
  call void @llvm.trap()
  unreachable
}

; SINGLE-IMPL-DAG: declare void @singleimpl1()
; SINGLE-IMPL-DAG: declare void @singleimpl2()

; VCP32: !0 = !{i32 -1, i32 -1}
; VCP64: !0 = !{i64 0, i64 4294967296}

; VCP32: !1 = !{i32 0, i32 256}
; VCP64: !1 = !{i64 0, i64 256}

declare void @llvm.assume(i1)
declare void @llvm.trap()
declare {ptr, i1} @llvm.type.checked.load(ptr, i32, metadata)
declare i1 @llvm.type.test(ptr, metadata)

attributes #0 = { "target-features"="+retpoline" }

; Devirt calls debug counter is not explicitly set. Expect 3 remark messages.
; RUN: opt -S -passes=wholeprogramdevirt -wholeprogramdevirt-summary-action=import \
; RUN:   -pass-remarks=wholeprogramdevirt \
; RUN:   -wholeprogramdevirt-read-summary=%S/Inputs/import-single-impl.yaml \
; RUN:   -print-debug-counter-queries < %s  2>&1 \
; RUN:   | grep "remark" | count 3
; Devirt calls debug counter is set to 1. Expect one remark messages.
; RUN: opt -S -passes=wholeprogramdevirt -wholeprogramdevirt-summary-action=import \
; RUN:   -pass-remarks=wholeprogramdevirt -debug-counter=calls-to-devirt=0 \
; RUN:   -wholeprogramdevirt-read-summary=%S/Inputs/import-single-impl.yaml \
; RUN:   -print-debug-counter-queries < %s  2>&1 \
; RUN:   | FileCheck --check-prefix=CHECK-SINGLE %s
; Devirt calls debug counter is set outside the range of calls. Expect no remark message.
; RUN: opt -S -passes=wholeprogramdevirt -wholeprogramdevirt-summary-action=import \
; RUN:   -pass-remarks=wholeprogramdevirt -debug-counter=calls-to-devirt=9999 \
; RUN:   -wholeprogramdevirt-read-summary=%S/Inputs/import-single-impl.yaml \
; RUN:   -print-debug-counter-queries < %s 2>&1  \
; RUN:   | FileCheck -implicit-check-not="remark" --check-prefix=CHECK-NONE %s

; CHECK-SINGLE: DebugCounter calls-to-devirt=0 execute
; CHECK-SINGLE: remark
; CHECK-SINGLE-SAME: devirtualized a call
; CHECK-SINGLE: DebugCounter calls-to-devirt=1 skip
; CHECK-SINGLE: DebugCounter calls-to-devirt=2 skip

; CHECK-NONE: DebugCounter calls-to-devirt=0 skip
; CHECK-NONE: DebugCounter calls-to-devirt=1 skip
; CHECK-NONE: DebugCounter calls-to-devirt=2 skip

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

define i32 @call1(ptr %obj) #0 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

define i1 @call2(ptr %obj, i32 %arg1) #0 {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 8, metadata !"typeid2")
  %fptr = extractvalue {ptr, i1} %pair, 0
  %p = extractvalue {ptr, i1} %pair, 1
  br i1 %p, label %cont, label %trap

cont:
  %result = call i1 %fptr(ptr %obj, i32 %arg1)
  ret i1 %result

trap:
  call void @llvm.trap()
  unreachable
}

define i1 @call3(ptr %obj) #0 {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 8, metadata !"typeid2")
  %fptr = extractvalue {ptr, i1} %pair, 0
  %p = extractvalue {ptr, i1} %pair, 1
  br i1 %p, label %cont, label %trap

cont:
  %result = call i1 %fptr(ptr %obj, i32 3)
  ret i1 %result

trap:
  call void @llvm.trap()
  unreachable
}

declare void @llvm.assume(i1)
declare void @llvm.trap()
declare {ptr, i1} @llvm.type.checked.load(ptr, i32, metadata)
declare i1 @llvm.type.test(ptr, metadata)

attributes #0 = { "target-features"="+retpoline" }

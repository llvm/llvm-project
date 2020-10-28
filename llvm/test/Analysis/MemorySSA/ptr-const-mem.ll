; RUN: opt -basic-aa -print-memoryssa -verify-memoryssa -enable-new-pm=0 -analyze -memssa-check-limit=0 < %s 2>&1 | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>' -verify-memoryssa -disable-output -memssa-check-limit=0 < %s 2>&1 | FileCheck %s
target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn"

@g4 = external unnamed_addr constant i8, align 1

define signext i8 @cmp_constant(i8* %q, i8 %v) local_unnamed_addr {
entry:

  store i8 %v, i8* %q, align 1
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8 %v, i8* %q, align 1

  %0 = load i8, i8* @g4, align 1
; Make sure that this load is liveOnEntry just based on the fact that @g4 is
; constant memory.
; CHECK: MemoryUse(liveOnEntry)
; CHECK-NEXT: load i8, i8* @g4, align 1

  ret i8 %0
}


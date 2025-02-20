; RUN: opt -passes='spirv-structurizer' -S -mtriple=spirv-unknown-unknown %s | FileCheck %s

; CHECK-LABEL: define spir_func noundef i32 @test_branch
; CHECK: call void @llvm.spv.selection.merge.p0(ptr blockaddress(@test_branch, %if.end), i32 1)
; CHECK-NEXT: br i1 %cmp, label %if.then, label %if.else, !hlsl.controlflow.hint !{{[0-9]+}}
define spir_func noundef i32 @test_branch(i32 noundef %X) {
entry:
  %X.addr = alloca i32, align 4
  %resp = alloca i32, align 4
  store i32 %X, ptr %X.addr, align 4
  %0 = load i32, ptr %X.addr, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else, !hlsl.controlflow.hint !0

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %X.addr, align 4
  %sub = sub nsw i32 0, %1
  store i32 %sub, ptr %resp, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %2 = load i32, ptr %X.addr, align 4
  %mul = mul nsw i32 %2, 2
  store i32 %mul, ptr %resp, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %3 = load i32, ptr %resp, align 4
  ret i32 %3
}

; CHECK-LABEL: define spir_func noundef i32 @test_flatten
; CHECK: call void @llvm.spv.selection.merge.p0(ptr blockaddress(@test_flatten, %if.end), i32 2)
; CHECK-NEXT: br i1 %cmp, label %if.then, label %if.else, !hlsl.controlflow.hint !{{[0-9]+}}
define spir_func noundef i32 @test_flatten(i32 noundef %X) {
entry:
  %X.addr = alloca i32, align 4
  %resp = alloca i32, align 4
  store i32 %X, ptr %X.addr, align 4
  %0 = load i32, ptr %X.addr, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else, !hlsl.controlflow.hint !1

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %X.addr, align 4
  %sub = sub nsw i32 0, %1
  store i32 %sub, ptr %resp, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %2 = load i32, ptr %X.addr, align 4
  %mul = mul nsw i32 %2, 2
  store i32 %mul, ptr %resp, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %3 = load i32, ptr %resp, align 4
  ret i32 %3
}
; CHECK-LABEL: define spir_func noundef i32 @test_no_attr
; CHECK: call void @llvm.spv.selection.merge.p0(ptr blockaddress(@test_no_attr, %if.end), i32 0)
; CHECK-NEXT: br i1 %cmp, label %if.then, label %if.else
define spir_func noundef i32 @test_no_attr(i32 noundef %X) {
entry:
  %X.addr = alloca i32, align 4
  %resp = alloca i32, align 4
  store i32 %X, ptr %X.addr, align 4
  %0 = load i32, ptr %X.addr, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %X.addr, align 4
  %sub = sub nsw i32 0, %1
  store i32 %sub, ptr %resp, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %2 = load i32, ptr %X.addr, align 4
  %mul = mul nsw i32 %2, 2
  store i32 %mul, ptr %resp, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %3 = load i32, ptr %resp, align 4
  ret i32 %3
}

!0 = !{!"hlsl.controlflow.hint", i32 1}
!1 = !{!"hlsl.controlflow.hint", i32 2}

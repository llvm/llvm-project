; RUN: opt -S -wholeprogramdevirt -wholeprogramdevirt-summary-action=import -wholeprogramdevirt-read-summary=%S/Inputs/import-vcp.yaml < %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

define i32 @call1(i1 %a, i8* %obj) {
  %vtableptr = bitcast i8* %obj to [3 x i8*]**
  %vtable = load [3 x i8*]*, [3 x i8*]** %vtableptr
  br i1 %a, label %bb1, label %bb2

bb1:
  ; CHECK: {{.*}} = bitcast {{.*}} to i8*
  %vtablei8_1 = bitcast [3 x i8*]* %vtable to i8*
  %p1 = call i1 @llvm.type.test(i8* %vtablei8_1, metadata !"typeid1")
  call void @llvm.assume(i1 %p1)
  %fptrptr1 = getelementptr [3 x i8*], [3 x i8*]* %vtable, i32 0, i32 0
  %fptr1 = load i8*, i8** %fptrptr1
  %fptr1_casted = bitcast i8* %fptr1 to i32 (i8*, i32)*
  ; CHECK: {{.*}} = bitcast {{.*}} to i8*
  %result1 = call i32 %fptr1_casted(i8* %obj, i32 1)
  br label %bb2

  ; CHECK: :
bb2:
  %vtablei8_2 = bitcast [3 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8_2, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptrptr2 = getelementptr [3 x i8*], [3 x i8*]* %vtable, i32 0, i32 0
  %fptr2 = load i8*, i8** %fptrptr2
  %fptr2_casted = bitcast i8* %fptr2 to i32 (i8*, i32)*
  ; CHECK: {{.*}} = bitcast {{.*}} to i8*
  %result2 = call i32 %fptr2_casted(i8* %obj, i32 1)
  ret i32 %result2
}

declare void @llvm.assume(i1)
declare i1 @llvm.type.test(i8*, metadata)

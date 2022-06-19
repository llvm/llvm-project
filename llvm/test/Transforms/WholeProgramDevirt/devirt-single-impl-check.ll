; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility -pass-remarks=wholeprogramdevirt %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: remark: <unknown>:0:0: single-impl: devirtualized a call to vf
; CHECK: remark: <unknown>:0:0: devirtualized vf
; CHECK-NOT: devirtualized

@vt1 = constant [1 x i8*] [i8* bitcast (void (i8*)* @vf to i8*)], !type !0
@vt2 = constant [1 x i8*] [i8* bitcast (void (i8*)* @vf to i8*)], !type !0

define void @vf(i8* %this) {
  ret void
}

; CHECK: define void @call
define void @call(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %pair = call {i8*, i1} @llvm.type.checked.load(i8* %vtablei8, i32 0, metadata !"typeid")
  %fptr = extractvalue {i8*, i1} %pair, 0
  %p = extractvalue {i8*, i1} %pair, 1
  ; CHECK: br i1 true,
  br i1 %p, label %cont, label %trap

cont:
  %fptr_casted = bitcast i8* %fptr to void (i8*)*
  ; CHECK: call void @vf(
  call void %fptr_casted(i8* %obj)
  ret void

trap:
  call void @llvm.trap()
  unreachable
}

declare {i8*, i1} @llvm.type.checked.load(i8*, i32, metadata)
declare void @llvm.trap()

!0 = !{i32 0, !"typeid"}

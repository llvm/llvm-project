; RUN: llc -mtriple=thumbv7-windows-msvc -fast-isel %s -o - -start-before=stack-protector -stop-after=stack-protector  | FileCheck %s

@var = global ptr null

declare void @callee()

define void @caller1() sspreq {
; CHECK-LABEL: define void @caller1()
; Prologue:

; CHECK: call void @__security_check_cookie

; CHECK: musttail call void @callee()
; CHECK-NEXT: ret void
  %var = alloca [2 x i64]
  store ptr %var, ptr @var
  musttail call void @callee()
  ret void
}

define void @justret() sspreq {
; CHECK-LABEL: define void @justret()
; Prologue:
; CHECK: @llvm.stackguard

; CHECK: call void @__security_check_cookie

; CHECK: ret void
  %var = alloca [2 x i64]
  store ptr %var, ptr @var
  br label %retblock

retblock:
  ret void
}

declare ptr @callee2()

define ptr @caller2() sspreq {
; CHECK-LABEL: define ptr @caller2()
; Prologue:
; CHECK: @llvm.stackguard

; CHECK: call void @__security_check_cookie

; CHECK: [[TMP:%.*]] = musttail call ptr @callee2()
; CHECK-NEXT: ret ptr [[TMP]]

  %var = alloca [2 x i64]
  store ptr %var, ptr @var
  %tmp = musttail call ptr @callee2()
  ret ptr %tmp
}

define void @caller3() sspreq {
; CHECK-LABEL: define void @caller3()
; Prologue:

; CHECK: call void @__security_check_cookie

; CHECK: tail call void @callee()
; CHECK-NEXT: ret void
  %var = alloca [2 x i64]
  store ptr %var, ptr @var
  tail call void @callee()
  ret void
}

define ptr @caller4() sspreq {
; CHECK-LABEL: define ptr @caller4()
; Prologue:
; CHECK: @llvm.stackguard

; CHECK: call void @__security_check_cookie

; CHECK: [[TMP:%.*]] = tail call ptr @callee2()
; CHECK-NEXT: ret ptr [[TMP]]

  %var = alloca [2 x i64]
  store ptr %var, ptr @var
  %tmp = tail call ptr @callee2()
  ret ptr %tmp
}

; RUN: llc -mtriple=thumbv7-windows-msvc -fast-isel %s -o - -start-before=stack-protector -stop-after=stack-protector  | FileCheck %s

@var = global [2 x i64]* null

declare void @callee()

define void @caller1() sspreq {
; CHECK-LABEL: define void @caller1()
; Prologue:

; CHECK: call void @__security_check_cookie

; CHECK: musttail call void @callee()
; CHECK-NEXT: ret void
  %var = alloca [2 x i64]
  store [2 x i64]* %var, [2 x i64]** @var
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
  store [2 x i64]* %var, [2 x i64]** @var
  br label %retblock

retblock:
  ret void
}


declare i64* @callee2()

define i8* @caller2() sspreq {
; CHECK-LABEL: define i8* @caller2()
; Prologue:
; CHECK: @llvm.stackguard

; CHECK: call void @__security_check_cookie

; CHECK: [[TMP:%.*]] = musttail call i64* @callee2()
; CHECK-NEXT: [[RES:%.*]] = bitcast i64* [[TMP]] to i8*
; CHECK-NEXT: ret i8* [[RES]]

  %var = alloca [2 x i64]
  store [2 x i64]* %var, [2 x i64]** @var
  %tmp = musttail call i64* @callee2()
  %res = bitcast i64* %tmp to i8*
  ret i8* %res
}

; RUN: llvm-extract -S -bb "foo:region_start" --replace-with-call %s | FileCheck %s

; CHECK-LABEL: define void @foo() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %a = alloca i32, align 4
; CHECK-NEXT:    %b = alloca i32, align 4
; CHECK-NEXT:    br label %codeRepl
; CHECK-EMPTY:
; CHECK-NEXT:  codeRepl:
; CHECK-NEXT:    call void @foo.region_start(ptr %b)
; CHECK-NEXT:    br label %return
; CHECK-EMPTY:
; CHECK-NEXT:  region_start:
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %a)
; CHECK-NEXT:    store i32 43, ptr %a, align 4
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %a)
; CHECK-NEXT:    store i32 44, ptr %b, align 4
; CHECK-NEXT:    br label %return
; CHECK-EMPTY:
; CHECK-NEXT:  return:
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }


; CHECK-LABEL: define internal void @foo.region_start(ptr %b) {
; CHECK-NEXT:  newFuncRoot:
; CHECK-NEXT:    %a = alloca i32, align 4
; CHECK-NEXT:    br label %region_start
; CHECK-EMPTY:
; CHECK-NEXT:  region_start:
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %a)
; CHECK-NEXT:    store i32 43, ptr %a, align 4
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %a)
; CHECK-NEXT:    store i32 44, ptr %b, align 4
; CHECK-NEXT:    br label %return.exitStub
; CHECK-EMPTY:
; CHECK-NEXT:  return.exitStub:
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }


declare void @llvm.lifetime.start.p0i32(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0i32(i64, ptr nocapture)

define void @foo() {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  br label %region_start

region_start:
  call void @llvm.lifetime.start.p0i32(i64 4, ptr nonnull %a)
  store i32 43, ptr %a
  call void @llvm.lifetime.end.p0i32(i64 4, ptr nonnull %a)
  store i32 44, ptr %b
  br label %return

return:
  ret void
}

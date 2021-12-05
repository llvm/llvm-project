; RUN: llvm-extract -S -bb "foo:region_start" %s --bb-keep-functions --bb-keep-blocks | FileCheck %s



; CHECK-LABEL: define void @foo(i1 %c) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a = alloca i32, align 4
; CHECK-NEXT:   %b = alloca i32, align 4
; CHECK-NEXT:   %A = alloca i32, align 4
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i32(i64 4, i32* nonnull %A)
; CHECK-NEXT:   br i1 %c, label %codeRepl, label %outsideonly
; CHECK-EMPTY:
; CHECK-NEXT: outsideonly:                                     
; CHECK-NEXT:   store i32 41, i32* %b, align 4
; CHECK-NEXT:   store i32 42, i32* %A, align 4
; CHECK-NEXT:   br label %return
; CHECK-EMPTY:
; CHECK-NEXT: codeRepl:                                      
; CHECK-NEXT:   call void @foo.region_start(i32* %a, i32* %b, i32* %A)
; CHECK-NEXT:   br label %region_start.split
; CHECK-EMPTY:
; CHECK-NEXT: region_start:                                
; CHECK-NEXT:   store i32 43, i32* %a, align 4
; CHECK-NEXT:   store i32 44, i32* %b, align 4
; CHECK-NEXT:   store i32 45, i32* %A, align 4
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i32(i64 4, i32* nonnull %B)
; CHECK-NEXT:   store i32 46, i32* %B, align 4
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i32(i64 4, i32* nonnull %B)
; CHECK-NEXT:   br label %region_start.split
; CHECK-EMPTY:
; CHECK-NEXT: region_start.split:                               
; CHECK-NEXT:   br label %return
; CHECK-EMPTY:
; CHECK-NEXT: return:                                          
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i32(i64 4, i32* nonnull %A)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


; CHECK-LABEL: define internal void @foo.region_start(i32* %a, i32* %b, i32* %A) {
; CHECK-NEXT: newFuncRoot:
; CHECK-NEXT:   %B = alloca i32, align 4
; CHECK-NEXT:   br label %region_start
; CHECK-EMPTY:
; CHECK-NEXT: region_start:                                     
; CHECK-NEXT:   store i32 43, i32* %a, align 4
; CHECK-NEXT:   store i32 44, i32* %b, align 4
; CHECK-NEXT:   store i32 45, i32* %A, align 4
; CHECK-NEXT:   call void @llvm.lifetime.start.p0i32(i64 4, i32* nonnull %B)
; CHECK-NEXT:   store i32 46, i32* %B, align 4
; CHECK-NEXT:   call void @llvm.lifetime.end.p0i32(i64 4, i32* nonnull %B)
; CHECK-NEXT:   br label %region_start.split.exitStub
; CHECK-EMPTY:
; CHECK-NEXT: region_start.split.exitStub:                     
; CHECK-NEXT:   ret void
; CHECK-NEXT: }







declare void @llvm.lifetime.start.p0i32(i64, i32* nocapture)
declare void @llvm.lifetime.end.p0i32(i64, i32* nocapture)

define void @foo(i1 %c) {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %A = alloca i32, align 4
  %B = alloca i32, align 4
  call void @llvm.lifetime.start.p0i32(i64 4, i32* nonnull %A)
  br i1 %c, label %region_start, label %outsideonly

outsideonly:
  store i32 41, i32* %b
  store i32 42, i32* %A
  br label %return

region_start:
  store i32 43, i32* %a
  store i32 44, i32* %b
  store i32 45, i32* %A
  call void @llvm.lifetime.start.p0i32(i64 4, i32* nonnull %B)
  store i32 46, i32* %B
  call void @llvm.lifetime.end.p0i32(i64 4, i32* nonnull %B)
  br label %return

return:
  call void @llvm.lifetime.end.p0i32(i64 4, i32* nonnull %A)
  ret void
}

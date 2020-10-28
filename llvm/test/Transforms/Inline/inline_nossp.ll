; RUN: opt -inline -o - -S %s | FileCheck %s
; RUN: opt -passes='cgscc(inline)' %s -S | FileCheck %s
; RUN: opt -always-inline -o - -S %s | FileCheck %s

declare dso_local void @foo(i8*)

define dso_local void @ssp(i64 %0) #0 {
  %2 = alloca i64, align 8
  store i64 %0, i64* %2, align 8
  %3 = load i64, i64* %2, align 8
  %4 = alloca i8, i64 %3, align 16
  call void @foo(i8* %4)
  ret void
}

define dso_local void @ssp_alwaysinline(i64 %0) #1 {
  %2 = alloca i64, align 8
  store i64 %0, i64* %2, align 8
  %3 = load i64, i64* %2, align 8
  %4 = alloca i8, i64 %3, align 16
  call void @foo(i8* %4)
  ret void
}

define dso_local void @nossp() #2 {
; Check that the calls to @ssp and @ssp_alwaysinline are not inlined into
; @nossp, since @nossp does not want a stack protector.
; CHECK-LABEL: @nossp
; CHECK-NEXT: call void @ssp
; CHECK-NEXT: call void @ssp_alwaysinline
  call void @ssp(i64 1024)
  call void @ssp_alwaysinline(i64 1024)
  ret void
}

define dso_local void @nossp_alwaysinline() #3 {
  call void @ssp(i64 1024)
  call void @ssp_alwaysinline(i64 1024)
  ret void
}

define dso_local void @nossp_caller() #2 {
; Permit nossp callee to be inlined into nossp caller.
; CHECK-LABEL: @nossp_caller
; CHECK-NEXT: call void @ssp
; CHECK-NEXT: call void @ssp_alwaysinline
; CHECK-NOT: call void @nossp_alwaysinline
  call void @nossp_alwaysinline()
  ret void
}

define dso_local void @ssp2() #0 {
; Check the call to @nossp is not inlined, since @nossp should not have a stack
; protector.
; CHECK-LABEL: @ssp2
; CHECK-NEXT: call void @nossp
  call void @nossp()
  ret void
}

attributes #0 = { sspstrong }
attributes #1 = { sspstrong alwaysinline }
attributes #2 = { nossp }
attributes #3 = { nossp alwaysinline}

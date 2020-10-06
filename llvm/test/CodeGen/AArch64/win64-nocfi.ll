; RUN: llc < %s -mtriple=aarch64-pc-windows-msvc | FileCheck %s

define dso_local void @"?f@@YAXXZ"() nounwind sspstrong uwtable {
; CHECK-LABEL: f@@YAXXZ
; CHECK-NOT: .seh_proc
; CHECK-NOT: .seh_handlerdata
; CHECK-NOT: .seh_endproc
entry:
  call void @llvm.trap()
  ret void
}

declare void @llvm.trap() noreturn nounwind 

define dso_local i32 @getValue() nounwind sspstrong uwtable {
; CHECK-LABEL: getValue
; CHECK-NOT: .seh_proc
; CHECK-NOT: .seh_endprologue
; CHECK-NOT: .seh_startepilogue
; CHECK-NOT: .seh_endepilogue
; CHECK-NOT: .seh_endproc
entry:
  ret i32 42
}

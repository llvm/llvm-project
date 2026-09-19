; RUN: opt -O3 -S < %s | FileCheck %s
declare void @llvm.assume(i1)

define i1 @src(i8 %x) {
; CHECK-LABEL: @src(
; CHECK:       [[TWICE_MASK:%.*]] = and i8 [[X:%.*]], 127
; CHECK-NEXT:  [[TWICE_NZ:%.*]] = icmp ne i8 [[TWICE_MASK]], 0
; CHECK-NEXT:  call void @llvm.assume(i1 [[TWICE_NZ]])
; CHECK-NEXT:  [[R:%.*]] = icmp ult i8 [[X]], 8
; CHECK-NEXT:  ret i1 [[R]]
;
entry:
  %twice.mask = and i8 %x, 127
  %twice.nz = icmp ne i8 %twice.mask, 0
  call void @llvm.assume(i1 %twice.nz)
  %dec = add i8 %x, -1
  %r = icmp ult i8 %dec, 7
  ret i1 %r
}
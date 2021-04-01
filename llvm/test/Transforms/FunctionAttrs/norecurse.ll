; RUN: opt < %s -basic-aa -function-attrs -rpo-function-attrs -S | FileCheck %s 
; RUN: opt < %s -aa-pipeline=basic-aa -passes='cgscc(function-attrs),rpo-function-attrs' -S | FileCheck %s

; CHECK: Function Attrs
; CHECK-SAME: norecurse nosync nounwind readnone
; CHECK-NEXT: define i32 @leaf()
define i32 @leaf() {
  ret i32 1
}

; CHECK: Function Attrs
; CHECK-SAME: readnone
; CHECK-NOT: norecurse
; CHECK-NEXT: define i32 @self_rec()
define i32 @self_rec() {
  %a = call i32 @self_rec()
  ret i32 4
}

; CHECK: Function Attrs
; CHECK-SAME: readnone
; CHECK-NOT: norecurse
; CHECK-NEXT: define i32 @indirect_rec()
define i32 @indirect_rec() {
  %a = call i32 @indirect_rec2()
  ret i32 %a
}
; CHECK: Function Attrs
; CHECK-SAME: readnone
; CHECK-NOT: norecurse
; CHECK-NEXT: define i32 @indirect_rec2()
define i32 @indirect_rec2() {
  %a = call i32 @indirect_rec()
  ret i32 %a
}

; CHECK: Function Attrs
; CHECK-SAME: readnone
; CHECK-NOT: norecurse
; CHECK-NEXT: define i32 @extern()
define i32 @extern() {
  %a = call i32 @k()
  ret i32 %a
}

; CHECK: Function Attrs
; CHECK-NEXT: declare i32 @k()
declare i32 @k() readnone

; CHECK: Function Attrs
; CHECK-SAME: nounwind
; CHECK-NOT: norecurse
; CHECK-NEXT: define void @intrinsic(i8* nocapture %dest, i8* nocapture readonly %src, i32 %len)
define void @intrinsic(i8* %dest, i8* %src, i32 %len) {
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 %len, i1 false)
  ret void
}

; CHECK: Function Attrs
; CHECK-NEXT: declare void @llvm.memcpy.p0i8.p0i8.i32
declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i1)

; CHECK: Function Attrs
; CHECK-SAME: norecurse nosync readnone
; FIXME: missing "norecurse"
; CHECK-NEXT: define internal i32 @called_by_norecurse()
define internal i32 @called_by_norecurse() {
  %a = call i32 @k()
  ret i32 %a
}
; CHECK: Function Attrs
; CHECK-NEXT: define void @m()
define void @m() norecurse {
  %a = call i32 @called_by_norecurse()
  ret void
}

; CHECK: Function Attrs
; CHECK-SAME: norecurse nosync readnone
; FIXME: missing "norecurse"
; CHECK-NEXT: define internal i32 @called_by_norecurse_indirectly()
define internal i32 @called_by_norecurse_indirectly() {
  %a = call i32 @k()
  ret i32 %a
}
define internal void @o() {
  %a = call i32 @called_by_norecurse_indirectly()
  ret void
}
define void @p() norecurse {
  call void @o()
  ret void
}

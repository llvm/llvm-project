; RUN: opt < %s -basicaa -functionattrs -rpo-functionattrs -S | FileCheck %s --check-prefixes=CHECK,BOTH
; RUN: opt < %s -aa-pipeline=basic-aa -passes='cgscc(function-attrs),rpo-functionattrs' -S | FileCheck %s --check-prefixes=CHECK,BOTH
; RUN: opt -passes=attributor --attributor-disable=false -attributor-max-iterations-verify -attributor-max-iterations=4 -S < %s | FileCheck %s --check-prefixes=ATTRIBUTOR,BOTH

; CHECK: Function Attrs
; CHECK-SAME: norecurse nounwind readnone
; ATTRIBUTOR: Function Attrs: nofree norecurse nosync nounwind readnone willreturn
; BOTH-NEXT: define i32 @leaf()
define i32 @leaf() {
  ret i32 1
}

; BOTH: Function Attrs
; BOTH-SAME: readnone
; BOTH-NOT: norecurse
; BOTH-NEXT: define i32 @self_rec()
define i32 @self_rec() {
  %a = call i32 @self_rec()
  ret i32 4
}

; BOTH: Function Attrs
; BOTH-SAME: readnone
; BOTH-NOT: norecurse
; BOTH-NEXT: define i32 @indirect_rec()
define i32 @indirect_rec() {
  %a = call i32 @indirect_rec2()
  ret i32 %a
}
; BOTH: Function Attrs
; BOTH-SAME: readnone
; BOTH-NOT: norecurse
; BOTH-NEXT: define i32 @indirect_rec2()
define i32 @indirect_rec2() {
  %a = call i32 @indirect_rec()
  ret i32 %a
}

; BOTH: Function Attrs
; BOTH-SAME: readnone
; BOTH-NOT: norecurse
; BOTH-NEXT: define i32 @extern()
define i32 @extern() {
  %a = call i32 @k()
  ret i32 %a
}

; BOTH: Function Attrs
; BOTH-NEXT: declare i32 @k()
declare i32 @k() readnone

; BOTH: Function Attrs
; CHECK-SAME: nounwind
; BOTH-NOT: norecurse
; CHECK-NEXT: define void @intrinsic(i8* nocapture %dest, i8* nocapture readonly %src, i32 %len)
; ATTRIBUTOR-NEXT: define void @intrinsic(i8* nocapture writeonly %dest, i8* nocapture readonly %src, i32 %len)
define void @intrinsic(i8* %dest, i8* %src, i32 %len) {
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 %len, i1 false)
  ret void
}

; BOTH: Function Attrs
; BOTH-NEXT: declare void @llvm.memcpy.p0i8.p0i8.i32
declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i1)

; BOTH: Function Attrs
; CHECK-SAME: norecurse readnone
; FIXME: missing "norecurse"
; ATTRIBUTOR-SAME: nosync readnone
; CHECK-NEXT: define internal i32 @called_by_norecurse()
define internal i32 @called_by_norecurse() {
  %a = call i32 @k()
  ret i32 %a
}
; BOTH: Function Attrs
; BOTH-NEXT: define void @m()
define void @m() norecurse {
  %a = call i32 @called_by_norecurse()
  ret void
}

; BOTH: Function Attrs
; CHECK-SAME: norecurse readnone
; FIXME: missing "norecurse"
; ATTRIBUTOR-SAME: nosync
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

; ATTRIBUTOR: Function Attrs: nofree nosync nounwind
; ATTRIBUTOR-NEXT: define void @f(i32 %x)
define void @f(i32 %x)  {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  %0 = load i32, i32* %x.addr, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:
  call void @g() norecurse
  br label %if.end

if.end:
  ret void
}

; BOTH: define void @g()
define void @g() norecurse {
entry:
  call void @f(i32 0)
  ret void
}

; ATTRIBUTOR-NOT: Function Attrs
; ATTRIBUTOR: define linkonce_odr i32 @leaf_redefinable()
define linkonce_odr i32 @leaf_redefinable() {
  ret i32 1
}

; Call through a function pointer
; ATTRIBUTOR-NOT: Function Attrs
; ATTRIBUTOR: define i32 @eval_func1(i32 (i32)* nocapture nonnull %0, i32 %1)
define i32 @eval_func1(i32 (i32)* , i32) local_unnamed_addr {
  %3 = tail call i32 %0(i32 %1) #2
  ret i32 %3
}

; ATTRIBUTOR-NOT: Function Attrs
; ATTRIBUTOR: define i32 @eval_func2(i32 (i32)* nocapture %0, i32 %1)
define i32 @eval_func2(i32 (i32)* , i32) local_unnamed_addr "null-pointer-is-valid"="true"{
  %3 = tail call i32 %0(i32 %1) #2
  ret i32 %3
}

declare void @unknown()
; Call an unknown function in a dead block.
; ATTRIBUTOR: Function Attrs: nofree norecurse nosync nounwind readnone willreturn
; ATTRIBUTOR: define i32 @call_unknown_in_dead_block()
define i32 @call_unknown_in_dead_block() local_unnamed_addr {
  ret i32 0
Dead:
  tail call void @unknown()
  ret i32 1
}


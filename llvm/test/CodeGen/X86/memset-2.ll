; RUN: llc -mtriple=i386-apple-darwin -mcpu=yonah < %s | FileCheck %s

declare void @llvm.memset.i32(i8*, i8, i32, i32) nounwind

define fastcc void @t1() nounwind {
entry:
; CHECK-LABEL: t1:
; CHECK: calll L_memset$stub
  call void @llvm.memset.p0i8.i32(i8* null, i8 0, i32 188, i32 1, i1 false)
  unreachable
}

define fastcc void @t2(i8 signext %c) nounwind {
entry:
; CHECK-LABEL: t2:
; CHECK: calll L_memset$stub
  call void @llvm.memset.p0i8.i32(i8* undef, i8 %c, i32 76, i32 1, i1 false)
  unreachable
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind

define void @t3(i8* nocapture %s, i8 %a) nounwind {
entry:
  tail call void @llvm.memset.p0i8.i32(i8* %s, i8 %a, i32 8, i32 1, i1 false)
  ret void
; CHECK-LABEL: t3:
; CHECK: imull $16843009
}

define void @t4(i8* nocapture %s, i8 %a) nounwind {
entry:
  tail call void @llvm.memset.p0i8.i32(i8* %s, i8 %a, i32 15, i32 1, i1 false)
  ret void
; CHECK-LABEL: t4:
; CHECK: imull $16843009
; CHECK-NOT: imul
; CHECK: ret
}

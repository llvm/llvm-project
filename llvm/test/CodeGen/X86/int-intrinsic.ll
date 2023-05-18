; RUN: llc < %s -mtriple=i686--    | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s

declare void @llvm.x86.int(i8) nounwind

; CHECK: int3
; CHECK: ret
define void @primitive_int3 () {
bb.entry:
  call void @llvm.x86.int(i8 3) nounwind
  ret void
}

; CHECK: int	$128
; CHECK: ret
define void @primitive_int128 () {
bb.entry:
  call void @llvm.x86.int(i8 128) nounwind
  ret void
}

; RUN: llc %s -enable-machine-outliner=never -mtriple=arm64-apple-darwin -o - | FileCheck %s --check-prefix=DARWIN
; RUN: llc %s -enable-machine-outliner=never -mtriple=arm64-linux-gnu    -o - | FileCheck %s --check-prefix=LINUX
; <rdar://problem/14199482> ARM64: Calls to bzero() replaced with calls to memset()

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1)

; CHECK-LABEL: fct1:
; Constant size memset to zero.
; DARWIN: {{b|bl}} _bzero
; LINUX: {{b|bl}} memset
define void @fct1(i8* nocapture %ptr) minsize {
  tail call void @llvm.memset.p0i8.i64(i8* %ptr, i8 0, i64 256, i1 false)
  ret void
}

; CHECK-LABEL: fct3:
; Variable size memset to zero.
; DARWIN: {{b|bl}} _bzero
; LINUX: {{b|bl}} memset
define void @fct3(i8* nocapture %ptr, i32 %unknown) minsize {
  %conv = sext i32 %unknown to i64
  tail call void @llvm.memset.p0i8.i64(i8* %ptr, i8 0, i64 %conv, i1 false)
  ret void
}

; CHECK-LABEL: fct4:
; Variable size checked memset to zero.
; DARWIN: {{b|bl}} _bzero
; LINUX: {{b|bl}} memset
define void @fct4(i8* %ptr) minsize {
  %tmp = tail call i64 @llvm.objectsize.i64(i8* %ptr, i1 false)
  %call = tail call i8* @__memset_chk(i8* %ptr, i32 0, i64 256, i64 %tmp)
  ret void
}

declare i8* @__memset_chk(i8*, i32, i64, i64)

declare i64 @llvm.objectsize.i64(i8*, i1)

; CHECK-LABEL: fct6:
; Size = unknown, change.
; DARWIN: {{b|bl}} _bzero
; LINUX: {{b|bl}} memset
define void @fct6(i8* %ptr, i32 %unknown) minsize {
  %conv = sext i32 %unknown to i64
  %tmp = tail call i64 @llvm.objectsize.i64(i8* %ptr, i1 false)
  %call = tail call i8* @__memset_chk(i8* %ptr, i32 0, i64 %conv, i64 %tmp)
  ret void
}

; Next functions check that memset is not turned into bzero
; when the set constant is non-zero, whatever the given size.

; CHECK-LABEL: fct7:
; memset with something that is not a zero, no change.
; DARWIN: {{b|bl}} _memset
; LINUX: {{b|bl}} memset
define void @fct7(i8* %ptr) minsize {
  %tmp = tail call i64 @llvm.objectsize.i64(i8* %ptr, i1 false)
  %call = tail call i8* @__memset_chk(i8* %ptr, i32 1, i64 256, i64 %tmp)
  ret void
}

; CHECK-LABEL: fct8:
; memset with something that is not a zero, no change.
; DARWIN: {{b|bl}} _memset
; LINUX: {{b|bl}} memset
define void @fct8(i8* %ptr) minsize {
  %tmp = tail call i64 @llvm.objectsize.i64(i8* %ptr, i1 false)
  %call = tail call i8* @__memset_chk(i8* %ptr, i32 1, i64 257, i64 %tmp)
  ret void
}

; CHECK-LABEL: fct9:
; memset with something that is not a zero, no change.
; DARWIN: {{b|bl}} _memset
; LINUX: {{b|bl}} memset
define void @fct9(i8* %ptr, i32 %unknown) minsize {
  %conv = sext i32 %unknown to i64
  %tmp = tail call i64 @llvm.objectsize.i64(i8* %ptr, i1 false)
  %call = tail call i8* @__memset_chk(i8* %ptr, i32 1, i64 %conv, i64 %tmp)
  ret void
}

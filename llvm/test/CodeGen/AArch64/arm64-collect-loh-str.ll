; RUN: llc -o - %s -mtriple=arm64-apple-ios -O2 | FileCheck %s --implicit-check-not=AdrpAddStr
; RUN: llc -o - %s -mtriple=arm64_32-apple-ios -O2 | FileCheck %s --implicit-check-not=AdrpAddStr
; Test case for <rdar://problem/15942912>.
; AdrpAddStr cannot be used when the store uses same
; register as address and value. Indeed, the related
; if applied, may completely remove the definition or
; at least provide a wrong one (with the offset folded
; into the definition).

@A = internal global i32 0, align 4

define void @str() {
entry:
  store ptr @A, ptr @A, align 4
  ret void
}

define void @stp0(i64 %t) {
entry:
  %addr = getelementptr inbounds i64, ptr @A, i32 1
  store ptr @A, ptr @A, align 4
  store i64 %t, ptr %addr, align 4
  ret void
}

define void @stp1(i64 %t) {
entry:
  %addr = getelementptr inbounds i64, ptr @A, i32 1
  store i64 %t, ptr @A, align 4
  store ptr @A, ptr %addr, align 4
  ret void
}

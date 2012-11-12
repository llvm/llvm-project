; RUN: llc < %s -mtriple=x86_64-apple-darwin10 | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mtriple=x86_64-pc-win32 | FileCheck %s -check-prefix=WIN64

; X64: @bar1
; X64: movzbl
; X64: jmp
; WIN64: @bar1
; WIN64: movzbl
; WIN64: callq
define void @bar1(i1 zeroext %v1) nounwind ssp {
entry:
  %conv = zext i1 %v1 to i32
  %call = tail call i32 (...)* @foo1(i32 %conv) nounwind
  ret void
}

; X64: @bar2
; X64-NOT: movzbl
; X64: jmp
; WIN64: @bar2
; WIN64-NOT: movzbl
; WIN64: callq
define void @bar2(i8 zeroext %v1) nounwind ssp {
entry:
  %conv = zext i8 %v1 to i32
  %call = tail call i32 (...)* @foo1(i32 %conv) nounwind
  ret void
}

; X64: @bar3
; X64: callq
; X64-NOT: movzbl
; X64-NOT: and
; X64: ret
; WIN64: @bar3
; WIN64: callq
; WIN64-NOT: movzbl
; WIN64-NOT: and
; WIN64: ret
define zeroext i1 @bar3() nounwind ssp {
entry:
  %call = call i1 @foo2() nounwind
  ret i1 %call
}

declare i32 @foo1(...)
declare zeroext i1 @foo2()

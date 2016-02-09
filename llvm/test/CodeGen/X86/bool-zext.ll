; RUN: llc < %s -mtriple=i686-unknown-linux-gnu | FileCheck %s -check-prefix=X86
; RUN: llc < %s -mtriple=x86_64-apple-darwin10 | FileCheck %s -check-prefix=X64
; RUN: llc < %s -mtriple=x86_64-pc-win32 | FileCheck %s -check-prefix=WIN64

; Check that the argument gets zero-extended before calling.
; X86-LABEL: bar1
; X86: movzbl
; X86: calll
; X64-LABEL: bar1
; X64: movzbl
; X64: jmp
; WIN64-LABEL: bar1
; WIN64: movzbl
; WIN64: callq
define void @bar1(i1 zeroext %v1) nounwind ssp {
entry:
  %conv = zext i1 %v1 to i32
  %call = tail call i32 (...) @foo1(i32 %conv) nounwind
  ret void
}

; Check that on x86-64 the arguments are simply forwarded.
; X64-LABEL: bar2
; X64-NOT: movzbl
; X64: jmp
; WIN64-LABEL: bar2
; WIN64-NOT: movzbl
; WIN64: callq
define void @bar2(i8 zeroext %v1) nounwind ssp {
entry:
  %conv = zext i8 %v1 to i32
  %call = tail call i32 (...) @foo1(i32 %conv) nounwind
  ret void
}

; Check that i1 return values are not zero-extended.
; X86-LABEL: bar3
; X86: call
; X86-NEXT: {{add|pop}}
; X86-NEXT: ret
; X64-LABEL: bar3
; X64: call
; X64-NEXT: {{add|pop}}
; X64-NEXT: ret
; WIN64-LABEL: bar3
; WIN64: call
; WIN64-NEXT: {{add|pop}}
; WIN64-NEXT: ret
define zeroext i1 @bar3() nounwind ssp {
entry:
  %call = call i1 @foo2() nounwind
  ret i1 %call
}

declare i32 @foo1(...)
declare zeroext i1 @foo2()

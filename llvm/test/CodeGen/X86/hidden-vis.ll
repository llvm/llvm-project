; RUN: llc < %s -mtriple=i686-pc-linux-gnu | FileCheck %s -check-prefix=LINUX
; RUN: llc < %s -mtriple=i686-apple-darwin8 | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mtriple=x86_64-w64-mingw32 | FileCheck %s -check-prefix=WINDOWS


@a = hidden global i32 0
@b = external hidden global i32
@c = global i32* @b

define weak hidden void @t1() nounwind {
; LINUX: .hidden t1
; LINUX: t1:

; DARWIN: .private_extern _t1
; DARWIN: t1:

; WINDOWS: t1:
; WINDOWS-NOT: hidden
  ret void
}

define weak void @t2() nounwind {
; DARWIN: .weak_definition	_t2
  ret void
}

; LINUX: .hidden a
; LINUX: .hidden b

; DARWIN: .private_extern _a
; DARWIN-NOT: private_extern

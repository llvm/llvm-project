; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s -check-prefix=WIN_X32
; RUN: llc < %s -mtriple=i686-pc-mingw32 | FileCheck %s -check-prefix=MINGW_X32
; RUN: llc < %s -mtriple=i386-pc-linux | FileCheck %s -check-prefix=LINUX
; RUN: llc < %s -O0 -mtriple=i686-pc-win32 | FileCheck %s -check-prefix=WIN_X32
; RUN: llc < %s -O0 -mtriple=i686-pc-mingw32 | FileCheck %s -check-prefix=MINGW_X32
; RUN: llc < %s -O0 -mtriple=i386-pc-linux | FileCheck %s -check-prefix=LINUX

; The SysV ABI used by most Unixes and Mingw on x86 specifies that an sret pointer
; is callee-cleanup. However, in MSVC's cdecl calling convention, sret pointer
; arguments are caller-cleanup like normal arguments.

define void @sret1(i8* sret) nounwind {
entry:
; WIN_X32:    {{ret$}}
; MINGW_X32:  ret $4
; LINUX:      ret $4
  ret void
}

define void @sret2(i32* sret %x, i32 %y) nounwind {
entry:
; WIN_X32:    {{ret$}}
; MINGW_X32:  ret $4
; LINUX:      ret $4
  store i32 %y, i32* %x
  ret void
}


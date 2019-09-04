; RUN: llc < %s -mtriple=i686-pc-win32   | FileCheck -check-prefix=CHECK -check-prefix=WIN32 %s
; RUN: llc < %s -mtriple=x86_64-pc-win32 | FileCheck -check-prefix=CHECK -check-prefix=WIN64 %s
; RUN: llc < %s -mtriple=i386-linux-gnu  | FileCheck -check-prefix=LINUX %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

; Don't emit empty functions on Windows; it can lead to duplicate entries
; (multiple functions sharing the same RVA) in the Guard CF Function Table which
; the kernel refuses to load.

define void @f() {
entry:
  unreachable

; CHECK-LABEL: f:
; WIN32: nop
; WIN64: ud2
; LINUX-NOT: nop
; LINUX-NOT: ud2

}

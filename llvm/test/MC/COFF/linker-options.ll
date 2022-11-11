; RUN: llc -O0 -mtriple=i386-pc-win32 -filetype=asm -o - %s | FileCheck %s

!0 = !{!"/DEFAULTLIB:msvcrt.lib"}
!1 = !{!"/DEFAULTLIB:msvcrt.lib", !"/DEFAULTLIB:secur32.lib"}
!2 = !{!"/DEFAULTLIB:\22C:\5Cpath to\5Casan_rt.lib\22"}
!3 = !{!"\22/with spaces\22"}
!llvm.linker.options = !{!0, !1, !2, !3}

define dllexport void @foo() {
  ret void
}

; CHECK: .section        .drectve,"yni"
; CHECK: .ascii   " /DEFAULTLIB:msvcrt.lib"
; CHECK: .ascii   " /DEFAULTLIB:msvcrt.lib"
; CHECK: .ascii   " /DEFAULTLIB:secur32.lib"
; CHECK: .ascii   " /DEFAULTLIB:\"C:\\path to\\asan_rt.lib\""
; CHECK: .ascii   " \"/with spaces\""
; CHECK: .ascii   " /EXPORT:_foo"

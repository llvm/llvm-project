; REQUIRES: system-windows
; RUN: opt -mtriple=x86_64-pc-win32-coff %s -o - | lli

@o = common global i32 0, align 4

define i32 @main() {
  %patatino = load i32, i32* @o, align 4
  ret i32 %patatino
}

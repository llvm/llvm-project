; RUN: not --crash llc -mtriple=x86_64-pc-windows-msvc %s -o /dev/null 2>&1 | FileCheck %s

; The exit for a tail call that changes the stack argument size doesn't release
; AMX tiles, so it is rejected rather than miscompiled.

; CHECK: Can't handle AMX in a Windows x64 tail call exit that changes the stack argument size

declare tailcc void @g(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare tailcc void @h(i64, i64, i64, i64, i64, i64)
declare x86_amx @llvm.x86.tilezero.internal(i16, i16)
declare void @llvm.x86.tilestored64.internal(i16, i16, ptr, i64, x86_amx)

define tailcc void @amx(i64 %a, i64 %b, ptr %p, i1 %c) "target-features"="+amx-tile,+amx-int8" {
  %t = call x86_amx @llvm.x86.tilezero.internal(i16 8, i16 8)
  call void @llvm.x86.tilestored64.internal(i16 8, i16 8, ptr %p, i64 8, x86_amx %t)
  br i1 %c, label %x, label %y
x:
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
y:
  musttail call tailcc void @h(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
}

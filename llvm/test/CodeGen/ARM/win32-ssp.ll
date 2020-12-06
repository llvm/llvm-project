; RUN: llc -mtriple=thumbv7-w64-mingw32 < %s -o - | FileCheck --check-prefix=MINGW %s

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare dso_local void @other(i8*)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

define dso_local void @func() sspstrong {
entry:
; MINGW-LABEL: func:
; MINGW: movw [[REG:r[0-9]+]], :lower16:.refptr.__stack_chk_guard
; MINGW: movt [[REG]], :upper16:.refptr.__stack_chk_guard
; MINGW: ldr [[REG2:r[0-9]+]], {{\[}}[[REG]]]
; MINGW: ldr {{r[0-9]+}}, {{\[}}[[REG2]]]
; MINGW: bl other
; MINGW: ldr {{r[0-9]+}}, {{\[}}[[REG2]]]
; MINGW: bl __stack_chk_fail

  %c = alloca i8, align 1
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %c)
  call void @other(i8* nonnull %c)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %c)
  ret void
}

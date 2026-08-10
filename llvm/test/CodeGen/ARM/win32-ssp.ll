; RUN: llc -mtriple=thumbv7-w64-mingw32 < %s -o - | FileCheck --check-prefix=MINGW %s

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare dso_local void @other(ptr)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)

define dso_local void @func() sspstrong {
entry:
; MINGW-LABEL: func:
; MINGW: movw [[REG:r[0-9]+]], :lower16:.refptr.__stack_chk_guard
; MINGW: movt [[REG]], :upper16:.refptr.__stack_chk_guard
; MINGW: ldr [[REG2:r[0-9]+]], [[[REG]]]
; MINGW: ldr {{r[0-9]+}}, [[[REG2]]]
; MINGW: bl other
; MINGW: movw [[REG3:r[0-9]+]], :lower16:.refptr.__stack_chk_guard
; MINGW: movt [[REG3]], :upper16:.refptr.__stack_chk_guard
; MINGW: ldr [[REG4:r[0-9]+]], [[[REG3]]]
; MINGW: ldr {{r[0-9]+}}, [[[REG4]]]
; MINGW: bl __stack_chk_fail

  %c = alloca i8, align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %c)
  call void @other(ptr nonnull %c)
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %c)
  ret void
}

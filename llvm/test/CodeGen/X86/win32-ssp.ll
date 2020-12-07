; RUN: llc -mtriple=x86_64-w64-mingw32        < %s -o - | FileCheck --check-prefix=MINGW %s
; RUN: llc -mtriple=x86_64-pc-windows-itanium < %s -o - | FileCheck --check-prefix=MSVC  %s
; RUN: llc -mtriple=x86_64-pc-windows-msvc    < %s -o - | FileCheck --check-prefix=MSVC  %s
; RUN: llc -mtriple=i686-w64-mingw32          < %s -o - | FileCheck --check-prefix=MINGW %s

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare dso_local void @other(i8*)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

define dso_local void @func() sspstrong {
entry:
; MINGW-LABEL: func:
; MINGW: mov{{l|q}}  .refptr.[[PREFIX:_?]]__stack_chk_guard{{(\(%rip\))?}}, [[REG:%[a-z]+]]
; MINGW: mov{{l|q}}  ([[REG]])
; MINGW: call{{l|q}} [[PREFIX]]other
; MINGW: mov{{l|q}}  ([[REG]])
; MINGW: call{{l|q}} [[PREFIX]]__stack_chk_fail

; MSVC-LABEL: func:
; MSVC: mov{{l|q}} __security_cookie
; MSVC: callq other
; MSVC: callq __security_check_cookie
; MSVC: .seh_endproc

  %c = alloca i8, align 1
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %c)
  call void @other(i8* nonnull %c)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %c)
  ret void
}

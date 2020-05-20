; RUN: llc < %s | FileCheck %s

; Make sure we check that forwarded memory arguments are not modified when tail
; calling. inalloca and copy arg elimination make argument slots mutable.

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.0.24215"

declare x86_stdcallcc void @tail_std(i32)
declare void @capture(i32*)

define x86_thiscallcc void @inalloca(i32* %this, i32* inalloca %args) {
entry:
  %val = load i32, i32* %args
  store i32 0, i32* %args
  tail call x86_stdcallcc void @tail_std(i32 %val)
  ret void
}

; CHECK-LABEL: _inalloca:                              # @inalloca
; CHECK:         movl    4(%esp), %[[reg:[^ ]*]]
; CHECK:         movl    $0, 4(%esp)
; CHECK:         pushl   %[[reg]]
; CHECK:         calll   _tail_std@4
; CHECK:         retl    $4

define x86_stdcallcc void @copy_elide(i32 %arg) {
entry:
  %arg.ptr = alloca i32
  store i32 %arg, i32* %arg.ptr
  call void @capture(i32* %arg.ptr)
  tail call x86_stdcallcc void @tail_std(i32 %arg)
  ret void
}

; CHECK-LABEL: _copy_elide@4:                          # @copy_elide
; CHECK:         leal    {{[0-9]+}}(%esp), %[[reg:[^ ]*]]
; CHECK:         pushl   %[[reg]]
; CHECK:         calll   _capture
; ...
; CHECK:         calll   _tail_std@4
; CHECK:         retl    $4

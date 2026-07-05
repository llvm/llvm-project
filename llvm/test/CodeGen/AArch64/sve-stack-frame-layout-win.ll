; RUN: llc -mtriple=aarch64-windows-msvc -mattr=+sve -pass-remarks-analysis=stack-frame-layout < %s 2>&1 -o /dev/null | FileCheck %s

; CHECK: Function: f10
; CHECK: Offset: [SP+0-16 x vscale], Type: Spill, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP+0-32 x vscale], Type: Spill, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP+0-48 x vscale], Type: Spill, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP+0-64 x vscale], Type: Spill, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP+0-80 x vscale], Type: Spill, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP+0-96 x vscale], Type: Spill, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP+0-112 x vscale], Type: Spill, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP+0-128 x vscale], Type: Spill, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP+0-144 x vscale], Type: Spill, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP+0-160 x vscale], Type: Spill, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP+0-176 x vscale], Type: Spill, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP+0-192 x vscale], Type: Spill, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP+0-208 x vscale], Type: Spill, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP+0-224 x vscale], Type: Spill, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP+0-240 x vscale], Type: Spill, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP+0-256 x vscale], Type: Spill, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP+0-258 x vscale], Type: Spill, Align: 2, Size: vscale x 2
; CHECK: Offset: [SP+0-260 x vscale], Type: Spill, Align: 2, Size: vscale x 2
; CHECK: Offset: [SP+0-262 x vscale], Type: Spill, Align: 2, Size: vscale x 2
; CHECK: Offset: [SP+0-264 x vscale], Type: Spill, Align: 2, Size: vscale x 2
; CHECK: Offset: [SP+0-266 x vscale], Type: Spill, Align: 2, Size: vscale x 2
; CHECK: Offset: [SP+0-268 x vscale], Type: Spill, Align: 2, Size: vscale x 2
; CHECK: Offset: [SP+0-270 x vscale], Type: Spill, Align: 2, Size: vscale x 2
; CHECK: Offset: [SP+0-272 x vscale], Type: Spill, Align: 2, Size: vscale x 2
; CHECK: Offset: [SP+0-274 x vscale], Type: Spill, Align: 2, Size: vscale x 2
; CHECK: Offset: [SP+0-276 x vscale], Type: Spill, Align: 2, Size: vscale x 2
; CHECK: Offset: [SP+0-278 x vscale], Type: Spill, Align: 2, Size: vscale x 2
; CHECK: Offset: [SP+0-280 x vscale], Type: Spill, Align: 2, Size: vscale x 2
; CHECK: Offset: [SP-16-288 x vscale], Type: Spill, Align: 16, Size: 8
; CHECK: Offset: [SP-24-288 x vscale], Type: Spill, Align: 8, Size: 8
; CHECK: Offset: [SP-32-288 x vscale], Type: Spill, Align: 8, Size: 8
; CHECK: Offset: [SP-32-304 x vscale], Type: Variable, Align: 16, Size: vscale x 16
; CHECK: Offset: [SP-48-304 x vscale], Type: Variable, Align: 8, Size: 16

declare void @g10(ptr,ptr)
define void @f10(i64 %n, <vscale x 2 x i64> %x) "frame-pointer"="all" {
  %p1 = alloca [2 x i64]
  %p2 = alloca <vscale x 2 x i64>
  call void @g10(ptr %p1, ptr %p2)
  ret void
}

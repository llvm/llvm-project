; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; RUN: llc -march=hexagon -O0 < %s | FileCheck -check-prefix=CHECK-CALL %s
; Hexagon Programmer's Reference Manual 11.9.1 SYSTEM/USER

; CHECK-CALL-NOT: call

; Data cache prefetch
declare void @llvm.hexagon.prefetch(i8*)
define void @prefetch(i8* %a) {
  call void @llvm.hexagon.prefetch(i8* %a)
  ret void
}
; CHECK: dcfetch({{.*}} + #0)

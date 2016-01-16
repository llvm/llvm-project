; RUN: llc -march=hexagon -disable-const64=0 < %s | FileCheck %s
; RUN: llc -march=hexagon -disable-const64=1 < %s | FileCheck %s --check-prefix=CHECKOLD

; CHECK: CONST64
; CHECKOLD-NOT: CONST64

target datalayout = "e-m:e-p:32:32-i1:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon"

; Function Attrs: nounwind
define void @foo() optsize {
entry:
  call void @bar(i32 32768, i32 32768, i8 zeroext 1)
  ret void
}

declare void @bar(i32, i32, i8 zeroext)


; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>' -verify-memoryssa -disable-output < %s 2>&1 | FileCheck %s

; CHECK-LABEL: define <vscale x 4 x i32> @f(
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK: MemoryUse(1)
define <vscale x 4 x i32> @f(<vscale x 4 x i32> %z) {
  %a = alloca <vscale x 4 x i32>
  store <vscale x 4 x i32> %z, ptr %a
  %zz = load <vscale x 4 x i32>, ptr %a
  ret <vscale x 4 x i32> %zz
}

; CHECK-LABEL: define i32 @g(
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK: MemoryUse(1)
declare ptr @gg(ptr %a)
define i32 @g(i32 %z, ptr %bb) {
  %a = alloca <vscale x 4 x i32>
  store i32 %z, ptr %a
  %bbb = call ptr @gg(ptr %a) readnone
  %zz = load i32, ptr %bbb
  ret i32 %zz
}

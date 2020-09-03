; RUN: opt -basic-aa -print-memoryssa -verify-memoryssa -analyze < %s 2>&1 | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='print<memoryssa>' -verify-memoryssa -disable-output < %s 2>&1 | FileCheck %s

; CHECK-LABEL: define <vscale x 4 x i32> @f(
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK: MemoryUse(1) MustAlias
define <vscale x 4 x i32> @f(<vscale x 4 x i32> %z) {
  %a = alloca <vscale x 4 x i32>
  store <vscale x 4 x i32> %z, <vscale x 4 x i32>* %a
  %zz = load <vscale x 4 x i32>, <vscale x 4 x i32>* %a
  ret <vscale x 4 x i32> %zz
}

; CHECK-LABEL: define i32 @g(
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK: MemoryUse(1) MayAlias
declare i32* @gg(<vscale x 4 x i32>* %a)
define i32 @g(i32 %z, i32 *%bb) {
  %a = alloca <vscale x 4 x i32>
  %aa = getelementptr <vscale x 4 x i32>, <vscale x 4 x i32>* %a, i32 0, i32 0
  store i32 %z, i32* %aa
  %bbb = call i32* @gg(<vscale x 4 x i32>* %a) readnone
  %zz = load i32, i32* %bbb
  ret i32 %zz
}

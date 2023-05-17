; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a9 %s -o - | FileCheck %s

;  CHECK-LABEL: max:
define i32 @max(i8 %ctx, ptr %ptr, i32 %val)
{
;  CHECK: ldrex
;  CHECK: cmp [[old:r[0-9]*]], [[val:r[0-9]*]]
;  CHECK: movhi {{r[0-9]*}}, [[old]]
  %old = atomicrmw umax ptr %ptr, i32 %val monotonic
  ret i32 %old
}

;  CHECK-LABEL: min:
define i32 @min(i8 %ctx, ptr %ptr, i32 %val)
{
;  CHECK: ldrex
;  CHECK: cmp [[old:r[0-9]*]], [[val:r[0-9]*]]
;  CHECK: movls {{r[0-9]*}}, [[old]]
  %old = atomicrmw umin ptr %ptr, i32 %val monotonic
  ret i32 %old
}

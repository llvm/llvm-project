;  RUN: llc -march=arm -mcpu=cortex-a9 < %s | FileCheck %s

;  CHECK: max:
define i32 @max(i8 %ctx, i32* %ptr, i32 %val)
{
;  CHECK: ldrex
;  CHECK: cmp [[old:r[0-9]*]], [[val:r[0-9]*]]
;  CHECK: movhi {{r[0-9]*}}, [[old]]
  %old = atomicrmw umax i32* %ptr, i32 %val monotonic
  ret i32 %old
}

;  CHECK: min:
define i32 @min(i8 %ctx, i32* %ptr, i32 %val)
{
;  CHECK: ldrex
;  CHECK: cmp [[old:r[0-9]*]], [[val:r[0-9]*]]
;  CHECK: movlo {{r[0-9]*}}, [[old]]
  %old = atomicrmw umin i32* %ptr, i32 %val monotonic
  ret i32 %old
}

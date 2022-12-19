; RUN: llc -stop-after=machine-scheduler -debug-only dagcombine,selectiondag -o - %s 2>&1 | FileCheck %s
; REQUIRES: asserts
; pr42638
target triple = "armv8r-arm-none-eabi"
%struct.__va_list = type { ptr }
define double @foo(i32 %P0, ...) #0 {
entry:
  %V1 = alloca [8 x i8], align 8
  %vl = alloca %struct.__va_list, align 4
  call void asm sideeffect "", "r"(ptr nonnull %V1)
  call void @llvm.va_start(ptr nonnull %vl)
  %argp.cur3 = load ptr, ptr %vl, align 4
  %v.sroa.0.0.copyload = load double, ptr %argp.cur3, align 4
  ret double %v.sroa.0.0.copyload
}

declare void @llvm.va_start(ptr)

attributes #0 = { "target-cpu"="cortex-r52" "target-features"="-fp64"  }

; Ensures that the machine scheduler does not move accessing the upper
; 32 bits of the double to before actually storing it to memory

; CHECK: Creating new node: {{.*}} = add FrameIndex:i32<2>, Constant:i32<4>
; CHECK-NEXT: Creating new node: {{.*}} i32,ch = load<(load (s32) from [[MEM:%.*]] + 4)>
; CHECK: INLINEASM
; CHECK: (load (s32) from [[MEM]] + 4)
; CHECK-NOT: (store (s32) into [[MEM]] + 4)



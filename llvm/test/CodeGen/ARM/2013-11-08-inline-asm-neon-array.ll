;RUN:  not llc -mtriple=arm-linux-gnueabihf < %s 2>&1 | FileCheck %s

; ModuleID = 'bug.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
target triple = "armv7--"

%struct.uint8x8x4_t = type { [4 x <8 x i8>] }

define void @foo() #0 {
  %vsrc = alloca %struct.uint8x8x4_t, align 8
  %ptr = alloca i8;
  %1 = call ptr asm sideeffect "vld4.u8 ${0:h}, [$1], $2", "=*w,=r,r,1"(ptr elementtype(%struct.uint8x8x4_t) %vsrc, i32 0, ptr %ptr)
  ret void
}

; CHECK: error: couldn't allocate output register for constraint 'w'

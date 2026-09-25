; RUN: llc -O0 -mtriple=mips-linux-gnu -global-isel -global-isel-abort=1 -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=CHECK,BE,FP32
; RUN: llc -O0 -mtriple=mipsel-linux-gnu -global-isel -global-isel-abort=1 -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=CHECK,LE,FP32
; RUN: llc -O0 -mtriple=mips-linux-gnu -mcpu=mips32r2 -mattr=+fp64 -global-isel -global-isel-abort=1 -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=CHECK,BE,FP64
; RUN: llc -O0 -mtriple=mipsel-linux-gnu -mcpu=mips32r2 -mattr=+fp64 -global-isel -global-isel-abort=1 -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=CHECK,LE,FP64
; RUN: llc -O0 -mtriple=mips-linux-gnu -global-isel -global-isel-abort=1 -verify-machineinstrs %s -o - > /dev/null
; RUN: llc -O0 -mtriple=mips-linux-gnu -mcpu=mips32r2 -mattr=+fp64 -global-isel -global-isel-abort=1 -verify-machineinstrs %s -o - > /dev/null

; Check high-word-first ABI assignments on BE.
define i64 @integer_regs(i64 %x) {
; CHECK-LABEL: name: integer_regs
; BE: [[LO:%[0-9]+]]:_(s32) = COPY $a1
; BE-NEXT: [[HI:%[0-9]+]]:_(s32) = COPY $a0
; LE: [[LO:%[0-9]+]]:_(s32) = COPY $a0
; LE-NEXT: [[HI:%[0-9]+]]:_(s32) = COPY $a1
; CHECK-NEXT: [[X:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[LO]](s32), [[HI]](s32)
; CHECK-NEXT: [[RLO:%[0-9]+]]:_(s32), [[RHI:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[X]](s64)
; BE-NEXT: $v1 = COPY [[RLO]](s32)
; BE-NEXT: $v0 = COPY [[RHI]](s32)
; LE-NEXT: $v0 = COPY [[RLO]](s32)
; LE-NEXT: $v1 = COPY [[RHI]](s32)
  ret i64 %x
}

; A preceding i32 forces the i64 into the aligned a2/a3 register pair.
declare i64 @callee_integer(i32, i64)
define i64 @integer_call(i64 %x) {
; CHECK-LABEL: name: integer_call
; CHECK: [[X:%[0-9]+]]:_(s64) = G_MERGE_VALUES
; CHECK: [[LO:%[0-9]+]]:_(s32), [[HI:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[X]](s64)
; CHECK: $a0 = COPY
; BE-NEXT: $a3 = COPY [[LO]](s32)
; BE-NEXT: $a2 = COPY [[HI]](s32)
; LE-NEXT: $a2 = COPY [[LO]](s32)
; LE-NEXT: $a3 = COPY [[HI]](s32)
; CHECK-NEXT: JAL @callee_integer,
; BE-NEXT: [[RLO:%[0-9]+]]:_(s32) = COPY $v1
; BE-NEXT: [[RHI:%[0-9]+]]:_(s32) = COPY $v0
; LE-NEXT: [[RLO:%[0-9]+]]:_(s32) = COPY $v0
; LE-NEXT: [[RHI:%[0-9]+]]:_(s32) = COPY $v1
; CHECK-NEXT: {{%[0-9]+}}:_(s64) = G_MERGE_VALUES [[RLO]](s32), [[RHI]](s32)
  %r = call i64 @callee_integer(i32 1, i64 %x)
  ret i64 %r
}

; Check high-word-first stack arguments on BE.
define i64 @integer_stack(i32 %a, i32 %b, i32 %c, i32 %d, i64 %x) {
; CHECK-LABEL: name: integer_stack
; BE: id: 0, type: default, offset: 16, size: 4, alignment: 8,
; BE: id: 1, type: default, offset: 20, size: 4, alignment: 4,
; LE: id: 0, type: default, offset: 20, size: 4, alignment: 4,
; LE: id: 1, type: default, offset: 16, size: 4, alignment: 8,
; CHECK: [[LP:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.1
; CHECK-NEXT: [[LO:%[0-9]+]]:_(s32) = G_LOAD [[LP]](p0)
; CHECK-NEXT: [[HP:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.0
; CHECK-NEXT: [[HI:%[0-9]+]]:_(s32) = G_LOAD [[HP]](p0)
; CHECK-NEXT: {{%[0-9]+}}:_(s64) = G_MERGE_VALUES [[LO]](s32), [[HI]](s32)
  ret i64 %x
}

declare void @callee_stack(i32, i32, i32, i32, i64)
define void @integer_stack_call(i64 %x) {
; CHECK-LABEL: name: integer_stack_call
; CHECK: [[X:%[0-9]+]]:_(s64) = G_MERGE_VALUES
; CHECK: [[LO:%[0-9]+]]:_(s32), [[HI:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[X]](s64)
; BE: G_STORE [[LO]](s32), {{.*}} :: (store (s32) into stack + 20)
; BE: G_STORE [[HI]](s32), {{.*}} :: (store (s32) into stack + 16, align 8)
; LE: G_STORE [[LO]](s32), {{.*}} :: (store (s32) into stack + 16, align 8)
; LE: G_STORE [[HI]](s32), {{.*}} :: (store (s32) into stack + 20)
; CHECK: JAL @callee_stack,
  call void @callee_stack(i32 1, i32 2, i32 3, i32 4, i64 %x)
  ret void
}

; Doubles passed in GPRs use the target's custom assignment handler.
define double @double_regs(i32 %a, double %x) {
; CHECK-LABEL: name: double_regs
; CHECK: [[A2:%[0-9]+]]:_(s32) = COPY $a2
; CHECK-NEXT: [[A3:%[0-9]+]]:_(s32) = COPY $a3
; BE-NEXT: [[X:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[A3]](s32), [[A2]](s32)
; LE-NEXT: [[X:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[A2]](s32), [[A3]](s32)
; FP32-NEXT: $d0 = COPY [[X]](s64)
; FP64-NEXT: $d0_64 = COPY [[X]](s64)
  ret double %x
}

declare double @callee_double(i32, double)
define double @double_call(double %x) {
; CHECK-LABEL: name: double_call
; FP32: [[X:%[0-9]+]]:_(s64) = COPY $d6
; FP64: [[X:%[0-9]+]]:_(s64) = COPY $d12_64
; CHECK: [[LO:%[0-9]+]]:_(s32), [[HI:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[X]](s64)
; CHECK: $a0 = COPY
; BE-NEXT: $a2 = COPY [[HI]](s32)
; BE-NEXT: $a3 = COPY [[LO]](s32)
; LE-NEXT: $a2 = COPY [[LO]](s32)
; LE-NEXT: $a3 = COPY [[HI]](s32)
; CHECK-NEXT: JAL @callee_double,
; FP32-NEXT: [[R:%[0-9]+]]:_(s64) = COPY $d0
; FP64-NEXT: [[R:%[0-9]+]]:_(s64) = COPY $d0_64
; CHECK-NEXT: ADJCALLSTACKUP
; FP32-NEXT: $d0 = COPY [[R]](s64)
; FP64-NEXT: $d0_64 = COPY [[R]](s64)
  %r = call double @callee_double(i32 1, double %x)
  ret double %r
}

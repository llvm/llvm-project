; RUN: llc -mtriple=aarch64-linux-gnu -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

declare void @use_s128(i128 %a, i128 %b)

; Check partially passing a split type on the stack (s128 -> 2 x s64)
; CHECK-LABEL: name: call_use_s128
; CHECK: fixedStack:
; CHECK:  - { id: 0, type: default, offset: 16, size: 4, alignment: 16
; CHECK:  - { id: 1, type: default, offset: 8, size: 8, alignment: 8, stack-id: default,
; CHECK:  - { id: 2, type: default, offset: 0, size: 8, alignment: 16, stack-id: default,
; CHECK: bb.1.entry:
; CHECK:   liveins: $w0, $w4, $w5, $w6, $x2, $x3
; CHECK:   [[COPY:%[0-9]+]]:_(s32) = COPY $w0
; CHECK:   [[COPY1:%[0-9]+]]:_(s64) = COPY $x2
; CHECK:   [[COPY2:%[0-9]+]]:_(s64) = COPY $x3
; CHECK:   [[MV:%[0-9]+]]:_(s128) = G_MERGE_VALUES [[COPY1]](s64), [[COPY2]](s64)
; CHECK:   [[COPY3:%[0-9]+]]:_(s32) = COPY $w4
; CHECK:   [[COPY4:%[0-9]+]]:_(s32) = COPY $w5
; CHECK:   [[COPY5:%[0-9]+]]:_(s32) = COPY $w6
; CHECK:   [[FRAME_INDEX:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.2
; CHECK:   [[LOAD:%[0-9]+]]:_(s64) = G_LOAD [[FRAME_INDEX]](p0) :: (invariant load 8 from %fixed-stack.2, align 16)
; CHECK:   [[FRAME_INDEX1:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.1
; CHECK:   [[LOAD1:%[0-9]+]]:_(s64) = G_LOAD [[FRAME_INDEX1]](p0) :: (invariant load 8 from %fixed-stack.1)
; CHECK:   [[MV1:%[0-9]+]]:_(s128) = G_MERGE_VALUES [[LOAD]](s64), [[LOAD1]](s64)
; CHECK:   [[FRAME_INDEX2:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.0
; CHECK:   [[LOAD2:%[0-9]+]]:_(s32) = G_LOAD [[FRAME_INDEX2]](p0) :: (invariant load 4 from %fixed-stack.0, align 16)
; CHECK:   [[C:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK:   ADJCALLSTACKDOWN 0, 0, implicit-def $sp, implicit $sp
; CHECK:   [[UV:%[0-9]+]]:_(s64), [[UV1:%[0-9]+]]:_(s64) = G_UNMERGE_VALUES [[MV]](s128)
; CHECK:   $x0 = COPY [[UV]](s64)
; CHECK:   $x1 = COPY [[UV1]](s64)
; CHECK:   [[UV2:%[0-9]+]]:_(s64), [[UV3:%[0-9]+]]:_(s64) = G_UNMERGE_VALUES [[MV1]](s128)
; CHECK:   $x2 = COPY [[UV2]](s64)
; CHECK:   $x3 = COPY [[UV3]](s64)
; CHECK:   BL @use_s128, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit $x0, implicit $x1, implicit $x2, implicit $x3
; CHECK:   ADJCALLSTACKUP 0, 0, implicit-def $sp, implicit $sp
; CHECK:   $w0 = COPY [[C]](s32)
; CHECK:   RET_ReallyLR implicit $w0
define i32 @call_use_s128(i32 %p1, i128 %p2, i32 %p3, i32 %p4, i32 %p5, i128 %p6, i32 %p7) {
entry:
  call void @use_s128(i128 %p2, i128 %p6)
  ret i32 0
}

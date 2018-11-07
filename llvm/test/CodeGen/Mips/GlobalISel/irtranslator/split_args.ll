; RUN: llc -O0 -mtriple=mipsel-linux-gnu -global-isel -stop-after=irtranslator -verify-machineinstrs %s -o - | FileCheck %s -check-prefixes=MIPS32

define i64 @i64_reg(i64 %a) {
  ; MIPS32-LABEL: name: i64_reg
  ; MIPS32: bb.1.entry:
  ; MIPS32:   liveins: $a0, $a1
  ; MIPS32:   [[COPY:%[0-9]+]]:_(s32) = COPY $a0
  ; MIPS32:   [[COPY1:%[0-9]+]]:_(s32) = COPY $a1
  ; MIPS32:   [[MV:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[COPY]](s32), [[COPY1]](s32)
  ; MIPS32:   [[UV:%[0-9]+]]:_(s32), [[UV1:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[MV]](s64)
  ; MIPS32:   $v0 = COPY [[UV]](s32)
  ; MIPS32:   $v1 = COPY [[UV1]](s32)
  ; MIPS32:   RetRA implicit $v0, implicit $v1
entry:
  ret i64 %a
}

define i64 @i64_stack(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i64 %a) {
  ; MIPS32-LABEL: name: i64_stack
  ; MIPS32: fixedStack:
  ; MIPS32-DAG:  - { id: [[STACK0:[0-9]+]], type: default, offset: 20, size: 4, alignment: 4,
  ; MIPS32-DAG:  - { id: [[STACK1:[0-9]+]], type: default, offset: 16, size: 4, alignment: 8,
  ; MIPS32: bb.1.entry:
  ; MIPS32:   liveins: $a0, $a1, $a2, $a3
  ; MIPS32:   [[COPY:%[0-9]+]]:_(s32) = COPY $a0
  ; MIPS32:   [[COPY1:%[0-9]+]]:_(s32) = COPY $a1
  ; MIPS32:   [[COPY2:%[0-9]+]]:_(s32) = COPY $a2
  ; MIPS32:   [[COPY3:%[0-9]+]]:_(s32) = COPY $a3
  ; MIPS32:   [[FRAME_INDEX:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.1
  ; MIPS32:   [[LOAD:%[0-9]+]]:_(s32) = G_LOAD [[FRAME_INDEX]](p0) :: (load 4 from %fixed-stack.[[STACK1]], align 0)
  ; MIPS32:   [[FRAME_INDEX1:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.0
  ; MIPS32:   [[LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[FRAME_INDEX1]](p0) :: (load 4 from %fixed-stack.[[STACK0]], align 0)
  ; MIPS32:   [[MV:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[LOAD]](s32), [[LOAD1]](s32)
  ; MIPS32:   [[UV:%[0-9]+]]:_(s32), [[UV1:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[MV]](s64)
  ; MIPS32:   $v0 = COPY [[UV]](s32)
  ; MIPS32:   $v1 = COPY [[UV1]](s32)
  ; MIPS32:   RetRA implicit $v0, implicit $v1
entry:
  ret i64 %a
}

define i64 @i64_reg_allign(i32 %a0, i64 %a) {
  ; MIPS32-LABEL: name: i64_reg_allign
  ; MIPS32: bb.1.entry:
  ; MIPS32:   liveins: $a0, $a2, $a3
  ; MIPS32:   [[COPY:%[0-9]+]]:_(s32) = COPY $a0
  ; MIPS32:   [[COPY1:%[0-9]+]]:_(s32) = COPY $a2
  ; MIPS32:   [[COPY2:%[0-9]+]]:_(s32) = COPY $a3
  ; MIPS32:   [[MV:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[COPY1]](s32), [[COPY2]](s32)
  ; MIPS32:   [[UV:%[0-9]+]]:_(s32), [[UV1:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[MV]](s64)
  ; MIPS32:   $v0 = COPY [[UV]](s32)
  ; MIPS32:   $v1 = COPY [[UV1]](s32)
  ; MIPS32:   RetRA implicit $v0, implicit $v1
entry:
  ret i64 %a
}

define i64 @i64_stack_allign(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %s16, i64 %a) {
  ; MIPS32-LABEL: name: i64_stack_allign
  ; MIPS32: fixedStack:
  ; MIPS32-DAG:  - { id: [[STACK0:[0-9]+]], type: default, offset: 28, size: 4, alignment: 4,
  ; MIPS32-DAG:  - { id: [[STACK1:[0-9]+]], type: default, offset: 24, size: 4, alignment: 8,
  ; MIPS32-DAG:  - { id: [[STACK2:[0-9]+]], type: default, offset: 16, size: 4, alignment: 8,
  ; MIPS32: bb.1.entry:
  ; MIPS32:   liveins: $a0, $a1, $a2, $a3
  ; MIPS32:   [[COPY:%[0-9]+]]:_(s32) = COPY $a0
  ; MIPS32:   [[COPY1:%[0-9]+]]:_(s32) = COPY $a1
  ; MIPS32:   [[COPY2:%[0-9]+]]:_(s32) = COPY $a2
  ; MIPS32:   [[COPY3:%[0-9]+]]:_(s32) = COPY $a3
  ; MIPS32:   [[FRAME_INDEX:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.2
  ; MIPS32:   [[LOAD:%[0-9]+]]:_(s32) = G_LOAD [[FRAME_INDEX]](p0) :: (load 4 from %fixed-stack.[[STACK2]], align 0)
  ; MIPS32:   [[FRAME_INDEX1:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.1
  ; MIPS32:   [[LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[FRAME_INDEX1]](p0) :: (load 4 from %fixed-stack.[[STACK1]], align 0)
  ; MIPS32:   [[FRAME_INDEX2:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.0
  ; MIPS32:   [[LOAD2:%[0-9]+]]:_(s32) = G_LOAD [[FRAME_INDEX2]](p0) :: (load 4 from %fixed-stack.[[STACK0]], align 0)
  ; MIPS32:   [[MV:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[LOAD1]](s32), [[LOAD2]](s32)
  ; MIPS32:   [[UV:%[0-9]+]]:_(s32), [[UV1:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[MV]](s64)
  ; MIPS32:   $v0 = COPY [[UV]](s32)
  ; MIPS32:   $v1 = COPY [[UV1]](s32)
  ; MIPS32:   RetRA implicit $v0, implicit $v1
entry:
  ret i64 %a
}

define i64 @i64_reg_stack(i32 %a0, i32 %a1, i32 %a2, i64 %a) {
  ; MIPS32-LABEL: name: i64_reg_stack
  ; MIPS32: fixedStack:
  ; MIPS32-DAG:  - { id: [[STACK0:[0-9]+]], type: default, offset: 20, size: 4, alignment: 4,
  ; MIPS32-DAG:  - { id: [[STACK1:[0-9]+]], type: default, offset: 16, size: 4, alignment: 8,
  ; MIPS32: bb.1.entry:
  ; MIPS32:   liveins: $a0, $a1, $a2
  ; MIPS32:   [[COPY:%[0-9]+]]:_(s32) = COPY $a0
  ; MIPS32:   [[COPY1:%[0-9]+]]:_(s32) = COPY $a1
  ; MIPS32:   [[COPY2:%[0-9]+]]:_(s32) = COPY $a2
  ; MIPS32:   [[FRAME_INDEX:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.1
  ; MIPS32:   [[LOAD:%[0-9]+]]:_(s32) = G_LOAD [[FRAME_INDEX]](p0) :: (load 4 from %fixed-stack.[[STACK1]], align 0)
  ; MIPS32:   [[FRAME_INDEX1:%[0-9]+]]:_(p0) = G_FRAME_INDEX %fixed-stack.0
  ; MIPS32:   [[LOAD1:%[0-9]+]]:_(s32) = G_LOAD [[FRAME_INDEX1]](p0) :: (load 4 from %fixed-stack.[[STACK0]], align 0)
  ; MIPS32:   [[MV:%[0-9]+]]:_(s64) = G_MERGE_VALUES [[LOAD]](s32), [[LOAD1]](s32)
  ; MIPS32:   [[UV:%[0-9]+]]:_(s32), [[UV1:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES [[MV]](s64)
  ; MIPS32:   $v0 = COPY [[UV]](s32)
  ; MIPS32:   $v1 = COPY [[UV1]](s32)
  ; MIPS32:   RetRA implicit $v0, implicit $v1
entry:
  ret i64 %a
}

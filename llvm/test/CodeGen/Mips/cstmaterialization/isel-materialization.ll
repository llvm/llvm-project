; RUN: llc -mtriple=mips < %s -debug 2>&1 | FileCheck %s --check-prefix=MIPS
; RUN: llc -mtriple=mips -mattr=+micromips < %s -debug 2>&1 | FileCheck %s --check-prefix=MM

; REQUIRES: asserts

; Test that the correct ISA is selected for the materialization of constants.

; The four parameters are picked to use these instructions: li16, addiu, lui,
; lui+addiu.

declare void @e(i32)
declare void @f(i32, i32, i32)
define void @g() {
entry:
  call void @f (i32 1, i32 2048, i32 8388608)
  call void @e (i32 150994946)
  ret void
}

; MIPS-LABEL: ===== Instruction selection ends:
; MIPS-DAG: t{{[0-9]+}}: i32 = ADDiu Register:i32 $zero, TargetConstant:i32<1>
; MIPS-DAG: t{{[0-9]+}}: i32 = ADDiu Register:i32 $zero, TargetConstant:i32<2048>
; MIPS-DAG: t{{[0-9]+}}: i32 = LUi TargetConstant:i32<128>
; MIPS:     t{{[0-9]+}}: ch,glue = JAL TargetGlobalAddress:i32<ptr @f>

; MIPS:     t[[A:[0-9]+]]: i32 = LUi TargetConstant:i32<2304>
; MIPS:     t{{[0-9]+}}: i32 = ORi t[[A]], TargetConstant:i32<2>

; MM-LABEL: ===== Instruction selection ends:
; MM-DAG: t{{[0-9]+}}: i32 = LI16_MM TargetConstant:i32<1>
; MM-DAG: t{{[0-9]+}}: i32 = ADDiu_MM Register:i32 $zero, TargetConstant:i32<2048>
; MM-DAG: t{{[0-9]+}}: i32 = LUi_MM TargetConstant:i32<128>
; MM:     t{{[0-9]+}}: ch,glue = JAL_MM TargetGlobalAddress:i32<ptr @f>

; MM:     t[[A:[0-9]+]]: i32 = LUi_MM TargetConstant:i32<2304>
; MM:     t{{[0-9]+}}: i32 = ORi_MM t[[A]], TargetConstant:i32<2>

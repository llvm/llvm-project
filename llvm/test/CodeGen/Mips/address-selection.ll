; RUN: llc -mtriple=mips < %s -debug 2>&1 | FileCheck %s --check-prefix=MIPS
; RUN: llc -mtriple=mips -relocation-model=pic -mattr=+xgot < %s \
; RUN:     -debug 2>&1 | FileCheck %s --check-prefix=MIPS-XGOT

; RUN: llc -mtriple=mips -mattr=+micromips < %s -debug 2>&1 | FileCheck %s --check-prefix=MM
; RUN: llc -mtriple=mips -relocation-model=pic -mattr=+xgot,+micromips < %s \
; RUN:     -debug 2>&1 | FileCheck %s --check-prefix=MM-XGOT

; REQUIRES: asserts

; Tests that the correct ISA is selected for computing a global address.

@x = global i32 0
@a = global i32 1
declare i32 @y(ptr, i32)

define i32 @z() {
entry:
  %0 = load i32, ptr @a, align 4
  %1 = call i32 @y(ptr @x, i32 %0)
  ret i32 %1
}

; MIPS-LABEL: ===== Instruction selection ends:
; MIPS: t[[A:[0-9]+]]: i32 = LUi TargetGlobalAddress:i32<ptr @x> 0 [TF=4]
; MIPS: t{{.*}}: i32 = ADDiu t[[A]], TargetGlobalAddress:i32<ptr @x> 0 [TF=5]

; MIPS-XGOT-LABEL: ===== Instruction selection ends:
; MIPS-XGOT: t[[B:[0-9]+]]: i32 = LUi TargetGlobalAddress:i32<ptr @x> 0 [TF=20]
; MIPS-XGOT: t[[C:[0-9]+]]: i32 = ADDu t[[B]], Register:i32 %0
; MIPS-XGOT: t{{.*}}: i32,ch = LW<Mem:(load (s32) from got)> t[[C]], TargetGlobalAddress:i32<ptr @x> 0 [TF=21], t{{.*}}

; MM-LABEL: ===== Instruction selection ends:
; MM: t[[A:[0-9]+]]: i32 = LUi_MM TargetGlobalAddress:i32<ptr @x> 0 [TF=4]
; MM: t{{.*}}: i32 = ADDiu_MM t[[A]], TargetGlobalAddress:i32<ptr @x> 0 [TF=5]

; MM-XGOT-LABEL: ===== Instruction selection ends:
; MM-XGOT: t[[B:[0-9]+]]: i32 = LUi_MM TargetGlobalAddress:i32<ptr @x> 0 [TF=20]
; MM-XGOT: t[[C:[0-9]+]]: i32 = ADDU16_MM t[[B]], Register:i32 %0
; MM-XGOT: t{{.*}}: i32,ch = LW_MM<Mem:(load (s32) from got)> t[[C]], TargetGlobalAddress:i32<ptr @x> 0 [TF=21], t0

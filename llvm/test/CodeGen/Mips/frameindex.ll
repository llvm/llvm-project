; RUN: llc -mtriple=mips-mti-linux-gnu < %s -debug 2>&1 | FileCheck %s --check-prefixes=CHECK,MIPS32
; RUN: llc -mtriple=mips-mti-linux-gnu -mattr=+micromips < %s -debug 2>&1 | FileCheck %s --check-prefixes=CHECK,MM
; RUN: llc -mtriple=mips64-mti-linux-gnu < %s -debug 2>&1 | FileCheck %s --check-prefixes=CHECK,MIPS64

; REQUIRES: asserts

; CHECK-LABEL: Instruction selection ends:

; MIPS32: t{{[0-9]+}}: i{{[0-9]+}} = LEA_ADDiu TargetFrameIndex:i32<0>, TargetConstant:i32<0>
; MM: t{{[0-9]+}}: i{{[0-9]+}} = LEA_ADDiu_MM TargetFrameIndex:i32<0>, TargetConstant:i32<0>
; MIPS64: t{{[0-9]+}}: i{{[0-9]+}} = LEA_ADDiu64 TargetFrameIndex:i64<0>, TargetConstant:i64<0>

define i32 @k() {
entry:
  %h = alloca i32, align 4
  %call = call i32 @g(ptr %h)
  ret i32 %call
}

declare i32 @g(ptr)

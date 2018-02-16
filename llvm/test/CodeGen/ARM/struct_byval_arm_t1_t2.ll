;RUN: llc < %s -mtriple=armv7-none-linux-gnueabi   -mattr=+neon -verify-machineinstrs -filetype=obj | llvm-objdump -triple armv7-none-linux-gnueabi   -disassemble - | FileCheck %s --check-prefix=ARM
;RUN: llc < %s -mtriple=thumbv7-none-linux-gnueabi -mattr=+neon -verify-machineinstrs -filetype=obj | llvm-objdump -triple thumbv7-none-linux-gnueabi -disassemble - | FileCheck %s --check-prefix=THUMB2
;RUN: llc < %s -mtriple=armv7-none-linux-gnueabi   -mattr=-neon -verify-machineinstrs -filetype=obj | llvm-objdump -triple armv7-none-linux-gnueabi   -disassemble - | FileCheck %s --check-prefix=NO_NEON
;We want to have both positive and negative checks for thumb1. These checks
;are not easy to do in a single pass so we generate the output once to a
;temp file and run filecheck twice with different prefixes.
;RUN: llc < %s -mtriple=thumbv5-none-linux-gnueabi              -verify-machineinstrs -filetype=obj | llvm-objdump -triple thumbv5-none-linux-gnueabi -disassemble - > %t
;RUN: cat %t | FileCheck %s --check-prefix=THUMB1
;RUN: cat %t | FileCheck %s --check-prefix=T1POST
;RUN: llc < %s -mtriple=thumbv8m.base-arm-none-eabi             -verify-machineinstrs -filetype=obj | llvm-objdump -triple thumbv8m.base-arm-none-eabi -disassemble - > %t
;RUN: cat %t | FileCheck %s --check-prefix=THUMB1
;RUN: cat %t | FileCheck %s --check-prefix=T1POST
;RUN: cat %t | FileCheck %s --check-prefix=V8MBASE

;This file contains auto generated tests for the lowering of passing structs
;byval in the arm backend. We have tests for both packed and unpacked
;structs at varying alignments. Each test is run for arm, thumb2 and thumb1.
;We check for the strings in the generated object code using llvm-objdump
;because it provides better assurance that we are generating instructions
;for the correct architecture. Otherwise we could accidentally generate an
;ARM instruction for THUMB1 and wouldn't detect it because the assembly
;code representation is the same, but the object code would be generated
;incorrectly. For each test we check for the label, a load instruction of the
;correct form, a branch if it will be generated with a loop, and the leftover
;cleanup if the number of bytes does not divide evenly by the store size

%struct.A = type <{ [ 10 x i32 ] }> ; 40 bytes
declare void @use_A(%struct.A* byval)
%struct.B = type <{ [ 10 x i32 ], i8 }> ; 41 bytes
declare void @use_B(%struct.B* byval)
%struct.C = type <{ [ 10 x i32 ], [ 3 x i8 ] }> ; 43 bytes
declare void @use_C(%struct.C* byval)
%struct.D = type <{ [ 100 x i32 ] }> ; 400 bytes
declare void @use_D(%struct.D* byval)
%struct.E = type <{ [ 100 x i32 ], i8 }> ; 401 bytes
declare void @use_E(%struct.E* byval)
%struct.F = type <{ [ 100 x i32 ], [ 3 x i8 ] }> ; 403 bytes
declare void @use_F(%struct.F* byval)
%struct.G = type  { [ 10 x i32 ] }  ; 40 bytes
declare void @use_G(%struct.G* byval)
%struct.H = type  { [ 10 x i32 ], i8 }  ; 41 bytes
declare void @use_H(%struct.H* byval)
%struct.I = type  { [ 10 x i32 ], [ 3 x i8 ] }  ; 43 bytes
declare void @use_I(%struct.I* byval)
%struct.J = type  { [ 100 x i32 ] }  ; 400 bytes
declare void @use_J(%struct.J* byval)
%struct.K = type  { [ 100 x i32 ], i8 }  ; 401 bytes
declare void @use_K(%struct.K* byval)
%struct.L = type  { [ 100 x i32 ], [ 3 x i8 ] }  ; 403 bytes
declare void @use_L(%struct.L* byval)
%struct.M = type  { [  64 x i8 ] }   ; 64 bytes
declare void @use_M(%struct.M* byval)
%struct.N = type  { [ 128 x i8 ] }  ; 128 bytes
declare void @use_N(%struct.N* byval)

;ARM-LABEL:    test_A_1:
;THUMB2-LABEL: test_A_1:
;NO_NEON-LABEL:test_A_1:
;THUMB1-LABEL: test_A_1:
;T1POST-LABEL: test_A_1:
  define void @test_A_1() {
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1

;T1POST-NOT:  ldrb    r{{[0-9]+}}, [{{.*}}], #1
  entry:
    %a = alloca %struct.A, align 1
    call void @use_A(%struct.A* byval align 1 %a)
    ret void
  }
;ARM-LABEL:    test_A_2:
;THUMB2-LABEL: test_A_2:
;NO_NEON-LABEL:test_A_2:
;THUMB1-LABEL: test_A_2:
;T1POST-LABEL: test_A_2:
  define void @test_A_2() {
;ARM:         ldrh    r{{[0-9]+}}, [{{.*}}], #2

;THUMB2:      ldrh    r{{[0-9]+}}, [{{.*}}], #2

;NO_NEON:     ldrh    r{{[0-9]+}}, [{{.*}}], #2

;THUMB1:      ldrh    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #2

;T1POST-NOT:  ldrh    r{{[0-9]+}}, [{{.*}}], #2
  entry:
    %a = alloca %struct.A, align 2
    call void @use_A(%struct.A* byval align 2 %a)
    ret void
  }
;ARM-LABEL:    test_A_4:
;THUMB2-LABEL: test_A_4:
;NO_NEON-LABEL:test_A_4:
;THUMB1-LABEL: test_A_4:
;T1POST-LABEL: test_A_4:
  define void @test_A_4() {
;ARM:         ldr     r{{[0-9]+}}, [{{.*}}], #4

;THUMB2:      ldr     r{{[0-9]+}}, [{{.*}}], #4

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4

;T1POST-NOT:  ldr     r{{[0-9]+}}, [{{.*}}], #4
  entry:
    %a = alloca %struct.A, align 4
    call void @use_A(%struct.A* byval align 4 %a)
    ret void
  }
;ARM-LABEL:    test_A_8:
;THUMB2-LABEL: test_A_8:
;NO_NEON-LABEL:test_A_8:
;THUMB1-LABEL: test_A_8:
;T1POST-LABEL: test_A_8:
  define void @test_A_8() {
;ARM:         vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;THUMB2:      vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.A, align 8
    call void @use_A(%struct.A* byval align 8 %a)
    ret void
  }
;ARM-LABEL:    test_A_16:
;THUMB2-LABEL: test_A_16:
;NO_NEON-LABEL:test_A_16:
;THUMB1-LABEL: test_A_16:
;T1POST-LABEL: test_A_16:
  define void @test_A_16() {
;ARM:         vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.A, align 16
    call void @use_A(%struct.A* byval align 16 %a)
    ret void
  }
;ARM-LABEL:    test_B_1:
;THUMB2-LABEL: test_B_1:
;NO_NEON-LABEL:test_B_1:
;THUMB1-LABEL: test_B_1:
;T1POST-LABEL: test_B_1:
  define void @test_B_1() {
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1

;T1POST-NOT:  ldrb    r{{[0-9]+}}, [{{.*}}], #1
  entry:
    %a = alloca %struct.B, align 1
    call void @use_B(%struct.B* byval align 1 %a)
    ret void
  }
;ARM-LABEL:    test_B_2:
;THUMB2-LABEL: test_B_2:
;NO_NEON-LABEL:test_B_2:
;THUMB1-LABEL: test_B_2:
;T1POST-LABEL: test_B_2:
  define void @test_B_2() {
;ARM:         ldrh    r{{[0-9]+}}, [{{.*}}], #2
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      ldrh    r{{[0-9]+}}, [{{.*}}], #2
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldrh    r{{[0-9]+}}, [{{.*}}], #2
;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB1:      ldrh    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #2
;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}

;T1POST-NOT:  ldrh    r{{[0-9]+}}, [{{.*}}], #2
  entry:
    %a = alloca %struct.B, align 2
    call void @use_B(%struct.B* byval align 2 %a)
    ret void
  }
;ARM-LABEL:    test_B_4:
;THUMB2-LABEL: test_B_4:
;NO_NEON-LABEL:test_B_4:
;THUMB1-LABEL: test_B_4:
;T1POST-LABEL: test_B_4:
  define void @test_B_4() {
;ARM:         ldr     r{{[0-9]+}}, [{{.*}}], #4
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      ldr     r{{[0-9]+}}, [{{.*}}], #4
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}

;T1POST-NOT:  ldr     r{{[0-9]+}}, [{{.*}}], #4
  entry:
    %a = alloca %struct.B, align 4
    call void @use_B(%struct.B* byval align 4 %a)
    ret void
  }
;ARM-LABEL:    test_B_8:
;THUMB2-LABEL: test_B_8:
;NO_NEON-LABEL:test_B_8:
;THUMB1-LABEL: test_B_8:
;T1POST-LABEL: test_B_8:
  define void @test_B_8() {
;ARM:         vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.B, align 8
    call void @use_B(%struct.B* byval align 8 %a)
    ret void
  }
;ARM-LABEL:    test_B_16:
;THUMB2-LABEL: test_B_16:
;NO_NEON-LABEL:test_B_16:
;THUMB1-LABEL: test_B_16:
;T1POST-LABEL: test_B_16:
  define void @test_B_16() {
;ARM:         vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.B, align 16
    call void @use_B(%struct.B* byval align 16 %a)
    ret void
  }
;ARM-LABEL:    test_C_1:
;THUMB2-LABEL: test_C_1:
;NO_NEON-LABEL:test_C_1:
;THUMB1-LABEL: test_C_1:
;T1POST-LABEL: test_C_1:
  define void @test_C_1() {
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1

;T1POST-NOT:  ldrb    r{{[0-9]+}}, [{{.*}}], #1
  entry:
    %a = alloca %struct.C, align 1
    call void @use_C(%struct.C* byval align 1 %a)
    ret void
  }
;ARM-LABEL:    test_C_2:
;THUMB2-LABEL: test_C_2:
;NO_NEON-LABEL:test_C_2:
;THUMB1-LABEL: test_C_2:
;T1POST-LABEL: test_C_2:
  define void @test_C_2() {
;ARM:         ldrh    r{{[0-9]+}}, [{{.*}}], #2
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      ldrh    r{{[0-9]+}}, [{{.*}}], #2
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldrh    r{{[0-9]+}}, [{{.*}}], #2
;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB1:      ldrh    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #2
;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}

;T1POST-NOT:  ldrh    r{{[0-9]+}}, [{{.*}}], #2
  entry:
    %a = alloca %struct.C, align 2
    call void @use_C(%struct.C* byval align 2 %a)
    ret void
  }
;ARM-LABEL:    test_C_4:
;THUMB2-LABEL: test_C_4:
;NO_NEON-LABEL:test_C_4:
;THUMB1-LABEL: test_C_4:
;T1POST-LABEL: test_C_4:
  define void @test_C_4() {
;ARM:         ldr     r{{[0-9]+}}, [{{.*}}], #4
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      ldr     r{{[0-9]+}}, [{{.*}}], #4
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1

;T1POST-NOT:  ldr     r{{[0-9]+}}, [{{.*}}], #4
  entry:
    %a = alloca %struct.C, align 4
    call void @use_C(%struct.C* byval align 4 %a)
    ret void
  }
;ARM-LABEL:    test_C_8:
;THUMB2-LABEL: test_C_8:
;NO_NEON-LABEL:test_C_8:
;THUMB1-LABEL: test_C_8:
;T1POST-LABEL: test_C_8:
  define void @test_C_8() {
;ARM:         vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.C, align 8
    call void @use_C(%struct.C* byval align 8 %a)
    ret void
  }
;ARM-LABEL:    test_C_16:
;THUMB2-LABEL: test_C_16:
;NO_NEON-LABEL:test_C_16:
;THUMB1-LABEL: test_C_16:
;T1POST-LABEL: test_C_16:
  define void @test_C_16() {
;ARM:         vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.C, align 16
    call void @use_C(%struct.C* byval align 16 %a)
    ret void
  }
;ARM-LABEL:    test_D_1:
;THUMB2-LABEL: test_D_1:
;NO_NEON-LABEL:test_D_1:
;THUMB1-LABEL: test_D_1:
;T1POST-LABEL: test_D_1:
  define void @test_D_1() {
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1
;ARM:         bne

;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1
;THUMB2:      bne

;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1
;NO_NEON:     bne

;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1
;THUMB1:      bne

;T1POST-NOT:  ldrb    r{{[0-9]+}}, [{{.*}}], #1
  entry:
    %a = alloca %struct.D, align 1
    call void @use_D(%struct.D* byval align 1 %a)
    ret void
  }
;ARM-LABEL:    test_D_2:
;THUMB2-LABEL: test_D_2:
;NO_NEON-LABEL:test_D_2:
;THUMB1-LABEL: test_D_2:
;T1POST-LABEL: test_D_2:
  define void @test_D_2() {
;ARM:         ldrh    r{{[0-9]+}}, [{{.*}}], #2
;ARM:         bne

;THUMB2:      ldrh    r{{[0-9]+}}, [{{.*}}], #2
;THUMB2:      bne

;NO_NEON:     ldrh    r{{[0-9]+}}, [{{.*}}], #2
;NO_NEON:     bne

;THUMB1:      ldrh    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #2
;THUMB1:      bne

;T1POST-NOT:  ldrh    r{{[0-9]+}}, [{{.*}}], #2
  entry:
    %a = alloca %struct.D, align 2
    call void @use_D(%struct.D* byval align 2 %a)
    ret void
  }
;ARM-LABEL:    test_D_4:
;THUMB2-LABEL: test_D_4:
;NO_NEON-LABEL:test_D_4:
;THUMB1-LABEL: test_D_4:
;T1POST-LABEL: test_D_4:
  define void @test_D_4() {
;ARM:         ldr     r{{[0-9]+}}, [{{.*}}], #4
;ARM:         bne

;THUMB2:      ldr     r{{[0-9]+}}, [{{.*}}], #4
;THUMB2:      bne

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne

;T1POST-NOT:  ldr     r{{[0-9]+}}, [{{.*}}], #4
  entry:
    %a = alloca %struct.D, align 4
    call void @use_D(%struct.D* byval align 4 %a)
    ret void
  }
;ARM-LABEL:    test_D_8:
;THUMB2-LABEL: test_D_8:
;NO_NEON-LABEL:test_D_8:
;THUMB1-LABEL: test_D_8:
;T1POST-LABEL: test_D_8:
  define void @test_D_8() {
;ARM:         vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
;ARM:         bne

;THUMB2:      vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      bne

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.D, align 8
    call void @use_D(%struct.D* byval align 8 %a)
    ret void
  }
;ARM-LABEL:    test_D_16:
;THUMB2-LABEL: test_D_16:
;NO_NEON-LABEL:test_D_16:
;THUMB1-LABEL: test_D_16:
;T1POST-LABEL: test_D_16:
  define void @test_D_16() {
;ARM:         vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;ARM:         bne

;THUMB2:      vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      bne

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.D, align 16
    call void @use_D(%struct.D* byval align 16 %a)
    ret void
  }
;ARM-LABEL:    test_E_1:
;THUMB2-LABEL: test_E_1:
;NO_NEON-LABEL:test_E_1:
;THUMB1-LABEL: test_E_1:
;T1POST-LABEL: test_E_1:
  define void @test_E_1() {
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1
;ARM:         bne

;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1
;THUMB2:      bne

;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1
;NO_NEON:     bne

;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1
;THUMB1:      bne

;T1POST-NOT:  ldrb    r{{[0-9]+}}, [{{.*}}], #1
  entry:
    %a = alloca %struct.E, align 1
    call void @use_E(%struct.E* byval align 1 %a)
    ret void
  }
;ARM-LABEL:    test_E_2:
;THUMB2-LABEL: test_E_2:
;NO_NEON-LABEL:test_E_2:
;THUMB1-LABEL: test_E_2:
;T1POST-LABEL: test_E_2:
  define void @test_E_2() {
;ARM:         ldrh    r{{[0-9]+}}, [{{.*}}], #2
;ARM:         bne
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      ldrh    r{{[0-9]+}}, [{{.*}}], #2
;THUMB2:      bne
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldrh    r{{[0-9]+}}, [{{.*}}], #2
;NO_NEON:     bne
;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB1:      ldrh    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #2
;THUMB1:      bne
;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}

;T1POST-NOT:  ldrh    r{{[0-9]+}}, [{{.*}}], #2
  entry:
    %a = alloca %struct.E, align 2
    call void @use_E(%struct.E* byval align 2 %a)
    ret void
  }
;ARM-LABEL:    test_E_4:
;THUMB2-LABEL: test_E_4:
;NO_NEON-LABEL:test_E_4:
;THUMB1-LABEL: test_E_4:
;T1POST-LABEL: test_E_4:
  define void @test_E_4() {
;ARM:         ldr     r{{[0-9]+}}, [{{.*}}], #4
;ARM:         bne
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      ldr     r{{[0-9]+}}, [{{.*}}], #4
;THUMB2:      bne
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne
;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne
;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}

;T1POST-NOT:  ldr     r{{[0-9]+}}, [{{.*}}], #4
  entry:
    %a = alloca %struct.E, align 4
    call void @use_E(%struct.E* byval align 4 %a)
    ret void
  }
;ARM-LABEL:    test_E_8:
;THUMB2-LABEL: test_E_8:
;NO_NEON-LABEL:test_E_8:
;THUMB1-LABEL: test_E_8:
;T1POST-LABEL: test_E_8:
  define void @test_E_8() {
;ARM:         vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
;ARM:         bne
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      bne
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne
;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne
;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.E, align 8
    call void @use_E(%struct.E* byval align 8 %a)
    ret void
  }
;ARM-LABEL:    test_E_16:
;THUMB2-LABEL: test_E_16:
;NO_NEON-LABEL:test_E_16:
;THUMB1-LABEL: test_E_16:
;T1POST-LABEL: test_E_16:
  define void @test_E_16() {
;ARM:         vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;ARM:         bne
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      bne
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne
;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne
;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.E, align 16
    call void @use_E(%struct.E* byval align 16 %a)
    ret void
  }
;ARM-LABEL:    test_F_1:
;THUMB2-LABEL: test_F_1:
;NO_NEON-LABEL:test_F_1:
;THUMB1-LABEL: test_F_1:
;T1POST-LABEL: test_F_1:
  define void @test_F_1() {
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1
;ARM:         bne

;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1
;THUMB2:      bne

;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1
;NO_NEON:     bne

;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1
;THUMB1:      bne

;T1POST-NOT:  ldrb    r{{[0-9]+}}, [{{.*}}], #1
  entry:
    %a = alloca %struct.F, align 1
    call void @use_F(%struct.F* byval align 1 %a)
    ret void
  }
;ARM-LABEL:    test_F_2:
;THUMB2-LABEL: test_F_2:
;NO_NEON-LABEL:test_F_2:
;THUMB1-LABEL: test_F_2:
;T1POST-LABEL: test_F_2:
  define void @test_F_2() {
;ARM:         ldrh    r{{[0-9]+}}, [{{.*}}], #2
;ARM:         bne
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      ldrh    r{{[0-9]+}}, [{{.*}}], #2
;THUMB2:      bne
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldrh    r{{[0-9]+}}, [{{.*}}], #2
;NO_NEON:     bne
;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB1:      ldrh    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #2
;THUMB1:      bne
;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}

;T1POST-NOT:  ldrh    r{{[0-9]+}}, [{{.*}}], #2
  entry:
    %a = alloca %struct.F, align 2
    call void @use_F(%struct.F* byval align 2 %a)
    ret void
  }
;ARM-LABEL:    test_F_4:
;THUMB2-LABEL: test_F_4:
;NO_NEON-LABEL:test_F_4:
;THUMB1-LABEL: test_F_4:
;T1POST-LABEL: test_F_4:
  define void @test_F_4() {
;ARM:         ldr     r{{[0-9]+}}, [{{.*}}], #4
;ARM:         bne
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      ldr     r{{[0-9]+}}, [{{.*}}], #4
;THUMB2:      bne
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne
;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne
;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1

;T1POST-NOT:  ldr     r{{[0-9]+}}, [{{.*}}], #4
  entry:
    %a = alloca %struct.F, align 4
    call void @use_F(%struct.F* byval align 4 %a)
    ret void
  }
;ARM-LABEL:    test_F_8:
;THUMB2-LABEL: test_F_8:
;NO_NEON-LABEL:test_F_8:
;THUMB1-LABEL: test_F_8:
;T1POST-LABEL: test_F_8:
  define void @test_F_8() {
;ARM:         vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
;ARM:         bne
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      bne
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne
;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne
;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.F, align 8
    call void @use_F(%struct.F* byval align 8 %a)
    ret void
  }
;ARM-LABEL:    test_F_16:
;THUMB2-LABEL: test_F_16:
;NO_NEON-LABEL:test_F_16:
;THUMB1-LABEL: test_F_16:
;T1POST-LABEL: test_F_16:
  define void @test_F_16() {
;ARM:         vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;ARM:         bne
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      bne
;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne
;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne
;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.F, align 16
    call void @use_F(%struct.F* byval align 16 %a)
    ret void
  }
;ARM-LABEL:    test_G_1:
;THUMB2-LABEL: test_G_1:
;NO_NEON-LABEL:test_G_1:
;THUMB1-LABEL: test_G_1:
;T1POST-LABEL: test_G_1:
  define void @test_G_1() {
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1

;T1POST-NOT:  ldrb    r{{[0-9]+}}, [{{.*}}], #1
  entry:
    %a = alloca %struct.G, align 1
    call void @use_G(%struct.G* byval align 1 %a)
    ret void
  }
;ARM-LABEL:    test_G_2:
;THUMB2-LABEL: test_G_2:
;NO_NEON-LABEL:test_G_2:
;THUMB1-LABEL: test_G_2:
;T1POST-LABEL: test_G_2:
  define void @test_G_2() {
;ARM:         ldrh    r{{[0-9]+}}, [{{.*}}], #2

;THUMB2:      ldrh    r{{[0-9]+}}, [{{.*}}], #2

;NO_NEON:     ldrh    r{{[0-9]+}}, [{{.*}}], #2

;THUMB1:      ldrh    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #2

;T1POST-NOT:  ldrh    r{{[0-9]+}}, [{{.*}}], #2
  entry:
    %a = alloca %struct.G, align 2
    call void @use_G(%struct.G* byval align 2 %a)
    ret void
  }
;ARM-LABEL:    test_G_4:
;THUMB2-LABEL: test_G_4:
;NO_NEON-LABEL:test_G_4:
;THUMB1-LABEL: test_G_4:
;T1POST-LABEL: test_G_4:
  define void @test_G_4() {
;ARM:         ldr     r{{[0-9]+}}, [{{.*}}], #4

;THUMB2:      ldr     r{{[0-9]+}}, [{{.*}}], #4

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4

;T1POST-NOT:  ldr     r{{[0-9]+}}, [{{.*}}], #4
  entry:
    %a = alloca %struct.G, align 4
    call void @use_G(%struct.G* byval align 4 %a)
    ret void
  }
;ARM-LABEL:    test_G_8:
;THUMB2-LABEL: test_G_8:
;NO_NEON-LABEL:test_G_8:
;THUMB1-LABEL: test_G_8:
;T1POST-LABEL: test_G_8:
  define void @test_G_8() {
;ARM:         vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;THUMB2:      vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.G, align 8
    call void @use_G(%struct.G* byval align 8 %a)
    ret void
  }
;ARM-LABEL:    test_G_16:
;THUMB2-LABEL: test_G_16:
;NO_NEON-LABEL:test_G_16:
;THUMB1-LABEL: test_G_16:
;T1POST-LABEL: test_G_16:
  define void @test_G_16() {
;ARM:         vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;THUMB2:      vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.G, align 16
    call void @use_G(%struct.G* byval align 16 %a)
    ret void
  }
;ARM-LABEL:    test_H_1:
;THUMB2-LABEL: test_H_1:
;NO_NEON-LABEL:test_H_1:
;THUMB1-LABEL: test_H_1:
;T1POST-LABEL: test_H_1:
  define void @test_H_1() {
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1

;T1POST-NOT:  ldrb    r{{[0-9]+}}, [{{.*}}], #1
  entry:
    %a = alloca %struct.H, align 1
    call void @use_H(%struct.H* byval align 1 %a)
    ret void
  }
;ARM-LABEL:    test_H_2:
;THUMB2-LABEL: test_H_2:
;NO_NEON-LABEL:test_H_2:
;THUMB1-LABEL: test_H_2:
;T1POST-LABEL: test_H_2:
  define void @test_H_2() {
;ARM:         ldrh    r{{[0-9]+}}, [{{.*}}], #2

;THUMB2:      ldrh    r{{[0-9]+}}, [{{.*}}], #2

;NO_NEON:     ldrh    r{{[0-9]+}}, [{{.*}}], #2

;THUMB1:      ldrh    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #2

;T1POST-NOT:  ldrh    r{{[0-9]+}}, [{{.*}}], #2
  entry:
    %a = alloca %struct.H, align 2
    call void @use_H(%struct.H* byval align 2 %a)
    ret void
  }
;ARM-LABEL:    test_H_4:
;THUMB2-LABEL: test_H_4:
;NO_NEON-LABEL:test_H_4:
;THUMB1-LABEL: test_H_4:
;T1POST-LABEL: test_H_4:
  define void @test_H_4() {
;ARM:         ldr     r{{[0-9]+}}, [{{.*}}], #4

;THUMB2:      ldr     r{{[0-9]+}}, [{{.*}}], #4

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4

;T1POST-NOT:  ldr     r{{[0-9]+}}, [{{.*}}], #4
  entry:
    %a = alloca %struct.H, align 4
    call void @use_H(%struct.H* byval align 4 %a)
    ret void
  }
;ARM-LABEL:    test_H_8:
;THUMB2-LABEL: test_H_8:
;NO_NEON-LABEL:test_H_8:
;THUMB1-LABEL: test_H_8:
;T1POST-LABEL: test_H_8:
  define void @test_H_8() {
;ARM:         vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;THUMB2:      vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.H, align 8
    call void @use_H(%struct.H* byval align 8 %a)
    ret void
  }
;ARM-LABEL:    test_H_16:
;THUMB2-LABEL: test_H_16:
;NO_NEON-LABEL:test_H_16:
;THUMB1-LABEL: test_H_16:
;T1POST-LABEL: test_H_16:
  define void @test_H_16() {
;ARM:         vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;THUMB2:      vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.H, align 16
    call void @use_H(%struct.H* byval align 16 %a)
    ret void
  }
;ARM-LABEL:    test_I_1:
;THUMB2-LABEL: test_I_1:
;NO_NEON-LABEL:test_I_1:
;THUMB1-LABEL: test_I_1:
;T1POST-LABEL: test_I_1:
  define void @test_I_1() {
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1

;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1

;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1

;T1POST-NOT:  ldrb    r{{[0-9]+}}, [{{.*}}], #1
  entry:
    %a = alloca %struct.I, align 1
    call void @use_I(%struct.I* byval align 1 %a)
    ret void
  }
;ARM-LABEL:    test_I_2:
;THUMB2-LABEL: test_I_2:
;NO_NEON-LABEL:test_I_2:
;THUMB1-LABEL: test_I_2:
;T1POST-LABEL: test_I_2:
  define void @test_I_2() {
;ARM:         ldrh    r{{[0-9]+}}, [{{.*}}], #2

;THUMB2:      ldrh    r{{[0-9]+}}, [{{.*}}], #2

;NO_NEON:     ldrh    r{{[0-9]+}}, [{{.*}}], #2

;THUMB1:      ldrh    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #2

;T1POST-NOT:  ldrh    r{{[0-9]+}}, [{{.*}}], #2
  entry:
    %a = alloca %struct.I, align 2
    call void @use_I(%struct.I* byval align 2 %a)
    ret void
  }
;ARM-LABEL:    test_I_4:
;THUMB2-LABEL: test_I_4:
;NO_NEON-LABEL:test_I_4:
;THUMB1-LABEL: test_I_4:
;T1POST-LABEL: test_I_4:
  define void @test_I_4() {
;ARM:         ldr     r{{[0-9]+}}, [{{.*}}], #4

;THUMB2:      ldr     r{{[0-9]+}}, [{{.*}}], #4

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4

;T1POST-NOT:  ldr     r{{[0-9]+}}, [{{.*}}], #4
  entry:
    %a = alloca %struct.I, align 4
    call void @use_I(%struct.I* byval align 4 %a)
    ret void
  }
;ARM-LABEL:    test_I_8:
;THUMB2-LABEL: test_I_8:
;NO_NEON-LABEL:test_I_8:
;THUMB1-LABEL: test_I_8:
;T1POST-LABEL: test_I_8:
  define void @test_I_8() {
;ARM:         vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;THUMB2:      vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.I, align 8
    call void @use_I(%struct.I* byval align 8 %a)
    ret void
  }
;ARM-LABEL:    test_I_16:
;THUMB2-LABEL: test_I_16:
;NO_NEON-LABEL:test_I_16:
;THUMB1-LABEL: test_I_16:
;T1POST-LABEL: test_I_16:
  define void @test_I_16() {
;ARM:         vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;THUMB2:      vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.I, align 16
    call void @use_I(%struct.I* byval align 16 %a)
    ret void
  }
;ARM-LABEL:    test_J_1:
;THUMB2-LABEL: test_J_1:
;NO_NEON-LABEL:test_J_1:
;THUMB1-LABEL: test_J_1:
;T1POST-LABEL: test_J_1:
  define void @test_J_1() {
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1
;ARM:         bne

;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1
;THUMB2:      bne

;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1
;NO_NEON:     bne

;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1
;THUMB1:      bne

;T1POST-NOT:  ldrb    r{{[0-9]+}}, [{{.*}}], #1
  entry:
    %a = alloca %struct.J, align 1
    call void @use_J(%struct.J* byval align 1 %a)
    ret void
  }
;ARM-LABEL:    test_J_2:
;THUMB2-LABEL: test_J_2:
;NO_NEON-LABEL:test_J_2:
;THUMB1-LABEL: test_J_2:
;T1POST-LABEL: test_J_2:
  define void @test_J_2() {
;ARM:         ldrh    r{{[0-9]+}}, [{{.*}}], #2
;ARM:         bne

;THUMB2:      ldrh    r{{[0-9]+}}, [{{.*}}], #2
;THUMB2:      bne

;NO_NEON:     ldrh    r{{[0-9]+}}, [{{.*}}], #2
;NO_NEON:     bne

;THUMB1:      ldrh    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #2
;THUMB1:      bne

;T1POST-NOT:  ldrh    r{{[0-9]+}}, [{{.*}}], #2
  entry:
    %a = alloca %struct.J, align 2
    call void @use_J(%struct.J* byval align 2 %a)
    ret void
  }
;ARM-LABEL:    test_J_4:
;THUMB2-LABEL: test_J_4:
;NO_NEON-LABEL:test_J_4:
;THUMB1-LABEL: test_J_4:
;T1POST-LABEL: test_J_4:
  define void @test_J_4() {
;ARM:         ldr     r{{[0-9]+}}, [{{.*}}], #4
;ARM:         bne

;THUMB2:      ldr     r{{[0-9]+}}, [{{.*}}], #4
;THUMB2:      bne

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne

;T1POST-NOT:  ldr     r{{[0-9]+}}, [{{.*}}], #4
  entry:
    %a = alloca %struct.J, align 4
    call void @use_J(%struct.J* byval align 4 %a)
    ret void
  }
;ARM-LABEL:    test_J_8:
;THUMB2-LABEL: test_J_8:
;NO_NEON-LABEL:test_J_8:
;THUMB1-LABEL: test_J_8:
;T1POST-LABEL: test_J_8:
  define void @test_J_8() {
;ARM:         vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
;ARM:         bne

;THUMB2:      vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      bne

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.J, align 8
    call void @use_J(%struct.J* byval align 8 %a)
    ret void
  }
;ARM-LABEL:    test_J_16:
;THUMB2-LABEL: test_J_16:
;NO_NEON-LABEL:test_J_16:
;THUMB1-LABEL: test_J_16:
;T1POST-LABEL: test_J_16:
  define void @test_J_16() {
;ARM:         vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;ARM:         bne

;THUMB2:      vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      bne

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.J, align 16
    call void @use_J(%struct.J* byval align 16 %a)
    ret void
  }
;ARM-LABEL:    test_K_1:
;THUMB2-LABEL: test_K_1:
;NO_NEON-LABEL:test_K_1:
;THUMB1-LABEL: test_K_1:
;T1POST-LABEL: test_K_1:
  define void @test_K_1() {
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1
;ARM:         bne

;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1
;THUMB2:      bne

;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1
;NO_NEON:     bne

;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1
;THUMB1:      bne

;T1POST-NOT:  ldrb    r{{[0-9]+}}, [{{.*}}], #1
  entry:
    %a = alloca %struct.K, align 1
    call void @use_K(%struct.K* byval align 1 %a)
    ret void
  }
;ARM-LABEL:    test_K_2:
;THUMB2-LABEL: test_K_2:
;NO_NEON-LABEL:test_K_2:
;THUMB1-LABEL: test_K_2:
;T1POST-LABEL: test_K_2:
  define void @test_K_2() {
;ARM:         ldrh    r{{[0-9]+}}, [{{.*}}], #2
;ARM:         bne

;THUMB2:      ldrh    r{{[0-9]+}}, [{{.*}}], #2
;THUMB2:      bne

;NO_NEON:     ldrh    r{{[0-9]+}}, [{{.*}}], #2
;NO_NEON:     bne

;THUMB1:      ldrh    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #2
;THUMB1:      bne

;T1POST-NOT:  ldrh    r{{[0-9]+}}, [{{.*}}], #2
  entry:
    %a = alloca %struct.K, align 2
    call void @use_K(%struct.K* byval align 2 %a)
    ret void
  }
;ARM-LABEL:    test_K_4:
;THUMB2-LABEL: test_K_4:
;NO_NEON-LABEL:test_K_4:
;THUMB1-LABEL: test_K_4:
;T1POST-LABEL: test_K_4:
  define void @test_K_4() {
;ARM:         ldr     r{{[0-9]+}}, [{{.*}}], #4
;ARM:         bne

;THUMB2:      ldr     r{{[0-9]+}}, [{{.*}}], #4
;THUMB2:      bne

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne

;T1POST-NOT:  ldr     r{{[0-9]+}}, [{{.*}}], #4
  entry:
    %a = alloca %struct.K, align 4
    call void @use_K(%struct.K* byval align 4 %a)
    ret void
  }
;ARM-LABEL:    test_K_8:
;THUMB2-LABEL: test_K_8:
;NO_NEON-LABEL:test_K_8:
;THUMB1-LABEL: test_K_8:
;T1POST-LABEL: test_K_8:
  define void @test_K_8() {
;ARM:         vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
;ARM:         bne

;THUMB2:      vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      bne

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.K, align 8
    call void @use_K(%struct.K* byval align 8 %a)
    ret void
  }
;ARM-LABEL:    test_K_16:
;THUMB2-LABEL: test_K_16:
;NO_NEON-LABEL:test_K_16:
;THUMB1-LABEL: test_K_16:
;T1POST-LABEL: test_K_16:
  define void @test_K_16() {
;ARM:         vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;ARM:         bne

;THUMB2:      vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      bne

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.K, align 16
    call void @use_K(%struct.K* byval align 16 %a)
    ret void
  }
;ARM-LABEL:    test_L_1:
;THUMB2-LABEL: test_L_1:
;NO_NEON-LABEL:test_L_1:
;THUMB1-LABEL: test_L_1:
;T1POST-LABEL: test_L_1:
  define void @test_L_1() {
;ARM:         ldrb    r{{[0-9]+}}, [{{.*}}], #1
;ARM:         bne

;THUMB2:      ldrb    r{{[0-9]+}}, [{{.*}}], #1
;THUMB2:      bne

;NO_NEON:     ldrb    r{{[0-9]+}}, [{{.*}}], #1
;NO_NEON:     bne

;THUMB1:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #1
;THUMB1:      bne

;T1POST-NOT:  ldrb    r{{[0-9]+}}, [{{.*}}], #1
  entry:
    %a = alloca %struct.L, align 1
    call void @use_L(%struct.L* byval align 1 %a)
    ret void
  }
;ARM-LABEL:    test_L_2:
;THUMB2-LABEL: test_L_2:
;NO_NEON-LABEL:test_L_2:
;THUMB1-LABEL: test_L_2:
;T1POST-LABEL: test_L_2:
  define void @test_L_2() {
;ARM:         ldrh    r{{[0-9]+}}, [{{.*}}], #2
;ARM:         bne

;THUMB2:      ldrh    r{{[0-9]+}}, [{{.*}}], #2
;THUMB2:      bne

;NO_NEON:     ldrh    r{{[0-9]+}}, [{{.*}}], #2
;NO_NEON:     bne

;THUMB1:      ldrh    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #2
;THUMB1:      bne

;T1POST-NOT:  ldrh    r{{[0-9]+}}, [{{.*}}], #2
  entry:
    %a = alloca %struct.L, align 2
    call void @use_L(%struct.L* byval align 2 %a)
    ret void
  }
;ARM-LABEL:    test_L_4:
;THUMB2-LABEL: test_L_4:
;NO_NEON-LABEL:test_L_4:
;THUMB1-LABEL: test_L_4:
;T1POST-LABEL: test_L_4:
  define void @test_L_4() {
;ARM:         ldr     r{{[0-9]+}}, [{{.*}}], #4
;ARM:         bne

;THUMB2:      ldr     r{{[0-9]+}}, [{{.*}}], #4
;THUMB2:      bne

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne

;T1POST-NOT:  ldr     r{{[0-9]+}}, [{{.*}}], #4
  entry:
    %a = alloca %struct.L, align 4
    call void @use_L(%struct.L* byval align 4 %a)
    ret void
  }
;ARM-LABEL:    test_L_8:
;THUMB2-LABEL: test_L_8:
;NO_NEON-LABEL:test_L_8:
;THUMB1-LABEL: test_L_8:
;T1POST-LABEL: test_L_8:
  define void @test_L_8() {
;ARM:         vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
;ARM:         bne

;THUMB2:      vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      bne

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.L, align 8
    call void @use_L(%struct.L* byval align 8 %a)
    ret void
  }
;ARM-LABEL:    test_L_16:
;THUMB2-LABEL: test_L_16:
;NO_NEON-LABEL:test_L_16:
;THUMB1-LABEL: test_L_16:
;T1POST-LABEL: test_L_16:
  define void @test_L_16() {
;ARM:         vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;ARM:         bne

;THUMB2:      vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
;THUMB2:      bne

;NO_NEON:     ldr     r{{[0-9]+}}, [{{.*}}], #4
;NO_NEON:     bne
;NO_NEON-NOT: vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!

;THUMB1:      ldr     r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;THUMB1:      adds    [[BASE]], #4
;THUMB1:      bne

;T1POST-NOT:  vld1.32 {d{{[0-9]+}}, d{{[0-9]+}}}, [{{.*}}]!
  entry:
    %a = alloca %struct.L, align 16
    call void @use_L(%struct.L* byval align 16 %a)
    ret void
  }
;V8MBASE-LABEL: test_M:
  define void @test_M() {

;V8MBASE:      ldrb    r{{[0-9]+}}, {{\[}}[[BASE:r[0-9]+]]{{\]}}
;V8MBASE:      adds    [[BASE]], #1
;V8MBASE-NOT:  movw
  entry:
    %a = alloca %struct.M, align 1
    call void @use_M(%struct.M* byval align 1 %a)
    ret void
  }
;V8MBASE-LABEL: test_N:
  define void @test_N() {

;V8MBASE:      movw    r{{[0-9]+}}, #{{[0-9]+}}
;V8MBASE-NOT:  b       #{{[0-9]+}}
  entry:
    %a = alloca %struct.N, align 1
    call void @use_N(%struct.N* byval align 1 %a)
    ret void
  }

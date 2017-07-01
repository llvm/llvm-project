; RUN: llc < %s -mtriple=thumbv7m -mattr=+execute-only -O0 %s -o - \
; RUN:  | FileCheck --check-prefix=CHECK-SUBW-ADDW %s
; RUN: llc < %s -mtriple=thumbv8m.base -mattr=+execute-only -O0 %s -o - \
; RUN:  | FileCheck --check-prefix=CHECK-MOVW-MOVT-ADD %s
; RUN: llc < %s -mtriple=thumbv8m.main -mattr=+execute-only -O0 %s -o - \
; RUN:  | FileCheck --check-prefix=CHECK-SUBW-ADDW %s

define i8 @test_big_stack_frame() {
; CHECK-SUBW-ADDW-LABEL: test_big_stack_frame:
; CHECK-SUBW-ADDW-NOT:   ldr {{r[0-9]+}}, .{{.*}}
; CHECK-SUBW-ADDW:       sub.w sp, sp, #65536
; CHECK-SUBW-ADDW-NOT:   ldr {{r[0-9]+}}, .{{.*}}
; CHECK-SUBW-ADDW:       add.w [[REG1:r[0-9]+|lr]], sp, #255
; CHECK-SUBW-ADDW:       add.w {{r[0-9]+}}, [[REG1]], #65280
; CHECK-SUBW-ADDW-NOT:   ldr {{r[0-9]+}}, .{{.*}}
; CHECK-SUBW-ADDW:       add.w [[REGX:r[0-9]+|lr]], sp, #61440
; CHECK-SUBW-ADDW-NOT:   ldr {{r[0-9]+}}, .{{.*}}
; CHECK-SUBW-ADDW:       add.w sp, sp, #65536

; CHECK-MOVW-MOVT-ADD-LABEL: test_big_stack_frame:
; CHECK-MOVW-MOVT-ADD-NOT:   ldr {{r[0-9]+}}, .{{.*}}
; CHECK-MOVW-MOVT-ADD:       movw [[REG1:r[0-9]+]], #0
; CHECK-MOVW-MOVT-ADD:       movt [[REG1]], #65535
; CHECK-MOVW-MOVT-ADD:       add sp, [[REG1]]
; CHECK-MOVW-MOVT-ADD-NOT:   ldr {{r[0-9]+}}, .{{.*}}
; CHECK-MOVW-MOVT-ADD:       movw [[REG2:r[0-9]+]], #65532
; CHECK-MOVW-MOVT-ADD:       movt [[REG2]], #0
; CHECK-MOVW-MOVT-ADD:       add [[REG2]], sp
; CHECK-MOVW-MOVT-ADD-NOT:   ldr {{r[0-9]+}}, .{{.*}}
; CHECK-MOVW-MOVT-ADD:       movw [[REG3:r[0-9]+]], #65532
; CHECK-MOVW-MOVT-ADD:       movt [[REG3]], #0
; CHECK-MOVW-MOVT-ADD:       add [[REG3]], sp
; CHECK-MOVW-MOVT-ADD-NOT:   ldr {{r[0-9]+}}, .{{.*}}
; CHECK-MOVW-MOVT-ADD:       movw [[REG4:r[0-9]+]], #0
; CHECK-MOVW-MOVT-ADD:       movt [[REG4]], #1
; CHECK-MOVW-MOVT-ADD:       add sp, [[REG4]]

entry:
  %s1 = alloca i8
  %buffer = alloca [65528 x i8], align 1
  call void @foo(i8* %s1)
  %load = load i8, i8* %s1
  ret i8 %load
}

declare void @foo(i8*)

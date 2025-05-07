; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -filetype=obj -o - \
; RUN:  | llvm-objdump --no-print-imm-hex --triple=aarch64-unknown-linux-gnu --show-all-symbols -d - \
; RUN:  | FileCheck %s

%struct.c = type { ptr }

@l = common hidden local_unnamed_addr global i32 0, align 4

; CHECK-LABEL: <test1>:
; CHECK-LABEL: <$d>:
; CHECK-LABEL: <$x>:
; CHECK-NEXT:    b 0x2c <test1+0x2c>
; CHECK-LABEL: <$x>:
; CHECK-NEXT:    mov w0, wzr
; CHECK-NEXT:    ldr x30, [sp], #16
; CHECK-NEXT:    ret
define hidden i32 @test1() {
  %1 = tail call i32 @g()
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %3, label %5

3:                                                ; preds = %0
  callbr void asm sideeffect "1: nop\0A\09.quad a\0A\09b ${1:l}\0A\09.quad ${0:c}", "i,!i"(ptr null)
          to label %4 [label %7]

4:                                                ; preds = %3
  br label %7

5:                                                ; preds = %0
  %6 = tail call i32 @i()
  br label %7

7:                                                ; preds = %3, %4, %5
  %8 = phi i32 [ %6, %5 ], [ 0, %4 ], [ 0, %3 ]
  ret i32 %8
}

declare dso_local i32 @g(...) local_unnamed_addr

declare dso_local i32 @i(...) local_unnamed_addr

; CHECK-LABEL: <test2>:
; CHECK:         b {{.*}} <test2+0x1c>
; CHECK-LABEL: <$d>:
; CHECK-LABEL: <$x>:
; CHECK-NEXT:    b {{.*}} <test2+0x18>
define hidden i32 @test2() local_unnamed_addr {
  %1 = load i32, ptr @l, align 4
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %10, label %3

3:                                                ; preds = %0
  %4 = tail call i32 @g()
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %3
  callbr void asm sideeffect "1: nop\0A\09.quad b\0A\09b ${1:l}\0A\09.quad ${0:c}", "i,!i"(ptr null)
          to label %10 [label %7]

7:                                                ; preds = %3
  %8 = tail call i32 @i()
  br label %10

9:                                                ; preds = %6
  br label %10

10:                                               ; preds = %7, %0, %6, %9
  ret i32 undef
}

; CHECK-LABEL: <test3>:
; CHECK-LABEL: <$d>:
; CHECK-LABEL: <$x>:
; CHECK-NEXT:    b {{.*}} <test3+0x34>
; CHECK-LABEL: <$x>:
; CHECK-NEXT:    ldr x30, [sp], #16
; CHECK-NEXT:    ret
define internal i1 @test3() {
  %1 = tail call i32 @g()
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %3, label %5

3:                                                ; preds = %0
  callbr void asm sideeffect "1: nop\0A\09.quad c\0A\09b ${1:l}\0A\09.quad ${0:c}", "i,!i"(ptr null)
          to label %4 [label %8]

4:                                                ; preds = %3
  br label %8

5:                                                ; preds = %0
  %6 = tail call i32 @i()
  %7 = icmp ne i32 %6, 0
  br label %8

8:                                                ; preds = %3, %4, %5
  %9 = phi i1 [ %7, %5 ], [ false, %4 ], [ false, %3 ]
  ret i1 %9
}

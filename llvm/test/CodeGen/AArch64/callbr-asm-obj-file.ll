; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -filetype=obj -o - \
; RUN:  | llvm-objdump --triple=aarch64-unknown-linux-gnu -d - \
; RUN:  | FileCheck %s

%struct.c = type { i1 (...)* }

@l = common hidden local_unnamed_addr global i32 0, align 4

; CHECK-LABEL: <test1>:
; CHECK-LABEL: <$d.1>:
; CHECK-LABEL: <$x.2>:
; CHECK-NEXT:    b #16 <$x.4+0x4>
; CHECK-LABEL: <$x.4>:
; CHECK-NEXT:    b #4 <$x.4+0x4>
; CHECK-NEXT:    mov w0, wzr
; CHECK-NEXT:    ldr x30, [sp], #16
; CHECK-NEXT:    ret
define hidden i32 @test1() {
  %1 = tail call i32 bitcast (i32 (...)* @g to i32 ()*)()
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %3, label %5

3:                                                ; preds = %0
  callbr void asm sideeffect "1: nop\0A\09.quad a\0A\09b ${1:l}\0A\09.quad ${0:c}", "i,X"(i32* null, i8* blockaddress(@test1, %7))
          to label %4 [label %7]

4:                                                ; preds = %3
  br label %7

5:                                                ; preds = %0
  %6 = tail call i32 bitcast (i32 (...)* @i to i32 ()*)()
  br label %7

7:                                                ; preds = %3, %4, %5
  %8 = phi i32 [ %6, %5 ], [ 0, %4 ], [ 0, %3 ]
  ret i32 %8
}

declare dso_local i32 @g(...) local_unnamed_addr

declare dso_local i32 @i(...) local_unnamed_addr

; CHECK-LABEL: <test2>:
; CHECK:         bl #0 <test2+0x18>
; CHECK-LABEL: <$d.5>:
; CHECK-LABEL: <$x.6>:
; CHECK-NEXT:    b #-20 <test2+0x18>
define hidden i32 @test2() local_unnamed_addr {
  %1 = load i32, i32* @l, align 4
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %10, label %3

3:                                                ; preds = %0
  %4 = tail call i32 bitcast (i32 (...)* @g to i32 ()*)()
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %7

6:                                                ; preds = %3
  callbr void asm sideeffect "1: nop\0A\09.quad b\0A\09b ${1:l}\0A\09.quad ${0:c}", "i,X"(i32* null, i8* blockaddress(@test2, %7))
          to label %10 [label %7]

7:                                                ; preds = %3
  %8 = tail call i32 bitcast (i32 (...)* @i to i32 ()*)()
  br label %10

9:                                                ; preds = %6
  br label %10

10:                                               ; preds = %7, %0, %6, %9
  ret i32 undef
}

; CHECK-LABEL: <test3>:
; CHECK-LABEL: <$d.9>:
; CHECK-LABEL: <$x.10>:
; CHECK-NEXT:    b #-20 <test3+0x18>
; CHECK-LABEL: <$x.12>:
; CHECK-NEXT:    b #4 <$x.12+0x4>
; CHECK-NEXT:    mov w0, wzr
; CHECK-NEXT:    ldr x30, [sp], #16
; CHECK-NEXT:    ret
define internal i1 @test3() {
  %1 = tail call i32 bitcast (i32 (...)* @g to i32 ()*)()
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %3, label %5

3:                                                ; preds = %0
  callbr void asm sideeffect "1: nop\0A\09.quad c\0A\09b ${1:l}\0A\09.quad ${0:c}", "i,X"(i32* null, i8* blockaddress(@test3, %8))
          to label %4 [label %8]

4:                                                ; preds = %3
  br label %8

5:                                                ; preds = %0
  %6 = tail call i32 bitcast (i32 (...)* @i to i32 ()*)()
  %7 = icmp ne i32 %6, 0
  br label %8

8:                                                ; preds = %3, %4, %5
  %9 = phi i1 [ %7, %5 ], [ false, %4 ], [ false, %3 ]
  ret i1 %9
}

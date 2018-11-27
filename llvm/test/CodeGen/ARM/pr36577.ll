; RUN: llc -mtriple armv6t2 %s -o - | FileCheck %s
; RUN: llc -mtriple thumbv6t2 %s -o - | FileCheck %s --check-prefix=CHECK-T2
; RUN: llc -mtriple armv7 %s -o - | FileCheck %s
; RUN: llc -mtriple thumbv7 %s -o - | FileCheck %s --check-prefix=CHECK-T2
; RUN: llc -mtriple thumbv7m %s -o - | FileCheck %s --check-prefix=CHECK-T2
; RUN: llc -mtriple thumbv8m.main %s -o - | FileCheck %s --check-prefix=CHECK-T2

@a = common dso_local local_unnamed_addr global i16 0, align 2

; CHECK-LABEL: pr36577
; CHECK: ldrh r0, [r0]
; CHECK: mvn	r0, r0, lsr #7
; CHECK: orr r0, r1, r0, lsl #2
; CHECK-T2: ldrh r0, [r0]
; CHECK-T2: mvn.w	r0, r0, lsr #7
; CHECK-T2: orr.w	r0, r1, r0, lsl #2
define dso_local arm_aapcscc i32** @pr36577() {
entry:
  %0 = load i16, i16* @a, align 2
  %1 = lshr i16 %0, 7
  %2 = and i16 %1, 1
  %3 = zext i16 %2 to i32
  %4 = xor i32 %3, -1
  %add.ptr = getelementptr inbounds i32*, i32** null, i32 %4
  ret i32** %add.ptr
}


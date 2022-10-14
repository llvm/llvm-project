; RUN: llc < %s -mcpu=z196 -mtriple=s390x-linux-gnu -O0 \
; RUN:   -stop-before=regallocfast 2>&1 | FileCheck %s
; RUN: llc < %s -mcpu=z196 -mtriple=s390x-linux-gnu -O3 \
; RUN:   -stop-before=livevars 2>&1 | FileCheck %s
;
; Test that copies to/from access registers are handled before regalloc with
; GR32 regs.

@x = dso_local thread_local global i32 0, align 4
define weak_odr hidden ptr @fun0() {
; CHECK: name: fun0
; CHECK: {{%[0-9]+}}:gr32bit = EAR $a0
; CHECK: {{%[0-9]+}}:gr32bit = EAR $a1
  ret ptr @x
}

define i32 @fun1() {
; CHECK: name: fun1
; CHECK: [[VREG0:%[0-9]+]]:gr32bit = COPY %0
; CHECK-NEXT: $a1 = SAR [[VREG0]]
; CHECK: {{%[0-9]+}}:gr32bit = EAR $a0
  %val = call i32 asm "blah", "={a0}, {a1}" (i32 0)
  ret i32 %val
}

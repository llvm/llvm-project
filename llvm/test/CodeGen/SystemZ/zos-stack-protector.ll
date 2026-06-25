; Test the stack protector under XPLINK on z/OS
;
; RUN: llc < %s -mtriple=s390x-ibm-zos -mcpu=z13 | FileCheck --check-prefixes=CHECK %s

; Test stack protector for non-XPLEAF.

; Small stack frame.
; CHECK-LABEL: func0
; CHECK:                    * DSA Size [[#%#x,DSA_SIZE:]]
; CHECK:                    aghi  4,-[[#%u,mul(div(DSA_SIZE,32),32)]]
; CHECK:                    llgt  [[REG1:[0-9]+]],1208
; CHECK:                    mvc   [[#%u,2040+mul(div(DSA_SIZE,32),32)]](8,4),152([[REG1]])
; ...
; CHECK:                    llgt  [[REG3:[0-9]+]],1208
; CHECK:                    clc   [[#%u,2040+mul(div(DSA_SIZE,32),32)]](8,4),152([[REG3]])
; CHECK:                    jlh   [[FAIL_LABEL:L#BB[0-9_]+]]
; success block
; CHECK:                    aghi  4,[[#%u,mul(div(DSA_SIZE,32),32)]]
; CHECK:                    b     2(7)
; failure block
; CHECK:                    [[FAIL_LABEL]] DS 0H
; invoke __stack_chk_fail
; CHECK:                    lg    6,[[#CHK_FAIL_OFF:]]({{[0-9]+}})
; CHECK:                    lg    5,[[#CHK_FAIL_OFF-8]]({{[0-9]+}})
; CHECK:                    basr  7,6
; CHECK-NEXT:               bcr   0,0

define void @func0() sspreq {
  call i64 (i64) @fun(i64 10)
  ret void
}

; Large stack frame.
; Larger than 1M in XPLINK64.
; CHECK-LABEL: func1
; CHECK:                    * DSA Size [[#%#x,DSA_SIZE:]]
; CHECK:                    stmg  6,{{[0-9]+}},2064(4)
; CHECK:                    llilh [[REG_CANARY_OFF_HIGH:[0-9]+]],[[#%u,CANARY_OFF_HIGH:div(DSA_SIZE,65536)]]
; CHECK:                    la    [[REG_CANARY_OFF_HIGH]],0([[REG_CANARY_OFF_HIGH]],4)
; CHECK:                    llgt  [[REG1:[0-9]+]],1208
; CHECK:                    mvc   [[#%u,2040+mul(div(DSA_SIZE,32),32)-mul(CANARY_OFF_HIGH,65536)]](8,[[REG_CANARY_OFF_HIGH]]),152([[REG1]])
; ...
; CHECK:                    llilh [[REG_CANARY_OFF_HIGH_2:[0-9]+]],[[#%u,CANARY_OFF_HIGH_2:div(DSA_SIZE,65536)]]
; CHECK:                    la    [[REG_CANARY_OFF_HIGH_2]],0([[REG_CANARY_OFF_HIGH_2]],4)
; CHECK:                    llgt  [[REG3:[0-9]+]],1208
; CHECK:                    clc   [[#%u,2040+mul(div(DSA_SIZE,32),32)-mul(CANARY_OFF_HIGH_2,65536)]](8,[[REG_CANARY_OFF_HIGH_2]]),152([[REG3]])
; CHECK:                    jlh   [[FAIL_LABEL:L#BB[0-9_]+]]
; success block
; CHECK:                    agfi  4,[[#%u,mul(div(DSA_SIZE,32),32)]]
; CHECK:                    b     2(7)
; failure block
; CHECK:                    [[FAIL_LABEL]] DS 0H
; invoke __stack_chk_fail
; CHECK:                    lg    6,[[#CHK_FAIL_OFF:]]({{[0-9]+}})
; CHECK:                    lg    5,[[#CHK_FAIL_OFF-8]]({{[0-9]+}})
; CHECK:                    basr  7,6
; CHECK-NEXT:               bcr   0,0

define void @func1() sspreq {
  %arr = alloca [131072 x i64], align 8
  call i64 (ptr) @fun1(ptr %arr)
  ret void
}

; Test converting XPLeaf functions to non-leaf functions if they need stack protection

; TODO: Currently any function that needs to store data on the stack is
; converted to a non-leaf function, so XPLeaf functions never write to the
; stack, and thus there is no way for them to cause stack corruption.
;
; Eventually we'll start taking advantage of the 2048 bytes of space between R4
; and the caller's stack frame to eliminate the need to convert some functions
; that would have been XPLeaf functions. At which point, it will be possible for
; an XPLeaf function to corrupt the stack.
;
; Since stack protection protects against corruption of the caller's stack
; frame and not corruption of the callee's stack frame, it doesn't matter that
; XPLeaf functions don't have a stack frame of their own - that's not what we'd
; be protecting anyways.
;
; Thus, we'll have to choose what to do with functions that need stack protection
; but could remain as XPLeaf functions by using those 2048 bytes of space.
; We have 3 options:
; 1. Convert them to non-leaf functions and continue protecting them as before.
; 2. Keep them as XPLeaf functions, but give up on stack protecting them
; 3. Keep them as XPLeaf functions, and try to stack protect them without making
;    any function calls in the failure case. This would probably involve
;    delaying the invocation of __stack_chk_fail/__CEL4SFCR until we return to
;    the caller.
;
; If we choose option 1, leave this test as is and remove this TODO.
; If we choose option 2, make sure we don't try to stack protect these functions.
; If we choose option 3, this test needs to be replaced, but what we replace it
; with will depend on how we implement the failure case.

; Based on func3_64 in call-zos-03.ll
; CHECK-LABEL: func2_64
; CHECK:                    * Entry Flags
; CHECK-NEXT:               *   Bit 1: 0 = Non-leaf function
define i64 @func2_64(i64 %arg0) sspreq {
  %out = add i64 %arg0, 55
  ret i64 %out
}

; Based on func6 in zos-prologue-epilog.ll. R15 is callee-saved, so needs to be
; spilled to the stack before we use it. As a result this currently gets
; converted to a non-leaf function (even without sspreq).
; CHECK-LABEL: func3
; CHECK:                    * Entry Flags
; CHECK-NEXT:               *   Bit 1: 0 = Non-leaf function
define void @func3() local_unnamed_addr sspreq #0 {
entry:
  tail call void asm sideeffect " lhi 15,1\0A", "~{r15}"()
  ret void
}

; Test stack protector for function that uses alloca()

; CHECK-LABEL: func4
; CHECK:                    * DSA Size [[#%#x,DSA_SIZE:]]
; CHECK:                    * Entry Flags
; CHECK-NEXT:               *   Bit 1: 0 = Non-leaf function
; CHECK-NEXT:               *   Bit 2: 1 = Uses alloca
; CHECK:                    stmg  4,[[SPILLHI:[0-9]+]],[[#%u,2048-mul(div(DSA_SIZE,32),32)]](4)
; CHECK:                    aghi  4,-[[#%u,mul(div(DSA_SIZE,32),32)]]
; CHECK:                    lgr   [[ALLOCAREG:[0-9]+]],4
; CHECK:                    llgt  [[REG1:[0-9]+]],1208
; CHECK:                    mvc   [[#%u,2040+mul(div(DSA_SIZE,32),32)]](8,[[ALLOCAREG]]),152([[REG1]])
; ...
; CHECK:                    llgt  [[REG3:[0-9]+]],1208
; CHECK:                    clc   [[#%u,2040+mul(div(DSA_SIZE,32),32)]](8,[[ALLOCAREG]]),152([[REG3]])
; CHECK:                    jlh   [[FAIL_LABEL:L#BB[0-9_]+]]
; success block
; CHECK:                    lmg   4,[[SPILLHI]],2048(4)
; CHECK:                    b     2(7)
; failure block
; CHECK:                    [[FAIL_LABEL]] DS 0H
; invoke __stack_chk_fail
; CHECK:                    lg    6,[[#CHK_FAIL_OFF:]]({{[0-9]+}})
; CHECK:                    lg    5,[[#CHK_FAIL_OFF-8]]({{[0-9]+}})
; CHECK:                    basr  7,6
; CHECK-NEXT:               bcr   0,0

define i64 @func4(i64 %n) sspreq {
  %vla = alloca i64, i64 %n, align 8
  %call = call i64 @fun2(i64 %n, ptr nonnull %vla, ptr nonnull %vla)
  ret i64 %call
}

; Test stack protector off.

; CHECK-LABEL: func5
define void @func5() {
  call i64 (i64) @fun(i64 10)
  ret void
}

declare i64 @fun(i64 %arg0)
declare i64 @fun1(ptr %ptr)
declare i64 @fun2(i64 %n, ptr %arr0, ptr %arr1)

; CHECK-LABEL: L#PPA1_func0_0 DS 0H
; CHECK:                    * PPA1 Flags 2
; CHECK-NOT:                * PPA1 Flags 3
; CHECK:                    *   Bit 3: 1 = STACKPROTECT is enabled

; CHECK-LABEL: L#PPA1_func1_0 DS 0H
; CHECK:                    * PPA1 Flags 2
; CHECK-NOT:                * PPA1 Flags 3
; CHECK:                    *   Bit 3: 1 = STACKPROTECT is enabled

; CHECK-LABEL: L#PPA1_func2_64_0 DS 0H
; CHECK:                    * PPA1 Flags 2
; CHECK-NOT:                * PPA1 Flags 3
; CHECK:                    *   Bit 3: 1 = STACKPROTECT is enabled

; CHECK-LABEL: L#PPA1_func3_0 DS 0H
; CHECK:                    * PPA1 Flags 2
; CHECK-NOT:                * PPA1 Flags 3
; CHECK:                    *   Bit 3: 1 = STACKPROTECT is enabled

; CHECK-LABEL: L#PPA1_func4_0 DS 0H
; CHECK:                    * PPA1 Flags 2
; CHECK-NOT:                * PPA1 Flags 3
; CHECK:                    *   Bit 3: 1 = STACKPROTECT is enabled

; CHECK-LABEL: L#PPA1_func5_0 DS 0H
; CHECK:                    * PPA1 Flags 2
; CHECK-NOT:                * PPA1 Flags 3
; CHECK:                    *   Bit 3: 0 = STACKPROTECT is not enabled

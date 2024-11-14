; RUN: llc --force-dwarf-frame-section %s -o - | FileCheck %s
; RUN: llc --filetype=obj %s -o - | llvm-readelf -s --unwind - | FileCheck %s --check-prefix=UNWIND
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-arm-none-eabi"

; -Oz
; __attribute__((noinline)) int h(int a, int b) { return a + b; }
;
; int f(int a, int b, int c, int d) {
;   if (a < 0)
;     return -1;
;   a = h(11 * a - b, b);
;   return 2 + a * (a + b) / (c + d);
; }
;
; int g(int a, int b, int c, int d) {
;   if (a < 0)
;     return -1;
;   a = h(11 * a - b, b);
;   return 1 + a * (a + b) / (c + d);
; }

define hidden i32 @h(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %add = add nsw i32 %b, %a
  ret i32 %add
}

define hidden i32 @f(i32 %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
entry:
  %cmp = icmp slt i32 %a, 0
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %mul = mul nsw i32 %a, 11
  %xor = xor i32 %mul, 100
  %sub = sub nsw i32 %xor, %b
  %call = tail call i32 @h(i32 %sub, i32 %b)
  %add = add nsw i32 %call, %b
  %mul1 = mul nsw i32 %add, %call
  %add2 = add nsw i32 %d, %c
  %div = sdiv i32 %mul1, %add2
  %add3 = add nsw i32 %div, 2
  br label %return

return:                                           ; preds = %entry, %if.end
  %retval.0 = phi i32 [ %add3, %if.end ], [ -1, %entry ]
  ret i32 %retval.0
}

; CHECK-LABEL: f:
; ...
; CHECK:       pac    r12, lr, sp
; CHECK-NEXT:  .save  {r4, r5, r6, lr}
; CHECK-NEXT:  push   {r4, r5, r6, lr}
; CHECK-NEXT:  .cfi_def_cfa_offset 16
; CHECK-NEXT:  .cfi_offset lr, -4
; CHECK-NEXT:  .cfi_offset r6, -8
; CHECK-NEXT:  .cfi_offset r5, -12
; CHECK-NEXT:  .cfi_offset r4, -16
; CHECK-NEXT:  .save  {ra_auth_code}
; CHECK-NEXT:  str    r12, [sp, #-4]!
; CHECK-NEXT:  .cfi_def_cfa_offset 20
; CHECK-NEXT:  .cfi_offset ra_auth_code, -20
; CHECK-NEXT:  .pad    #4
; CHECK-NEXT:  sub     sp, #4
; CHECK-NEXT:  .cfi_def_cfa_offset 24
; ...
; CHECK:        bl  OUTLINED_FUNCTION_0
; ...
; CHECK:        add    sp, #4
; CHECK-NEXT:   ldr    r12, [sp], #4
; CHECK-NEXT:   pop.w  {r4, r5, r6, lr}
; CHECK-NEXT:   aut    r12, lr, sp
; CHECK-NEXT:   bx     lr


define hidden i32 @g(i32 %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
entry:
  %cmp = icmp slt i32 %a, 0
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %mul = mul nsw i32 %a, 11
  %xor = xor i32 %mul, 100
  %sub = sub nsw i32 %xor, %b
  %call = tail call i32 @h(i32 %sub, i32 %b)
  %add = add nsw i32 %call, %b
  %mul1 = mul nsw i32 %add, %call
  %add2 = add nsw i32 %d, %c
  %div = sdiv i32 %mul1, %add2
  %add3 = add nsw i32 %div, 1
  br label %return

return:                                           ; preds = %entry, %if.end
  %retval.0 = phi i32 [ %add3, %if.end ], [ -1, %entry ]
  ret i32 %retval.0
}
; CHECK-LABEL: g:
; CHECK:       pac    r12, lr, sp
; CHECK-NEXT:  .save  {r4, r5, r6, lr}
; CHECK-NEXT:  push   {r4, r5, r6, lr}
; CHECK-NEXT:  .cfi_def_cfa_offset 16
; CHECK-NEXT:  .cfi_offset lr, -4
; CHECK-NEXT:  .cfi_offset r6, -8
; CHECK-NEXT:  .cfi_offset r5, -12
; CHECK-NEXT:  .cfi_offset r4, -16
; CHECK-NEXT:  .save  {ra_auth_code}
; CHECK-NEXT:  str    r12, [sp, #-4]!
; CHECK-NEXT:  .cfi_def_cfa_offset 20
; CHECK-NEXT:  .cfi_offset ra_auth_code, -20
; CHECK-NEXT:  .pad   #4
; CHECK-NEXT:  sub    sp, #4
; CHECK-NEXT:  .cfi_def_cfa_offset 24
; ...
; CHECK:        bl  OUTLINED_FUNCTION_0
; ...
; CHECK:       add    sp, #4
; CHECK-NEXT:  ldr    r12, [sp], #4
; CHECK-NEXT:  pop.w  {r4, r5, r6, lr}
; CHECK-NEXT:  aut    r12, lr, sp
; CHECK-NEXT:  bx     lr

; CHECK-LABEL: OUTLINED_FUNCTION_0:
; CHECK:       pac    r12, lr, sp
; CHECK-NEXT:  .save  {ra_auth_code, lr}
; CHECK-NEXT:  strd    r12, lr, [sp, #-8]!
; CHECK-NEXT:  .cfi_def_cfa_offset 8
; CHECK-NEXT:  .cfi_offset lr, -4
; CHECK-NEXT:  .cfi_offset ra_auth_code, -8
; ...
; CHECK:       ldrd   r12, lr, [sp], #8
; CHECK-NEXT:  .cfi_def_cfa_offset 0
; CHECK-NEXT:  .cfi_restore lr
; CHECK-NEXT:  .cfi_undefined ra_auth_code
; CHECK-NEXT:  aut   r12, lr, sp
; CHECK-NEXT:  bx    lr

attributes #0 = { minsize noinline norecurse nounwind optsize readnone uwtable "sign-return-address"="non-leaf" }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 8, !"branch-target-enforcement", i32 0}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 0}


; UNWIND-LABEL: FunctionAddress: 0x4
; UNWIND:       0x00      ; vsp = vsp + 4
; UNWIND-NEXT:  0xB4      ; pop ra_auth_code
; UNWIND-NEXT:  0xAA      ; pop {r4, r5, r6, lr}

; UNWIND-LABEL: FunctionAddress: 0x30
; UNWIND:       0x00      ; vsp = vsp + 4
; UNWIND-NEXT:  0xB4      ; pop ra_auth_code
; UNWIND-NEXT:  0xAA      ; pop {r4, r5, r6, lr}

; UNWIND-LABEL: FunctionAddress: 0x5C
; UNWIND:       0xB4      ; pop ra_auth_code
; UNWIND:       0x84 0x00 ; pop {lr}
 
; UNWIND-LABEL: 0000005d {{.*}} OUTLINED_FUNCTION_0
; UNWIND-LABEL: 00000005 {{.*}} f
; UNWIND-LABEL: 00000031 {{.*}} g

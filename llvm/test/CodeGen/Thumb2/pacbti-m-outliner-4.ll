; RUN: llc --force-dwarf-frame-section %s -o - | FileCheck %s
; RUN: llc --filetype=obj %s -o - | llvm-readelf -s --unwind - | FileCheck %s --check-prefix=UNWIND
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-arm-none-eabi"

; C++, -Oz
; __attribute__((noinline)) int h(int a, int b) {
;   if (a < 0)
;     throw 1;
;   return a + b;
; }
;
; int f(int a, int b, int c, int d) {
;   if (a < 0)
;     return -1;
;   a = h(a, b);
;   return 2 + a * (a + b) / (c + d);
; }
;
; int g(int a, int b, int c, int d) {
;   if (a < 0)
;     return -1;
;   a = h(a, b);
;   return 1 + a * (a + b) / (c + d);
; }

@_ZTIi = external dso_local constant ptr

define hidden i32 @_Z1hii(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %cmp = icmp slt i32 %a, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %exception = tail call ptr @__cxa_allocate_exception(i32 4) #1
  store i32 1, ptr %exception, align 8
  tail call void @__cxa_throw(ptr %exception, ptr @_ZTIi, ptr null) #2
  unreachable

if.end:                                           ; preds = %entry
  %add = add nsw i32 %b, %a
  ret i32 %add
}

; CHECK-LABEL: _Z1hii:
; ...
; CHECK:    pac    r12, lr, sp
; CHECK-NEXT:    .save    {r7, lr}
; CHECK-NEXT:    push    {r7, lr}
; CHECK-NEXT:    .cfi_def_cfa_offset 8
; CHECK-NEXT:    .cfi_offset lr, -4
; CHECK-NEXT:    .cfi_offset r7, -8
; CHECK-NEXT:    .save    {ra_auth_code}
; CHECK-NEXT:    str    r12, [sp, #-4]!
; CHECK-NEXT:    .cfi_def_cfa_offset 12
; CHECK-NEXT:    .cfi_offset ra_auth_code, -12
; CHECK-NEXT:    .pad    #4
; CHECK-NEXT:    sub    sp, #4
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; ...
; CHECK-NOT: pac
; CHECK: aut
; CHECK:     .cfi_endproc

declare dso_local ptr @__cxa_allocate_exception(i32) local_unnamed_addr

declare dso_local void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr

define hidden i32 @_Z1fiiii(i32 %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
entry:
  %cmp = icmp slt i32 %a, 0
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %call = tail call i32 @_Z1hii(i32 %a, i32 %b)
  %add = add nsw i32 %call, %b
  %mul = mul nsw i32 %add, %call
  %add1 = add nsw i32 %d, %c
  %div = sdiv i32 %mul, %add1
  %add2 = add nsw i32 %div, 2
  br label %return

return:                                           ; preds = %entry, %if.end
  %retval.0 = phi i32 [ %add2, %if.end ], [ -1, %entry ]
  ret i32 %retval.0
}

; CHECK-LABEL: _Z1fiiii:
; ...
; CHECK:    pac    r12, lr, sp
; CHECK-NEXT:    .save    {r4, r5, r6, lr}
; CHECK-NEXT:    push    {r4, r5, r6, lr}
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    .cfi_offset lr, -4
; CHECK-NEXT:    .cfi_offset r6, -8
; CHECK-NEXT:    .cfi_offset r5, -12
; CHECK-NEXT:    .cfi_offset r4, -16
; CHECK-NEXT:    .save    {ra_auth_code}
; CHECK-NEXT:    str    r12, [sp, #-4]!
; CHECK-NEXT:    .cfi_def_cfa_offset 20
; CHECK-NEXT:    .cfi_offset ra_auth_code, -20
; CHECK-NEXT:    .pad    #4
; CHECK-NEXT:    sub    sp, #4
; CHECK-NEXT:    .cfi_def_cfa_offset 24
; ...
; CHECK:    bl	OUTLINED_FUNCTION_0
; ...
; CHECK:    add    sp, #4
; CHECK-NEXT:    ldr    r12, [sp], #4
; CHECK-NEXT:    pop.w    {r4, r5, r6, lr}
; CHECK-NEXT:    aut    r12, lr, sp
; CHECK-NEXT:    bx    lr



define hidden i32 @_Z1giiii(i32 %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
entry:
  %cmp = icmp slt i32 %a, 0
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %call = tail call i32 @_Z1hii(i32 %a, i32 %b)
  %add = add nsw i32 %call, %b
  %mul = mul nsw i32 %add, %call
  %add1 = add nsw i32 %d, %c
  %div = sdiv i32 %mul, %add1
  %add2 = add nsw i32 %div, 1
  br label %return

return:                                           ; preds = %entry, %if.end
  %retval.0 = phi i32 [ %add2, %if.end ], [ -1, %entry ]
  ret i32 %retval.0
}

; CHECK-LABEL: _Z1giiii:
; ...
; CHECK:    pac    r12, lr, sp
; CHECK-NEXT:    .save    {r4, r5, r6, lr}
; CHECK-NEXT:    push    {r4, r5, r6, lr}
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    .cfi_offset lr, -4
; CHECK-NEXT:    .cfi_offset r6, -8
; CHECK-NEXT:    .cfi_offset r5, -12
; CHECK-NEXT:    .cfi_offset r4, -16
; CHECK-NEXT:    .save    {ra_auth_code}
; CHECK-NEXT:    str    r12, [sp, #-4]!
; CHECK-NEXT:    .cfi_def_cfa_offset 20
; CHECK-NEXT:    .cfi_offset ra_auth_code, -20
; CHECK-NEXT:    .pad    #4
; CHECK-NEXT:    sub    sp, #4
; CHECK-NEXT:    .cfi_def_cfa_offset 24
; ...
; CHECK:    bl	OUTLINED_FUNCTION_0
; ...
; CHECK:    add    sp, #4
; CHECK-NEXT:    ldr    r12, [sp], #4
; CHECK-NEXT:    pop.w    {r4, r5, r6, lr}
; CHECK-NEXT:    aut    r12, lr, sp
; CHECK-NEXT:    bx    lr


; CHEK-LABEL: OUTLINED_FUNCTION_0:
; CHECK-NOT: pac
; CHECK-NOT: aut
; CHECK:        b    _Z1hii

attributes #0 = { minsize noinline optsize "sign-return-address"="non-leaf" "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="cortex-m3" "target-features"="+armv7-m,+hwdiv,+thumb-mode" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "sign-return-address"="non-leaf" }
attributes #2 = { noreturn "sign-return-address"="non-leaf" }


!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 8, !"branch-target-enforcement", i32 0}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 0}


; UNWIND-LABEL: FunctionAddress: 0x0
; UNWIND:       Opcodes
; UNWIND-NEXT:  0x00      ; vsp = vsp + 4
; UNWIND-NEXT:  0xB4      ; pop ra_auth_code
; UNWIND-NEXT:  0x84 0x08 ; pop {r7, lr}
; UNWIND-NEXT:  0xB0      ; finish
; UNWIND-NEXT:  0xB0      ; finish

; UNWIND-LABEL: FunctionAddress: 0x3C
; UNWIND:       Opcodes
; UNWIND-NEXT:  0x00      ; vsp = vsp + 4
; UNWIND-NEXT:  0xB4      ; pop ra_auth_code
; UNWIND-NEXT:  0xAA      ; pop {r4, r5, r6, lr}

; UNWIND-LABEL: FunctionAddress: 0x72
; UNWIND:       Opcodes
; UNWIND-NEXT:  0x00      ; vsp = vsp + 4
; UNWIND-NEXT:  0xB4      ; pop ra_auth_code
; UNWIND-NEXT:  0xAA      ; pop {r4, r5, r6, lr}

; UNWIND-LABEL: FunctionAddress: 0xA8
; UNWIND:       Opcodes
; UNWIND-NEXT:  0xB0      ; finish
; UNWIND-NEXT:  0xB0      ; finish
; UNWIND-NEXT:  0xB0      ; finish

; UNWIND: 000000a9 {{.*}} OUTLINED_FUNCTION_0
; UWNIND: 00000001 {{.*}} _Z1hii
; UWNIND: 0000003d {{.*}} _Z1fiiii
; UWNIND: 00000073 {{.*}} _Z1giiii

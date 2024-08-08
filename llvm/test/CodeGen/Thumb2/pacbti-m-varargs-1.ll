; RUN: llc --force-dwarf-frame-section %s -o - | FileCheck %s
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-none-eabi"

%"struct.std::__va_list" = type { ptr }

define hidden i32 @_Z1fiz(i32 %n, ...) local_unnamed_addr #0 {
entry:
  %ap = alloca %"struct.std::__va_list", align 4
  call void @llvm.va_start(ptr nonnull %ap)
  %cmp7 = icmp sgt i32 %n, 0
  br i1 %cmp7, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %argp.cur.pre = load ptr, ptr %ap, align 4
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %s.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  call void @llvm.va_end(ptr nonnull %ap)
  ret i32 %s.0.lcssa

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %argp.cur = phi ptr [ %argp.cur.pre, %for.body.lr.ph ], [ %argp.next, %for.body ]
  %i.09 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %s.08 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
  store ptr %argp.next, ptr %ap, align 4
  %0 = load i32, ptr %argp.cur, align 4
  %add = add nsw i32 %0, %s.08
  %inc = add nuw nsw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: _Z1fiz:
; CHECK:      pac    r12, lr, sp
; CHECK-NEXT: .pad    #12
; CHECK-NEXT: sub    sp, #12
; CHECK-NEXT: .cfi_def_cfa_offset 12
; CHECK-NEXT:  .save    {r7, lr}
; CHECK-NEXT: push    {r7, lr}
; CHECK-NEXT: .cfi_def_cfa_offset 20
; CHECK-NEXT: .cfi_offset lr, -16
; CHECK-NEXT: .cfi_offset r7, -20
; CHECK-NEXT: .save    {ra_auth_code}
; CHECK-NEXT: str    r12, [sp, #-4]!
; CHECK-NEXT: .cfi_def_cfa_offset 24
; CHECK-NEXT: .cfi_offset ra_auth_code, -24
; CHECK-NEXT: .pad    #4
; CHECK-NEXT: sub    sp, #4
; CHECK-NEXT: .cfi_def_cfa_offset 28
; ...
; CHECK:      add.w r[[N:[0-9]*]], sp, #16
; CHECK:      stm.w r[[N]], {r1, r2, r3}
; ...
; CHECK:      add    sp, #4
; CHECK-NEXT: ldr    r12, [sp], #4
; CHECK-NEXT: pop.w    {r7, lr}
; CHECK-NEXT: add    sp, #12
; CHECK-NEXT: aut    r12, lr, sp
; CHECK-NEXT: bx    lr

declare void @llvm.va_start(ptr) #1
declare void @llvm.va_end(ptr) #1

attributes #0 = { nounwind optsize "sign-return-address"="non-leaf" }
attributes #1 = { nounwind "sign-return-address"="non-leaf" }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 8, !"branch-target-enforcement", i32 0}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 0}

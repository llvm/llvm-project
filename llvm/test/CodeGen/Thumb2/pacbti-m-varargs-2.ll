; RUN: llc --force-dwarf-frame-section %s -o - | FileCheck %s
; RUN: llc --filetype=obj %s -o - | llvm-readelf --unwind - | FileCheck %s --check-prefix=UNWIND
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-none-eabi"

; C++
; int g(int);
;
; int f(int n, ...) {
;   __builtin_va_list ap;
;   __builtin_va_start(ap, n);
;   int s = 0;
;   for (int i = 0; i < n; ++i)
;     s += g(__builtin_va_arg(ap, int));
;   __builtin_va_end(ap);
;   return s;
; }

%"struct.std::__va_list" = type { i8* }

define hidden i32 @_Z1fiz(i32 %n, ...) local_unnamed_addr #0 {
entry:
  %ap = alloca %"struct.std::__va_list", align 4
  %0 = bitcast %"struct.std::__va_list"* %ap to i8*
  call void @llvm.va_start(i8* nonnull %0)
  %cmp7 = icmp sgt i32 %n, 0
  br i1 %cmp7, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %1 = getelementptr inbounds %"struct.std::__va_list", %"struct.std::__va_list"* %ap, i32 0, i32 0
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %s.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  call void @llvm.va_end(i8* nonnull %0)
  ret i32 %s.0.lcssa

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.09 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %s.08 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %argp.cur = load i8*, i8** %1, align 4
  %argp.next = getelementptr inbounds i8, i8* %argp.cur, i32 4
  store i8* %argp.next, i8** %1, align 4
  %2 = bitcast i8* %argp.cur to i32*
  %3 = load i32, i32* %2, align 4
  %call = call i32 @_Z1gi(i32 %3)
  %add = add nsw i32 %call, %s.08
  %inc = add nuw nsw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: _Z1fiz:
; CHECK:       pac    r12, lr, sp
; CHECK-NEXT:  .pad   #12
; CHECK-NEXT:  sub    sp, #12
; CHECK-NEXT:  .cfi_def_cfa_offset 12
; CHECK-NEXT:  .save   {r4, r5, r7, lr}
; CHECK-NEXT:  push    {r4, r5, r7, lr}
; CHECK-NEXT:  .cfi_def_cfa_offset 28
; CHECK-NEXT:  .cfi_offset lr, -16
; CHECK-NEXT:  .cfi_offset r7, -20
; CHECK-NEXT:  .cfi_offset r5, -24
; CHECK-NEXT:  .cfi_offset r4, -28
; CHECK-NEXT:  .save  {ra_auth_code}
; CHECK-NEXT:  str    r12, [sp, #-4]!
; CHECK-NEXT: .cfi_def_cfa_offset 32
; CHECK-NEXT: .cfi_offset ra_auth_code, -32
; CHECK-NEXT:  .pad   #8
; CHECK-NEXT:  sub    sp, #8
; CHECK-NEXT: .cfi_def_cfa_offset 40
; ...
; CHECK:       add    r[[N:[0-9]*]], sp, #28
; CHECK:       stm    r[[N]]!, {r1, r2, r3}
; ...
; CHECK:       add    sp, #8
; CHECK-NEXT:  ldr    r12, [sp], #4
; CHECK-NEXT:  pop.w  {r4, r5, r7, lr}
; CHECK-NEXT:  add    sp, #12
; CHECK-NEXT:  aut    r12, lr, sp
; CHECK-NEXT:  bx     lr

declare void @llvm.va_start(i8*) #1
declare void @llvm.va_end(i8*) #1

declare dso_local i32 @_Z1gi(i32) local_unnamed_addr

attributes #0 = { optsize }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 8, !"branch-target-enforcement", i32 0}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 0}

; UNWIND-LABEL: FunctionAddress
; UNWIND:       0x01      ; vsp = vsp + 8
; UNWIND-NEXT:  0xB4      ; pop ra_auth_code
; UNWIND-NEXT:  0x84 0x0B ; pop {r4, r5, r7, lr}
; UNWIND-NEXT:  0x02      ; vsp = vsp + 12

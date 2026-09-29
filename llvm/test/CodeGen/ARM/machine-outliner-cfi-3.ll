; RUN: llc --verify-machineinstrs --force-dwarf-frame-section %s -o - | FileCheck %s
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-unknown-unknown-eabi"

; Derived from
; __attribute__((noinline)) int h(int a, int b) { return a + b; }
;
; int f(int a, int b, int c, int d) {
;     if (a < 0)
;         return -1;
;     a = h(a, b);
;     return 2 + a * (a + b) / (c + d);
; }
;
; int g(int a, int b, int c, int d) {
;     if (a < 0)
;         return -1;
;     a = h(a, b);
;     return 1 + a * (a + b) / (c + d);
; }
; Check CFI instructions inside the outlined function.

define dso_local i32 @h(i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %add = add nsw i32 %b, %a
  ret i32 %add
}

define dso_local i32 @f(i32 %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #1 {
entry:
  %cmp = icmp slt i32 %a, 0
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %call = tail call i32 @h(i32 %a, i32 %b) #2
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

define dso_local i32 @g(i32 %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #1 {
entry:
  %cmp = icmp slt i32 %a, 0
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %call = tail call i32 @h(i32 %a, i32 %b) #2
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

; CHECK-LABEL: OUTLINED_FUNCTION_0:
; CHECK:      str lr, [sp, #-8]!
; CHECK-NEXT: .cfi_def_cfa_offset 8
; CHECK-NEXT: .cfi_offset lr, -8
; CHECK:      ldr lr, [sp], #8
; CHECK-NEXT: .cfi_def_cfa_offset 0
; CHECK-NEXT: .cfi_restore lr
; CHECK-NEXT: bx  lr

attributes #0 = { minsize noinline norecurse nounwind optsize readnone }
attributes #1 = { minsize norecurse nounwind optsize readnone }
attributes #2 = { minsize optsize }

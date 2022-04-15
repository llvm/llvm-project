; RUN: llc -mtriple=thumbv8.1m.main-arm-none-eabi -enable-machine-outliner < %s | \
; RUN: FileCheck %s --check-prefix=BTI
; RUN: llc -mtriple=thumbv8.1m.main-arm-none-eabi -enable-machine-outliner -mattr=+no-bti-at-return-twice < %s | FileCheck %s --check-prefix=NOBTI

; C source
; --------
; jmp_buf buf;
;
; extern void h(int a, int b, int *c);
;
; int f(int a, int b, int c, int d) {
;   if (setjmp(buf) != 0)
;     return -1;
;   h(a, b, &a);
;   return 2 + a * (a + b) / (c + d);
; }
;
; int g(int a, int b, int c, int d) {
;   if (setjmp(buf) != 0)
;     return -1;
;   h(a, b, &a);
;   return 1 + a * (a + b) / (c + d);
; }

@buf = global [20 x i64] zeroinitializer, align 8

define i32 @f(i32 %a, i32 %b, i32 %c, i32 %d) {
; BTI-LABEL: f:
; BTI:       bl OUTLINED_FUNCTION_0
; BTI-NEXT:  bti
; NOBTI-LABEL: f:
; NOBTI:       bl OUTLINED_FUNCTION_0
; NOBTI-NEXT:   cbz	r0, .LBB0_2
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  %call = call i32 @setjmp(i64* getelementptr inbounds ([20 x i64], [20 x i64]* @buf, i32 0, i32 0)) #0
  %cmp.not = icmp eq i32 %call, 0
  br i1 %cmp.not, label %if.end, label %return

if.end:                                           ; preds = %entry
  call void @h(i32 %a, i32 %b, i32* nonnull %a.addr)
  %0 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %0, %b
  %mul = mul nsw i32 %add, %0
  %add1 = add nsw i32 %d, %c
  %div = sdiv i32 %mul, %add1
  %add2 = add nsw i32 %div, 2
  br label %return

return:                                           ; preds = %entry, %if.end
  %retval.0 = phi i32 [ %add2, %if.end ], [ -1, %entry ]
  ret i32 %retval.0
}

define i32 @g(i32 %a, i32 %b, i32 %c, i32 %d) {
; BTI-LABEL: g:
; BTI:       bl OUTLINED_FUNCTION_0
; BTI-NEXT:  bti
; NOBTI-LABEL: g:
; NOBTI:       bl OUTLINED_FUNCTION_0
; NOBTI-NEXT:  cbz	r0, .LBB1_2
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  %call = call i32 @setjmp(i64* getelementptr inbounds ([20 x i64], [20 x i64]* @buf, i32 0, i32 0)) #0
  %cmp.not = icmp eq i32 %call, 0
  br i1 %cmp.not, label %if.end, label %return

if.end:                                           ; preds = %entry
  call void @h(i32 %a, i32 %b, i32* nonnull %a.addr)
  %0 = load i32, i32* %a.addr, align 4
  %add = add nsw i32 %0, %b
  %mul = mul nsw i32 %add, %0
  %add1 = add nsw i32 %d, %c
  %div = sdiv i32 %mul, %add1
  %add2 = add nsw i32 %div, 1
  br label %return

return:                                           ; preds = %entry, %if.end
  %retval.0 = phi i32 [ %add2, %if.end ], [ -1, %entry ]
  ret i32 %retval.0
}

declare void @h(i32, i32, i32*)
declare i32 @setjmp(i64*) #0

attributes #0 = { returns_twice }

!llvm.module.flags = !{!0}

!0 = !{i32 8, !"branch-target-enforcement", i32 1}

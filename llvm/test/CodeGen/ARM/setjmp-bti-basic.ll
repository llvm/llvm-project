; RUN: llc -mtriple=thumbv8.1m.main-arm-none-eabi < %s | FileCheck %s --check-prefix=BTI
; RUN: llc -mtriple=thumbv8.1m.main-arm-none-eabi -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefix=NOBTI

; C source
; --------
; jmp_buf buf;
;
; extern void bar(int x);
;
; int foo(int x) {
;   if (setjmp(buf))
;     x = 0;
;   else
;     bar(x);
;   return x;
; }

@buf = global [20 x i64] zeroinitializer, align 8

define i32 @foo(i32 %x)  "branch-target-enforcement" {
; BTI-LABEL: foo:
; BTI:       bl setjmp
; BTI-NEXT:  bti
; NOBTI-LABEL: foo:
; NOBTI:       bl setjmp
; NOBTI-NOT:   bti

entry:
  %call = call i32 @setjmp(ptr @buf) #0
  %tobool.not = icmp eq i32 %call, 0
  br i1 %tobool.not, label %if.else, label %if.end

if.else:                                          ; preds = %entry
  call void @bar(i32 %x)
  br label %if.end

if.end:                                           ; preds = %entry, %if.else
  %x.addr.0 = phi i32 [ %x, %if.else ], [ 0, %entry ]
  ret i32 %x.addr.0
}

declare void @bar(i32)
declare i32 @setjmp(ptr) #0

attributes #0 = { returns_twice }


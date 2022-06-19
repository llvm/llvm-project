; RUN: llc -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefix=BTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel < %s | FileCheck %s --check-prefix=BTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -fast-isel < %s | FileCheck %s --check-prefix=BTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefix=NOBTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefix=NOBTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -fast-isel -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefix=NOBTI

; C source
; --------
; extern int setjmp(void*);
; extern void notsetjmp(void);
;
; void bbb(void) {
;   setjmp(0);
;   int (*fnptr)(void*) = setjmp;
;   fnptr(0);
;   notsetjmp();
; }

define void @bbb() {
; BTI-LABEL: bbb:
; BTI:       bl setjmp
; BTI-NEXT:  hint #36
; BTI:       blr x{{[0-9]+}}
; BTI-NEXT:  hint #36
; BTI:       bl notsetjmp
; BTI-NOT:   hint #36

; NOBTI-LABEL: bbb:
; NOBTI:     bl setjmp
; NOBTI-NOT: hint #36
; NOBTI:     blr x{{[0-9]+}}
; NOBTI-NOT: hint #36
; NOBTI:     bl notsetjmp
; NOBTI-NOT: hint #36
entry:
  %fnptr = alloca i32 (i8*)*, align 8
  %call = call i32 @setjmp(i8* noundef null) #0
  store i32 (i8*)* @setjmp, i32 (i8*)** %fnptr, align 8
  %0 = load i32 (i8*)*, i32 (i8*)** %fnptr, align 8
  %call1 = call i32 %0(i8* noundef null) #0
  call void @notsetjmp()
  ret void
}

declare i32 @setjmp(i8* noundef) #0
declare void @notsetjmp()

attributes #0 = { returns_twice }

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"branch-target-enforcement", i32 1}

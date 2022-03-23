; RUN: llc -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefix=NOBTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel < %s | FileCheck %s --check-prefix=NOBTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -fast-isel < %s | FileCheck %s --check-prefix=NOBTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefix=NOBTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefix=NOBTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -fast-isel -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefix=NOBTI

; Same as setjmp-bti.ll except that we do not enable branch target enforcement for this
; module. There should be no combination of options that leads to a bti being emitted.

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

; !llvm.module.flags = !{!0}
; !0 = !{i32 1, !"branch-target-enforcement", i32 1}

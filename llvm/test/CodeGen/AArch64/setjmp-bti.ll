; RUN: llc -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s --check-prefix=BTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel < %s | FileCheck %s --check-prefix=BTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -fast-isel < %s | FileCheck %s --check-prefix=BTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+harden-sls-blr< %s | FileCheck %s --check-prefix=BTISLS
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel -mattr=+harden-sls-blr< %s | FileCheck %s --check-prefix=BTISLS
; RUN: llc -mtriple=aarch64-none-linux-gnu -fast-isel   -mattr=+harden-sls-blr< %s | FileCheck %s --check-prefix=BTISLS
; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefix=NOBTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -global-isel -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefix=NOBTI
; RUN: llc -mtriple=aarch64-none-linux-gnu -fast-isel -mattr=+no-bti-at-return-twice < %s | \
; RUN: FileCheck %s --check-prefix=NOBTI

; C source
; --------
; extern int setjmp(ptr);
; extern void notsetjmp(void);
;
; void bbb(void) {
;   setjmp(0);
;   setjmp(0); // With the attributes removed.
;   int (*fnptr)(ptr) = setjmp;
;   fnptr(0); // With attributes added.
;   notsetjmp();
; }

define void @bbb() #1 {
; BTI-LABEL: bbb:
; BTI:       bl setjmp
; BTI-NEXT:  hint #36
; BTI:       bl setjmp
; BTI-NEXT:  hint #36
; BTI:       blr x{{[0-9]+}}
; BTI-NEXT:  hint #36
; BTI:       bl notsetjmp
; BTI-NOT:   hint #36

; BTISLS-LABEL: bbb:
; BTISLS:       bl setjmp
; BTISLS-NEXT:  hint #36
; BTISLS:       bl setjmp
; BTISLS-NEXT:  hint #36
; BTISLS:       bl __llvm_slsblr_thunk_x{{[0-9]+}}
; BTISLS-NEXT:  hint #36
; BTISLS:       bl notsetjmp
; BTISLS-NOT:   hint #36

; NOBTI-LABEL: bbb:
; NOBTI:     bl setjmp
; NOBTI-NOT: hint #36
; NOBTI:     bl setjmp
; NOBTI-NOT: hint #36
; NOBTI:     blr x{{[0-9]+}}
; NOBTI-NOT: hint #36
; NOBTI:     bl notsetjmp
; NOBTI-NOT: hint #36
entry:
  %fnptr = alloca ptr, align 8
  ; The frontend may apply attributes to the call, but it doesn't have to. We
  ; should be looking at the call base, which looks past that to the called function.
  %call = call i32 @setjmp(ptr noundef null) #0
  %call1 = call i32 @setjmp(ptr noundef null)
  store ptr @setjmp, ptr %fnptr, align 8
  %0 = load ptr, ptr %fnptr, align 8
  ; Clang does not attach the attribute here but if it did, it should work.
  %call2 = call i32 %0(ptr noundef null) #0
  call void @notsetjmp()
  ret void
}

declare i32 @setjmp(ptr noundef) #0
declare void @notsetjmp()

attributes #0 = { returns_twice }
attributes #1 = { "branch-target-enforcement" }

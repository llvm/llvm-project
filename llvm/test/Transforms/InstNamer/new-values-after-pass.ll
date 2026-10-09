; RUN: opt -S -passes=lower-atomic -instnamer-after-each-pass %s | FileCheck %s --check-prefix=FINAL
; RUN: opt -disable-output -passes=lower-atomic -instnamer-after-each-pass -print-changed %s 2>&1 | FileCheck %s --check-prefix=CHANGED
; RUN: %if system-linux %{ opt -disable-output -passes=lower-atomic -instnamer-after-each-pass -print-changed=diff %s 2>&1 | FileCheck %s --check-prefix=DIFF %}

; LowerAtomic replaces the atomicrmw with two unnamed instructions. Use nand
; to avoid a select, which fails the separate profile-verification check.
; Their names must be fresh in the after-pass dump.
define i32 @f(ptr %ptr, i32 %value) {
entry:
  %0 = atomicrmw nand ptr %ptr, i32 %value seq_cst
  ret i32 %0
}

; FINAL-LABEL: define i32 @f(ptr %ptr, i32 %value)
; FINAL: %i.1 = load i32, ptr %ptr
; FINAL: %i.2 = and i32 %i.1, %value
; FINAL: ret i32 %i.1

; CHANGED: *** IR Dump At Start ***
; CHANGED: %i.0 = atomicrmw nand ptr %ptr, i32 %value seq_cst
; CHANGED: *** IR Dump After LowerAtomicPass on f ***
; CHANGED: %i.1 = load i32, ptr %ptr
; CHANGED: %i.2 = and i32 %i.1, %value

; DIFF: *** IR Dump At Start ***
; DIFF: %i.0 = atomicrmw nand ptr %ptr, i32 %value seq_cst
; DIFF: *** IR Dump After LowerAtomicPass on f ***
; DIFF: -  %i.0 = atomicrmw nand ptr %ptr, i32 %value seq_cst
; DIFF: +  %i.1 = load i32, ptr %ptr
; DIFF: +  %i.2 = and i32 %i.1, %value

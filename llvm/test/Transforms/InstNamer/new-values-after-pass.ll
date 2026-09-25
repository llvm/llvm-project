; RUN: opt -S -passes=lower-atomic -instnamer-after-each-pass %s | FileCheck %s --check-prefix=FINAL
; RUN: opt -disable-output -passes=lower-atomic -instnamer-after-each-pass -print-changed=diff %s 2>&1 | FileCheck %s --check-prefix=DIFF

; LowerAtomic removes the original atomicrmw and creates unnamed load and
; compare instructions. Their names must be fresh in the after-pass dump.
define i32 @f(ptr %ptr, i32 %value) {
entry:
  %0 = atomicrmw max ptr %ptr, i32 %value seq_cst
  ret i32 %0
}

; FINAL-LABEL: define i32 @f(ptr %ptr, i32 %value)
; FINAL: %i.1 = load i32, ptr %ptr
; FINAL: %i.2 = icmp sgt i32 %i.1, %value
; FINAL: ret i32 %i.1

; DIFF: *** IR Dump At Start ***
; DIFF: %i.0 = atomicrmw max ptr %ptr, i32 %value seq_cst
; DIFF: *** IR Dump After LowerAtomicPass on f ***
; DIFF: %i.1 = load i32, ptr %ptr
; DIFF: %i.2 = icmp sgt i32 %i.1, %value

; Do setup work for all below tests: generate bitcode and combined index
; RUN: llvm-as -function-summary %s -o %t.bc
; RUN: llvm-as -function-summary %p/Inputs/adjustable_threshold.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; Test import with default progressive instruction factor
; RUN: opt -function-import -summary-file %t3.thinlto.bc %s -import-instr-limit=10 -S | FileCheck %s --check-prefix=CHECK --check-prefix=INSTLIM-DEFAULT
; INSTLIM-DEFAULT: call void @staticfunc2.llvm.2()

; Test import with a reduced progressive instruction factor
; RUN: opt -function-import -summary-file %t3.thinlto.bc %s -import-instr-limit=10 -import-instr-evolution-factor=0.5 -S | FileCheck %s --check-prefix=CHECK --check-prefix=INSTLIM-PROGRESSIVE
; INSTLIM-PROGRESSIVE-NOT: call void @staticfunc



declare void @globalfunc1()
declare void @globalfunc2()

define void @entry() {
entry:
; Call site are processed in reversed order!

; On the direct call, we reconsider @largefunction with a higher threshold and
; import it
  call void @globalfunc2()
; When importing globalfunc1, the threshold was limited and @largefunction was
; not imported.
  call void @globalfunc1()
  ret void
}


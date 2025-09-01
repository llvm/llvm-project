; REQUIRES: x86-registered-target
; RUN: llc -print-after=slotindexes -stop-after=slotindexes -mtriple=x86_64-unknown-linux-gnu %s -filetype=asm -o %t 2>&1 | FileCheck %s
; RUN: llc -print-after=slotindexes -stop-after=slotindexes -mtriple=x86_64-unknown-windows-msvc %s -filetype=asm -o %t 2>&1 | FileCheck %s

define void @foo(ptr %p) {
  store i32 0, ptr %p
  call void @llvm.pseudoprobe(i64 5116412291814990879, i64 1, i32 0, i64 -1)
  store i32 0, ptr %p
  ret void
}

;; Check the pseudo probe instruction isn't assigned a slot index.
;CHECK: IR Dump {{.*}}
;CHECK: # Machine code for function foo{{.*}}
;CHECK: {{[0-9]+}}B  bb.0 (%ir-block.0)
;CHECK: {{[0-9]+}}B	 %0:gr64 = COPY killed $r{{di|cx}}
;CHECK: {{^}}        PSEUDO_PROBE 5116412291814990879
;CHECK: {{[0-9]+}}B	 MOV32mi
;CHECK: {{[0-9]+}}B	 RET 0

declare void @llvm.pseudoprobe(i64, i64, i32, i64) #0

attributes #0 = { inaccessiblememonly nounwind willreturn }

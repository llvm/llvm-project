; REQUIRES: x86_64-linux
; RUN: llc -print-after=slotindexes -stop-after=slotindexes -mtriple=x86_64-- %s -filetype=asm -o %t 2>&1 | FileCheck %s

define void @foo(i32* %p) {
  store i32 0, i32* %p
  call void @llvm.pseudoprobe(i64 5116412291814990879, i64 1, i32 0, i64 -1)
  store i32 0, i32* %p
  ret void
}

;; Check the pseudo probe instruction isn't assigned a slot index.
;CHECK: IR Dump {{.*}}
;CHECK: # Machine code for function foo{{.*}}
;CHECK: {{[0-9]+}}B  bb.0 (%ir-block.0)
;CHECK: {{[0-9]+}}B	 %0:gr64 = COPY killed $rdi
;CHECK: {{^}}        PSEUDO_PROBE 5116412291814990879
;CHECK: {{[0-9]+}}B	 MOV32mi
;CHECK: {{[0-9]+}}B	 RET 0

declare void @llvm.pseudoprobe(i64, i64, i32, i64) #0

attributes #0 = { inaccessiblememonly nounwind willreturn }
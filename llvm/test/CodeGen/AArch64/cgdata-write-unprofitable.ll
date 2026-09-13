; RUN: rm -rf %t && split-file %s %t
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner \
; RUN:   -codegen-data-generate -filetype=obj %t/module-a.ll -o %t/a.base.o
; RUN: llvm-objdump -h %t/a.base.o | \
; RUN:   FileCheck /dev/null --implicit-check-not=__llvm_outline
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner \
; RUN:   -codegen-data-generate \
; RUN:   -machine-outliner-publish-unprofitable-candidates \
; RUN:   -filetype=obj %t/module-a.ll -o %t/a.publish.o
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner \
; RUN:   -codegen-data-generate \
; RUN:   -machine-outliner-publish-unprofitable-candidates \
; RUN:   -filetype=obj %t/module-b.ll -o %t/b.publish.o
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner \
; RUN:   -codegen-data-generate \
; RUN:   -machine-outliner-publish-unprofitable-candidates \
; RUN:   -filetype=obj %t/module-c.ll -o %t/c.publish.o
; RUN: llvm-objdump -h %t/a.publish.o %t/b.publish.o | \
; RUN:   FileCheck %s --check-prefix=SECTION
; RUN: llvm-objdump -h %t/c.publish.o | \
; RUN:   FileCheck /dev/null --implicit-check-not=__llvm_outline
; RUN: llvm-cgdata --merge %t/a.publish.o -o %t/count-two.cgdata
; RUN: llvm-cgdata --convert --format text %t/count-two.cgdata -o - | \
; RUN:   FileCheck %s --check-prefix=COUNT-TWO \
; RUN:     --implicit-check-not='Terminals: {{[1-9]}}'
; RUN: llvm-cgdata --merge %t/a.publish.o %t/b.publish.o \
; RUN:   -o %t/count-four.cgdata
; RUN: llvm-cgdata --convert --format text %t/count-four.cgdata -o - | \
; RUN:   FileCheck %s --check-prefix=COUNT-FOUR \
; RUN:     --implicit-check-not='Terminals: {{[1-9]}}'
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner \
; RUN:   -codegen-data-use-path=%t/count-four.cgdata -verify-machineinstrs \
; RUN:   -filetype=obj %t/module-a.ll -o %t/a.read-four.o
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner \
; RUN:   -codegen-data-use-path=%t/count-four.cgdata -verify-machineinstrs \
; RUN:   -filetype=obj %t/module-b.ll -o %t/b.read-four.o
; RUN: llvm-nm --defined-only %t/a.read-four.o %t/b.read-four.o | \
; RUN:   FileCheck %s --check-prefix=READ-FOUR
; RUN: llvm-cgdata --merge %t/a.publish.o %t/c.publish.o \
; RUN:   -o %t/count-two-plus-one.cgdata
; RUN: llvm-cgdata --convert --format text %t/count-two-plus-one.cgdata -o - | \
; RUN:   FileCheck %s --check-prefix=COUNT-TWO \
; RUN:     --implicit-check-not='Terminals: {{[1-9]}}'
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner \
; RUN:   -codegen-data-use-path=%t/count-two-plus-one.cgdata \
; RUN:   -verify-machineinstrs -filetype=obj %t/module-a.ll \
; RUN:   -o %t/a.read-two-plus-one.o
; RUN: llvm-nm --defined-only %t/a.read-two-plus-one.o | \
; RUN:   FileCheck /dev/null --implicit-check-not=OUTLINED_FUNCTION
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner \
; RUN:   -codegen-data-generate \
; RUN:   -machine-outliner-publish-unprofitable-candidates \
; RUN:   -machine-outliner-unprofitable-candidate-max-instrs=1 \
; RUN:   -filetype=obj %t/module-a.ll -o %t/a.max-one-instr.o
; RUN: llvm-objdump -h %t/a.max-one-instr.o | \
; RUN:   FileCheck /dev/null --implicit-check-not=__llvm_outline

; Each of modules A and B has two instances of the two-instruction `mov; bl`
; sequence. The sequence is locally unprofitable, but merging both modules'
; profiles raises its global count to four and makes Read mode outline it.
;
; A single occurrence in module C is not found by the suffix tree and therefore
; cannot be published. Merging A and C consequently retains a count of two and
; preserves the known 2+1 limitation.
;
; SECTION-COUNT-2: __llvm_outline
; COUNT-TWO: Terminals:       2
; COUNT-FOUR: Terminals:       4
; READ-FOUR-COUNT-2: _OUTLINED_FUNCTION_0.content.

;--- module-a.ll
target triple = "arm64-apple-darwin"

declare void @g(i64)

define void @a1() minsize noredzone {
  call void @g(i64 7)
  ret void
}

define void @a2() minsize noredzone {
  call void @g(i64 7)
  ret void
}

;--- module-b.ll
target triple = "arm64-apple-darwin"

declare void @g(i64)

define void @b1() minsize noredzone {
  call void @g(i64 7)
  ret void
}

define void @b2() minsize noredzone {
  call void @g(i64 7)
  ret void
}

;--- module-c.ll
target triple = "arm64-apple-darwin"

declare void @g(i64)

define void @c1() minsize noredzone {
  call void @g(i64 7)
  ret void
}

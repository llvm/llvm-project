; Test that AsmPrinter emits both .type <sym>.cold,@function (STT_FUNC) and
; .size <sym>.cold, <end>-<sym>.cold when splitting cold basic blocks into
; .text.split.<sym> so ELF unwinders and profilers resolve cold frames.

; RUN: split-file %s %t
; RUN: llc < %t/mfs.ll -mtriple=aarch64-linux-gnu -function-sections -basic-block-sections=%t/bb.list -unique-basic-block-section-names | FileCheck %s --check-prefix=MFS

; MFS: .section .text.split.split_func,"ax",@progbits
; MFS-NEXT: split_func.cold:
; MFS: [[COLD_END:\.LBB_END[0-9_]+|\.Lfunc_end[0-9]+]]:
; MFS-NEXT: .type split_func.cold,@function
; MFS-NEXT: .size split_func.cold, [[COLD_END]]-split_func.cold

;--- bb.list
v1
f split_func
c 0

;--- mfs.ll
declare i32 @cold_work()
declare i32 @hot_work()

define i32 @split_func(i1 %cond) nounwind {
entry:
  br i1 %cond, label %cold, label %hot

cold:
  %c = call i32 @cold_work()
  ret i32 %c

hot:
  %h = call i32 @hot_work()
  ret i32 %h
}

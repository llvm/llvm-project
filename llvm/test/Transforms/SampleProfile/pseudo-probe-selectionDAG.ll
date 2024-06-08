; REQUIRES: x86_64-linux
; RUN: opt < %s -codegenprepare -mtriple=x86_64 -S -o %t 
; RUN: FileCheck %s < %t --check-prefix=IR
; RUN: llc -mtriple=x86_64-- -stop-after=finalize-isel %t -o - | FileCheck %s --check-prefix=MIR

define internal i32 @arc_compare(i1 %c) {
entry:
  %0 = load i64, ptr undef, align 8
  br i1 %c, label %return, label %if.end

if.end:                                           ; preds = %entry
;; Check pseudo probes are next to each other at the beginning of this block.
; IR-LABEL: if.end
; IR: call void @llvm.pseudoprobe(i64 5116412291814990879, i64 1, i32 0, i64 -1)
; IR: call void @llvm.pseudoprobe(i64 5116412291814990879, i64 3, i32 0, i64 -1)
  call void @llvm.pseudoprobe(i64 5116412291814990879, i64 1, i32 0, i64 -1)
  %1          = load i16, ptr undef, align 8
  call void @llvm.pseudoprobe(i64 5116412291814990879, i64 3, i32 0, i64 -1)
  %2          = and i16 %1, 16
  %3          = icmp eq i16 %2, 0
;; Check the load-and-cmp sequence is fold into a test instruction.
; MIR-LABEL: bb.1.if.end
; MIR: %[[#REG:]]:gr64 = IMPLICIT_DEF
; MIR: TEST8mi killed %[[#REG]], 1, $noreg, 0, $noreg, 16
; MIR: JCC_1
  br i1 %3, label %return, label %if.end6

if.end6:                                          ; preds = %if.end
  call void @llvm.pseudoprobe(i64 5116412291814990879, i64 5, i32 0, i64 -1)
  br label %return

return:                                           ; preds = %if.end6, %if.end, %entry
  ret i32 undef
}

; Function Attrs: inaccessiblememonly nounwind willreturn
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #0

attributes #0 = { inaccessiblememonly nounwind willreturn }

; REQUIRES: x86_64-linux
; RUN: llc -mtriple=x86_64-- -stop-after=peephole-opt -o - %s | FileCheck %s

define internal i32 @arc_compare(i1 %c) {
entry:
  %0 = load i64, ptr undef, align 8
  br i1 %c, label %return, label %if.end

if.end:                                           ; preds = %entry
; Chek a register copy has been sinked into the compare instruction.
; CHECK: %[[#REG:]]:gr64 = IMPLICIT_DEF 
; CHECK-NOT: %[[#]]:gr64 = MOV64rm %[[#REG]]
; CHECK: PSEUDO_PROBE 5116412291814990879, 3, 0, 0
; CHECK: CMP64mr %[[#REG]], 1
  call void @llvm.pseudoprobe(i64 5116412291814990879, i64 3, i32 0, i64 -1)
  %cmp4 = icmp slt i64 %0, undef
  br i1 %cmp4, label %return, label %if.end6

if.end6:                                          ; preds = %if.end
  call void @llvm.pseudoprobe(i64 5116412291814990879, i64 5, i32 0, i64 -1)
  br label %return

return:                                           ; preds = %if.end6, %if.end, %entry
  ret i32 undef
}

; Function Attrs: inaccessiblememonly nounwind willreturn
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #0

attributes #0 = { inaccessiblememonly nounwind willreturn }

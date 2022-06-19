; RUN: opt < %s -S -early-cse-memssa | FileCheck %s

define i16 @f1() readonly {
  ret i16 0
}

declare void @f2()

; Check that EarlyCSE correctly handles pseudo probes that don't have
; a MemoryAccess. 

define void @f3() {
; CHECK-LABEL: @f3(
; CHECK-NEXT:    [[CALL1:%.*]] = call i16 @f1()
; CHECK-NEXT:    call void @llvm.pseudoprobe
; CHECK-NEXT:    ret void
;
  %call1 = call i16 @f1()
  call void @llvm.pseudoprobe(i64 6878943695821059507, i64 9, i32 0, i64 -1)
  %call2 = call i16 @f1()
  ret void
}


; Function Attrs: inaccessiblememonly nounwind willreturn
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #0

attributes #0 = { inaccessiblememonly nounwind willreturn }
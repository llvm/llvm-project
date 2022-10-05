; RUN: opt < %s -inline -instcombine -function-attrs | llvm-dis
;
; Check that nocapture attributes are added when run after an SCC pass.
; PR3520

define i32 @use(ptr %x) nounwind readonly {
; CHECK: @use(ptr nocapture %x)
  %1 = tail call i64 @strlen(ptr %x) nounwind readonly
  %2 = trunc i64 %1 to i32
  ret i32 %2
}

declare i64 @strlen(ptr) nounwind readonly
; CHECK: declare i64 @strlen(ptr nocapture) nounwind readonly

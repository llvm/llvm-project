; RUN: opt < %s -inline -instcombine -functionattrs | llvm-dis
;
; Check that nocapture attributes are added when run after an SCC pass.
; PR3520

define i32 @use(i8* %x) nounwind readonly {
; CHECK: @use(i8* nocapture %x)
  %1 = tail call i64 @strlen(i8* %x) nounwind readonly
  %2 = trunc i64 %1 to i32
  ret i32 %2
}

declare i64 @strlen(i8*) nounwind readonly
; CHECK: declare i64 @strlen(i8* nocapture) nounwind readonly

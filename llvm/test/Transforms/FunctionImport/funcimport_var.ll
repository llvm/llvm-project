; This test makes sure a static var is not selected as a callee target
; (which will crash compilation).
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/funcimport_var2.ll -o %t2.bc
; RUN: llvm-lto -thinlto -thinlto-action=thinlink -o %t3 %t.bc %t2.bc
; RUN: llvm-lto -thinlto -thinlto-action=import -thinlto-index=%t3 %t.bc %t2.bc
define i32 @_Z4LinkPKcS0_(i8*, i8*) local_unnamed_addr  {
  %3 = tail call i32 @link(i8* %0, i8* %1) #2
  ret i32 %3
}

; Function Attrs: nounwind
declare i32 @link(i8*, i8*) local_unnamed_addr 

; RUN: llc -mtriple=x86_64 -stop-after=finalize-isel < %s | FileCheck %s

define i32 @identity(i32 %a) {
  ret i32 %a
}

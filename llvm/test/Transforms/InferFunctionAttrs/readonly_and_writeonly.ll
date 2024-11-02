; RUN: opt < %s -mtriple=x86_64-- -passes=inferattrs -S | FileCheck --match-full-lines %s

; Frontends can emit math functions with 'readonly', don't crash on it.

; CHECK: declare double @acos(double) [[NOFREE_NOUNWIND_WILLRETURN_READNONE:#[0-9]+]]
declare double @acos(double) readonly

; CHECK-DAG: attributes [[NOFREE_NOUNWIND_WILLRETURN_READNONE]] = { mustprogress nofree nosync nounwind willreturn memory(none) }

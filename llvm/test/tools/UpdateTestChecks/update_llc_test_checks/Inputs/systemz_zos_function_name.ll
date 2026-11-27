; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

define i64 @leaf_func0(i64 %a, i64 %b) {
  %n = add i64 %a, %b
  ret i64 %n
}

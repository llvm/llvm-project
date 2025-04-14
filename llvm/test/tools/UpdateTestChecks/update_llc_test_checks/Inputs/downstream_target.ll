; RUN: llc < %s -mtriple=i686-- | FileCheck %s
define void @foo() {
  ret void
}

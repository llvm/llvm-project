; RUN: opt -S -passes=lowertypetests %s | FileCheck %s

target datalayout = "e-p:32:32"

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset) nounwind readnone

define i1 @foo(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid1")
  ; CHECK: ret i1 false
  ret i1 %x
}

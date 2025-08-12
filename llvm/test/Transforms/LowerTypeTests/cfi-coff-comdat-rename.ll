; REQUIRES: x86-registered-target
; RUN: opt -S -passes=lowertypetests %s | FileCheck %s
; RUN: opt -S -passes=lowertypetests -lowertypetests-summary-action=import -lowertypetests-read-summary=%p/Inputs/import-thinlto-funcs.yaml %s | FileCheck %s

;; This is a check to assert we don't crash with:
;;
;;   LLVM ERROR: Associative COMDAT symbol '...' does not exist.
;;
;; So this just needs to exit normally.
; RUN: opt -S -passes=lowertypetests %s | llc -asm-verbose=false
; RUN: opt -S -passes=lowertypetests -lowertypetests-summary-action=import -lowertypetests-read-summary=%p/Inputs/import-thinlto-funcs.yaml %s | llc -asm-verbose=false

target datalayout = "e-p:64:64"
target triple = "x86_64-pc-windows-msvc"

@a = global [2 x ptr] [ptr @f1, ptr @f2]

; CHECK: $f1.cfi = comdat any
$f1 = comdat any

; CHECK: @f1.cfi() comdat !type !0
define void @f1() comdat !type !0 {
  ret void
}

; CHECK: @f2.cfi() comdat($f1.cfi) !type !0
define void @f2() comdat($f1) !type !0 {
  ret void
}

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset) nounwind readnone

define i1 @foo(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid1")
  ret i1 %x
}

!0 = !{i32 0, !"typeid1"}

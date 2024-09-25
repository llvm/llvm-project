; RUN: opt -global-merge -global-merge-max-offset=100 -S -o - %s | FileCheck %s
; RUN: opt -passes='global-merge<max-offset=100>' -S -o - %s | FileCheck %s

;; Check that we do _not_ merge data with certain special section-names under Mach-O

target datalayout = "e-p:64:64"
target triple = "x86_64-apple-macos11"

; CHECK-NOT: @_MergedGlobals

@cfstring1 = private global i32 1, section "__DATA,__cfstring"
@cfstring2 = private global i32 2, section "__DATA,__cfstring"
@objcclassrefs1 = private global i32 3, section "__DATA,__objc_classrefs,regular,no_dead_strip"
@objcclassrefs2 = private global i32 4, section "__DATA,__objc_classrefs,regular,no_dead_strip"
@objcselrefs1 = private global i32 5, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip"
@objcselrefs2 = private global i32 6, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip"
define void @use() {
  load ptr, ptr @cfstring1
  load ptr, ptr @cfstring2
  load ptr, ptr @objcclassrefs1
  load ptr, ptr @objcclassrefs2
  load ptr, ptr @objcselrefs1
  load ptr, ptr @objcselrefs2
  ret void
}

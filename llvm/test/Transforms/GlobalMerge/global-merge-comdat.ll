; RUN: opt -global-merge -global-merge-max-offset=16 -global-merge-group-by-use=false %s -S -o - | FileCheck %s
; CHECK: @_MergedGlobals = private global <{ i64, i64 }> zeroinitializer, section "__foo", comdat($__foo), align 8

$__foo = comdat nodeduplicate

@__bar = private global i64 0, section "__foo", comdat($__foo), align 8
@__baz = private global i64 0, section "__foo", comdat($__foo), align 8

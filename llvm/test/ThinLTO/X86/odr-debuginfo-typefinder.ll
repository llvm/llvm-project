; RUN: split-file %s %t
; RUN: opt -module-summary %t/src.ll -o %t/src.bc

; RUN: opt -module-summary %t/dst-gv.ll -o %t/dst-gv.bc
; RUN: llvm-lto -thinlto -o %t/idx-gv %t/dst-gv.bc %t/src.bc
; RUN: opt -passes=function-import -summary-file=%t/idx-gv.thinlto.bc %t/dst-gv.bc -S | FileCheck %s

; RUN: opt -module-summary %t/dst-fn.ll -o %t/dst-fn.bc
; RUN: llvm-lto -thinlto -o %t/idx-fn %t/dst-fn.bc %t/src.bc
; RUN: opt -passes=function-import -summary-file=%t/idx-fn.thinlto.bc %t/dst-fn.bc -S | FileCheck %s

; RUN: opt -module-summary %t/dst-inst-dbg.ll -o %t/dst-inst-dbg.bc
; RUN: llvm-lto -thinlto -o %t/idx-inst-dbg %t/dst-inst-dbg.bc %t/src.bc
; RUN: opt -passes=function-import -summary-file=%t/idx-inst-dbg.thinlto.bc %t/dst-inst-dbg.bc -S | FileCheck %s

; CHECK: define available_externally void @g()

;--- src.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
%T = type {}
define void @g() !dbg !3 {
  ret void
}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1)
!1 = !DIFile(filename: "a", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(type: !4, unit: !0, templateParams: !5)
!4 = !DISubroutineType(types: null)
!5 = !{!6}
!6 = !DITemplateValueParameter(type: !7, value: %T zeroinitializer)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, identifier: "ID")

;--- dst-gv.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
%T = type {}
@gv = global i32 0, !dbg !3
define void @f(%T %0) {
  call void @g()
  ret void
}
declare void @g()
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1)
!1 = !DIFile(filename: "a", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIGlobalVariableExpression(var: !4, expr: !DIExpression())
!4 = distinct !DIGlobalVariable(type: !7)
!5 = !{!6}
!6 = !DITemplateValueParameter(type: !7, value: %T zeroinitializer)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, templateParams: !5, identifier: "ID")

;--- dst-fn.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
%T = type {}
define void @caller(%T %0) {
  call void @g()
  ret void
}
define void @f() !dbg !3 {
  ret void
}
declare void @g()
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1)
!1 = !DIFile(filename: "a", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(type: !4, unit: !0, templateParams: !5)
!4 = !DISubroutineType(types: null)
!5 = !{!6}
!6 = !DITemplateValueParameter(type: !7, value: %T zeroinitializer)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, templateParams: !5, identifier: "ID")

;--- dst-inst-dbg.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
%T = type {}
define void @f(%T %0) {
  call void @g(), !dbg !8
  ret void
}
declare void @g()
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1)
!1 = !DIFile(filename: "a", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(type: !4, unit: !0, templateParams: !5)
!4 = !DISubroutineType(types: null)
!5 = !{!6}
!6 = !DITemplateValueParameter(type: !7, value: %T zeroinitializer)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, templateParams: !5, identifier: "ID")
!8 = !DILocation(line: 1, scope: !3)

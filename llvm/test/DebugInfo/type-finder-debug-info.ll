; RUN: opt -S %s | opt -S | FileCheck %s

; CHECK: %T_gv = type {}
; CHECK: %T_fn = type {}
; CHECK: %T_inst_dbg = type {}

%T_gv = type {}
%T_fn = type {}
%T_inst_dbg = type {}

@gv = global i32 0, !dbg !3

define void @f_fn() !dbg !7 {
  ret void
}

define void @f_other() {
  ret void, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1)
!1 = !DIFile(filename: "a", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}

; Reachable only via GlobalVariable metadata
!3 = !DIGlobalVariableExpression(var: !4, expr: !DIExpression())
!4 = distinct !DIGlobalVariable(type: !5)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, templateParams: !6)
!6 = !{!DITemplateValueParameter(value: %T_gv zeroinitializer)}

; Reachable only via Function metadata
!7 = distinct !DISubprogram(type: !8, unit: !0, templateParams: !9)
!8 = !DISubroutineType(types: null)
!9 = !{!DITemplateValueParameter(value: %T_fn zeroinitializer)}

; Reachable only via Instruction DebugLoc
!10 = distinct !DISubprogram(type: !8, unit: !0, templateParams: !11)
!11 = !{!DITemplateValueParameter(value: %T_inst_dbg zeroinitializer)}
!12 = !DILocation(line: 1, scope: !10)

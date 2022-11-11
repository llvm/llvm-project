; Verify functionality of NVPTXGenericToNVVM.cpp pass.
;
; RUN: opt < %s -march nvptx64 -S -generic-to-nvvm | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Generic space variables should be converted to global space AKA addrspace(1).
; The debug info reference should be preserved.
; CHECK-DAG: @static_var = {{.*}}addrspace(1) {{.*}} !dbg !{{[0-9]+}}
@static_var = externally_initialized global i8 0, align 1, !dbg !4
; CHECK-DAG: @.str = {{.*}}addrspace(1)
@.str = private unnamed_addr constant [4 x i8] c"XXX\00", align 1

; Function Attrs: convergent
define void @func() !dbg !8 {
;CHECK-LABEL: @func()
;CHECK-SAME: !dbg [[FUNCNODE:![0-9]+]]
entry:
; References to the variables must be converted back to generic address space.
; CHECK-DAG: addrspacecast ([4 x i8] addrspace(1)* @.str to [4 x i8]*)
  %0 = load i8, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), align 1
  call void @extfunc(i8 signext %0)
; CHECK-DAG: addrspacecast (i8 addrspace(1)* @static_var to i8*)
  %1 = load i8, i8* @static_var, align 1
  call void @extfunc(i8 signext %1)
  ret void
; CHECK: ret void
}

declare void @extfunc(i8 signext)

!llvm.dbg.cu = !{!0}
; CHECK: !llvm.dbg.cu = !{[[CUNODE:![0-9]+]]}
!llvm.module.flags = !{!6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1,
      producer: "clang version 4.0.0",
      isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
; CHECK-DAG: [[CUNODE]] = distinct !DICompileUnit({{.*}} globals: [[GLOBALSNODE:![0-9]+]]
!1 = !DIFile(filename: "foo.cu", directory: "/usr/local/google/home/tra/work/llvm/build/gpu/debug")
!2 = !{}
!3 = !{!4}
; Find list of global variables and make sure it's the one used by DICompileUnit
; CHECK-DAG: [[GLOBALSNODE]] = !{[[GVNODE:![0-9]+]]}
!4 = !DIGlobalVariableExpression(var: !DIGlobalVariable(name: "static_var", scope: !0, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true), expr: !DIExpression())
; Debug info must be updated to reflect new address space.
; CHECK-DAG: [[GVNODE]] = !DIGlobalVariableExpression(var: [[GVVAR:.*]], expr: !DIExpression())
; CHECK-DAG: [[GVVAR]] = !DIGlobalVariable(name: "static_var", scope: [[CUNODE]],{{.*}} type: [[TYPENODE:![0-9]+]]
!5 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
; CHECK: [[TYPENODE]] = !DIBasicType(name: "char"
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "foo", linkageName: "func",
      scope: !1, file: !1, line: 3, type: !9, isLocal: false, isDefinition: true, scopeLine: 3,
      flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
; CHECK: [[FUNCNODE]] = distinct !DISubprogram(name: "foo",
; CHECK-SAME: type: [[STYPENODE:![0-9]+]]
; CHECK-SAME: unit: [[CUNODE]],
!9 = !DISubroutineType(types: !10)
; CHECK: [[STYPENODE]] = !DISubroutineType
!10 = !{null}

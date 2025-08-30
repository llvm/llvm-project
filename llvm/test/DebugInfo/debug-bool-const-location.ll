; REQUIRES: object-emission
; RUN: %llc_dwarf %s -O3 -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s

; CHECK: {{.*}}DW_TAG_variable
; CHECK: {{.*}} DW_OP_lit1
; CHECK-NOT: {{.*}} DW_OP_lit0, DW_OP_not
; CHECK: {{.*}} DW_OP_lit0
; CHECK: {{.*}} DW_AT_name    ("arg")

define void @foo(i8 %"arg.arg") !dbg !5
{
entry:
  %".4" = alloca i1
  %".5" = icmp eq i8 %"arg.arg", 0
  %arg = alloca i1
  br i1 %".5", label %"entry.if", label %"entry.else"
entry.if:
  store i1 false, i1* %arg
  call void @"llvm.dbg.value"(metadata i1 false , metadata !9, metadata !10), !dbg !6
  br label %"entry.endif"
entry.else:
  store i1 true, i1* %arg
  call void @"llvm.dbg.value"(metadata i1 true , metadata !9, metadata !10), !dbg !7
  br label %"entry.endif"
entry.endif:
  %".11" = load i1, i1* %arg
  store i1 %".11", i1* %".4", !dbg !8
  call void @"llvm.dbg.value"(metadata i1 %".11" , metadata !9, metadata !10), !dbg !8
  ret void, !dbg !8
}

declare void @"llvm.dbg.value"(metadata %".1", metadata %".2", metadata %".3")

!llvm.dbg.cu = !{ !2 }
!llvm.module.flags = !{ !11, !12 }

!1 = !DIFile(directory: "", filename: "test")
!2 = distinct !DICompileUnit(emissionKind: FullDebug, file: !1, isOptimized: false, language: DW_LANG_C_plus_plus, runtimeVersion: 0)
!3 = !DIBasicType(encoding: DW_ATE_boolean, name: "bool", size: 8)
!4 = !DISubroutineType(types: !{null})
!5 = distinct !DISubprogram(file: !1, isDefinition: true, isLocal: false, isOptimized: false, line: 5, linkageName: "foo", name: "foo", scope: !1, scopeLine: 5, type: !4, unit: !2)
!6 = !DILocation(column: 1, line: 5, scope: !5)
!7 = !DILocation(column: 1, line: 7, scope: !5)
!8 = !DILocation(column: 1, line: 8, scope: !5)
!9 = !DILocalVariable(arg: 0, file: !1, line: 5, name: "arg", scope: !5, type: !3)
!10 = !DIExpression()
!11 = !{ i32 2, !"Dwarf Version", i32 4 }
!12 = !{ i32 2, !"Debug Info Version", i32 3 }

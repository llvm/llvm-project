;RUN: opt -globalopt -f %s | llc -filetype=obj -o -| llvm-dwarfdump - | FileCheck %s
;REQUIRES: x86-registered-target

;CHECK: DW_AT_name [DW_FORM_strp] {{.*}} "foo"
;CHECK-NEXT: DW_AT_type {{.*}}
;CHECK-NEXT: DW_AT_decl_file {{.*}}
;CHECK-NEXT: DW_AT_decl_line {{.*}}
;CHECK-NEXT: DW_AT_location [DW_FORM_exprloc] (DW_OP_addr 0x0, DW_OP_deref, DW_OP_constu 0x6f, DW_OP_mul, DW_OP_constu 0x0, DW_OP_plus, DW_OP_stack_value)

source_filename = "integer-bool-dwarf.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = internal global i32 0, align 4, !dbg !0

; Function Attrs: noinline nounwind optnone uwtable
define void @set1() #0 !dbg !11 {
entry:
  store i32 111, i32* @foo, align 4, !dbg !14
  ret void, !dbg !15
}

; Function Attrs: noinline nounwind optnone uwtable
define void @set2() #0 !dbg !16 {
entry:
  store i32 0, i32* @foo, align 4, !dbg !17
  ret void, !dbg !18
}

; Function Attrs: noinline nounwind optnone uwtable
define i32 @get() #0 !dbg !19 {
entry:
  %0 = load i32, i32* @foo, align 4, !dbg !22
  ret i32 %0, !dbg !23
}

attributes #0 = { noinline nounwind optnone uwtable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !3, line: 1, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 6.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "integer-bool-dwarf.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 6.0.0 "}
!11 = distinct !DISubprogram(name: "set1", scope: !3, file: !3, line: 3, type: !12, isLocal: false, isDefinition: true, scopeLine: 4, isOptimized: false, unit: !2, variables: !4)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !DILocation(line: 5, column: 7, scope: !11)
!15 = !DILocation(line: 6, column: 1, scope: !11)
!16 = distinct !DISubprogram(name: "set2", scope: !3, file: !3, line: 8, type: !12, isLocal: false, isDefinition: true, scopeLine: 9, isOptimized: false, unit: !2, variables: !4)
!17 = !DILocation(line: 10, column: 7, scope: !16)
!18 = !DILocation(line: 11, column: 1, scope: !16)
!19 = distinct !DISubprogram(name: "get", scope: !3, file: !3, line: 13, type: !20, isLocal: false, isDefinition: true, scopeLine: 14, isOptimized: false, unit: !2, variables: !4)
!20 = !DISubroutineType(types: !21)
!21 = !{!6}
!22 = !DILocation(line: 15, column: 10, scope: !19)
!23 = !DILocation(line: 15, column: 3, scope: !19)

; RUN: not llc -mtriple=bpfel -filetype=obj %s -o %t 2>&1 | FileCheck %s --implicit-check-not=error:
;
; Source code:
;   int f(unsigned n) {
;     int a[n];
;     return __builtin_btf_type_id(&a, 1);
;   }
;   char (*p)[0x100000000ULL];
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm bounds.c
;
; CHECK-DAG: error: BTF array dimension requires a constant element count
; CHECK-DAG: error: BTF array length exceeds the 32-bit limit

@"llvm.btf_type_id.0$7" = external global i64, !llvm.preserve.access.index !0 #0
@p = global ptr null, !dbg !12

define i32 @f(i32 %n) !dbg !6 {
  %id = load i64, ptr @"llvm.btf_type_id.0$7"
  %value = call i64 @llvm.bpf.passthrough.i64.i64(i32 0, i64 %id)
  %result = trunc i64 %value to i32
  ret i32 %result, !dbg !11
}

declare i64 @llvm.bpf.passthrough.i64.i64(i32, i64)
attributes #0 = { "btf_type_id" }

!llvm.dbg.cu = !{!8}
!llvm.module.flags = !{!10}

!0 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1, size: 64)
!1 = !DICompositeType(tag: DW_TAG_array_type, baseType: !2, elements: !{!3})
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = !DISubrange(count: !4)
!4 = !DILocalVariable(name: "__vla_expr0", scope: !6, type: !5, flags: DIFlagArtificial)
!5 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!6 = distinct !DISubprogram(name: "f", scope: !7, file: !7, line: 1, type: !9, spFlags: DISPFlagDefinition, unit: !8)
!7 = !DIFile(filename: "bounds.c", directory: "/")
!8 = distinct !DICompileUnit(language: DW_LANG_C11, file: !7, emissionKind: FullDebug, globals: !{!12})
!9 = !DISubroutineType(types: !{!2, !17})
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !DILocation(line: 3, column: 3, scope: !6)
!12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression())
!13 = distinct !DIGlobalVariable(name: "p", scope: !8, file: !7, line: 5, type: !14, isLocal: false, isDefinition: true)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 34359738368, elements: !{!DISubrange(count: 4294967296)})
!16 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!17 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)

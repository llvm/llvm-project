; AIX doesn't support the debug_addr section
; UNSUPPORTED:  target={{.*}}-aix{{.*}}

; RUN: %llc_dwarf -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -v -debug-info %t.o | FileCheck %s -check-prefix=CHECK 

; LLVM IR generated using: clang -emit-llvm -S -g -gdwarf-5
;
; struct C
; {
;   static int a;
;   const static bool b = true;
;   static constexpr int c = 15;
; };
; 
; int C::a = 10;
; 
; int main()
; {
;   C instance_C;
;   return C::a + (int)C::b + C::c;
; }

source_filename = "llvm/test/DebugInfo/X86/dwarf5-debug-info-static-member.ll"

%struct.C = type { i8 }

@_ZN1C1aE = global i32 10, align 4, !dbg !0

; Function Attrs: mustprogress noinline norecurse nounwind optnone ssp uwtable(sync)
define noundef i32 @main() #0 !dbg !26 {
entry:
  %retval = alloca i32, align 4
  %instance_C = alloca %struct.C, align 1
  store i32 0, ptr %retval, align 4
  call void @llvm.dbg.declare(metadata ptr %instance_C, metadata !30, metadata !DIExpression()), !dbg !31
  %0 = load i32, ptr @_ZN1C1aE, align 4, !dbg !32
  %add = add nsw i32 %0, 1, !dbg !33
  %add1 = add nsw i32 %add, 15, !dbg !34
  ret i32 %add1, !dbg !35
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!19, !20, !21, !22, !23, !24}
!llvm.ident = !{!25}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", linkageName: "_ZN1C1aE", scope: !2, file: !3, line: 8, type: !5, isLocal: false, isDefinition: true, declaration: !8)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 18.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4, globals: !14, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!3 = !DIFile(filename: "main.cpp", directory: "/tmp", checksumkind: CSK_MD5, checksum: "b2547f7df8f54777c012dbfbd626f155")
!4 = !{!5, !6}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C", file: !3, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTS1C")
!7 = !{!8, !9, !12}
!8 = !DIDerivedType(tag: DW_TAG_variable, name: "a", scope: !6, file: !3, line: 3, baseType: !5, flags: DIFlagStaticMember)
!9 = !DIDerivedType(tag: DW_TAG_variable, name: "b", scope: !6, file: !3, line: 4, baseType: !10, flags: DIFlagStaticMember)
!10 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!11 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!12 = !DIDerivedType(tag: DW_TAG_variable, name: "c", scope: !6, file: !3, line: 5, baseType: !13, flags: DIFlagStaticMember)
!13 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !5)
!14 = !{!0, !15, !17}
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression(DW_OP_constu, 1, DW_OP_stack_value))
!16 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 4, type: !10, isLocal: true, isDefinition: true, declaration: !9)
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression(DW_OP_constu, 15, DW_OP_stack_value))
!18 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !3, line: 5, type: !13, isLocal: true, isDefinition: true, declaration: !12)
!19 = !{i32 7, !"Dwarf Version", i32 5}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !{i32 1, !"wchar_size", i32 4}
!22 = !{i32 8, !"PIC Level", i32 2}
!23 = !{i32 7, !"uwtable", i32 1}
!24 = !{i32 7, !"frame-pointer", i32 1}
!25 = !{!"clang version 18.0.0"}
!26 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 10, type: !27, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !29)
!27 = !DISubroutineType(types: !28)
!28 = !{!5}
!29 = !{}
!30 = !DILocalVariable(name: "instance_C", scope: !26, file: !3, line: 12, type: !6)
!31 = !DILocation(line: 12, column: 5, scope: !26)
!32 = !DILocation(line: 13, column: 10, scope: !26)
!33 = !DILocation(line: 13, column: 15, scope: !26)
!34 = !DILocation(line: 13, column: 27, scope: !26)
!35 = !DILocation(line: 13, column: 3, scope: !26)

; CHECK:      .debug_info contents:
; CHECK:      DW_TAG_variable
; CHECK-NEXT: DW_AT_specification {{.*}} "a"
; CHECK:      DW_TAG_structure_type
; CHECK:      DW_AT_name {{.*}} "C"
; CHECK:      DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "a"
; CHECK:      DW_AT_external
; CHECK:      DW_AT_declaration
; CHECK:      DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "b"
; CHECK:      DW_AT_external
; CHECK:      DW_AT_declaration
; CHECK:      DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} "c"
; CHECK:      DW_AT_external
; CHECK:      DW_AT_declaration
; CHECK:      NULL
; CHECK:      DW_TAG_variable
; CHECK-NEXT: DW_AT_specification {{.*}} "b"
; CHECK-NEXT: DW_AT_const_value
; CHECK:      DW_TAG_variable
; CHECK-NEXT: DW_AT_specification {{.*}} "c"

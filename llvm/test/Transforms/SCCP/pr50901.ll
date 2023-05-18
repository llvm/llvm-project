; RUN: opt -passes=ipsccp -S -o - < %s | FileCheck %s

; Global variables g_11, g_22, g_33, g_44, g_55, g_66 and g_77
; are not visible in the debugger.

;  1	int       g_1 = -4;
;  2	float     g_2 = 4.44;
;  3	char      g_3 = 'a';
;  4	unsigned  g_4 = 4;
;  5	bool      g_5 = true;
;  6	int      *g_6 = nullptr;
;  7	float    *g_7 = nullptr;
;  8
;  9	static int       g_11 = -5;
; 10	static float     g_22 = 5.55;
; 11	static char      g_33 = 'b';
; 12	static unsigned  g_44 = 5;
; 13	static bool      g_55 = true;
; 14	static int      *g_66 = nullptr;
; 15	static float    *g_77 = (float *)(55 + 15);
; 16
; 17	void bar() {
; 18	  g_1 = g_11;
; 19	  g_2 = g_22;
; 20	  g_3 = g_33;
; 21	  g_4 = g_44;
; 22	  g_5 = g_55;
; 23	  g_6 = g_66;
; 24	  g_7 = g_77;
; 25	}
; 26
; 27	int main() {
; 28	  {
; 29	    bar();
; 30	  }
; 31	}

; CHECK: ![[G1:[0-9]+]] = !DIGlobalVariableExpression(var: ![[DBG1:[0-9]+]], expr: !DIExpression(DW_OP_constu, 18446744073709551611, DW_OP_stack_value))
; CHECK-DAG: ![[DBG1]] = distinct !DIGlobalVariable(name: "g_11", {{.*}}
; CHECK: ![[G2:[0-9]+]] = !DIGlobalVariableExpression(var: ![[DBG2:[0-9]+]], expr: !DIExpression(DW_OP_constu, 1085381018, DW_OP_stack_value))
; CHECK-DAG: ![[DBG2]] = distinct !DIGlobalVariable(name: "g_22", {{.*}}
; CHECK: ![[G3:[0-9]+]] = !DIGlobalVariableExpression(var: ![[DBG3:[0-9]+]], expr: !DIExpression(DW_OP_constu, 98, DW_OP_stack_value))
; CHECK-DAG: ![[DBG3]] = distinct !DIGlobalVariable(name: "g_33", {{.*}}
; CHECK: ![[G4:[0-9]+]] = !DIGlobalVariableExpression(var: ![[DBG4:[0-9]+]], expr: !DIExpression(DW_OP_constu, 5, DW_OP_stack_value))
; CHECK-DAG: ![[DBG4]] = distinct !DIGlobalVariable(name: "g_44", {{.*}}
; CHECK: ![[G5:[0-9]+]] = !DIGlobalVariableExpression(var: ![[DBG5:[0-9]+]], expr: !DIExpression(DW_OP_constu, 1, DW_OP_stack_value))
; CHECK-DAG: ![[DBG5]] = distinct !DIGlobalVariable(name: "g_55", {{.*}}
; CHECK: ![[G6:[0-9]+]] = !DIGlobalVariableExpression(var: ![[DBG6:[0-9]+]], expr: !DIExpression(DW_OP_constu, 0, DW_OP_stack_value))
; CHECK-DAG: ![[DBG6]] = distinct !DIGlobalVariable(name: "g_66", {{.*}}
; CHECK: ![[G7:[0-9]+]] = !DIGlobalVariableExpression(var: ![[DBG7:[0-9]+]], expr: !DIExpression(DW_OP_constu, 70, DW_OP_stack_value))
; CHECK-DAG: ![[DBG7]] = distinct !DIGlobalVariable(name: "g_77", {{.*}}
; CHECK:     = !DIGlobalVariableExpression(var: ![[DBG_FLOAT_UNDEF:.+]], expr: !DIExpression())
; CHECK-DAG: ![[DBG_FLOAT_UNDEF]]  = distinct !DIGlobalVariable(name: "g_float_undef"

@g_1 = dso_local global i32 -4, align 4, !dbg !0
@g_2 = dso_local global float 0x4011C28F60000000, align 4, !dbg !8
@g_3 = dso_local global i8 97, align 1, !dbg !10
@g_4 = dso_local global i32 4, align 4, !dbg !13
@g_5 = dso_local global i8 1, align 1, !dbg !16
@g_6 = dso_local global ptr null, align 8, !dbg !19
@g_7 = dso_local global ptr null, align 8, !dbg !23
@_ZL4g_11 = internal global i32 -5, align 4, !dbg !25
@_ZL4g_22 = internal global float 0x4016333340000000, align 4, !dbg !27
@_ZL4g_33 = internal global i8 98, align 1, !dbg !29
@_ZL4g_44 = internal global i32 5, align 4, !dbg !31
@_ZL4g_55 = internal global i8 1, align 1, !dbg !33
@_ZL4g_66 = internal global ptr null, align 8, !dbg !35
@_ZL4g_77 = internal global ptr inttoptr (i64 70 to ptr), align 8, !dbg !37
@g_float_undef = internal global float undef, align 4, !dbg !83

define dso_local void @_Z3barv() !dbg !46 {
entry:
  %0 = load i32, ptr @_ZL4g_11, align 4, !dbg !59
  store i32 %0, ptr @g_1, align 4, !dbg !59
  %1 = load float, ptr @_ZL4g_22, align 4, !dbg !59
  store float %1, ptr @g_2, align 4, !dbg !59
  %2 = load i8, ptr @_ZL4g_33, align 1, !dbg !59
  store i8 %2, ptr @g_3, align 1, !dbg !59
  %3 = load i32, ptr @_ZL4g_44, align 4, !dbg !59
  store i32 %3, ptr @g_4, align 4, !dbg !59
  %4 = load i8, ptr @_ZL4g_55, align 1, !dbg !59
  %tobool = trunc i8 %4 to i1, !dbg !59
  %frombool = zext i1 %tobool to i8, !dbg !59
  store i8 %frombool, ptr @g_5, align 1, !dbg !59
  %5 = load ptr, ptr @_ZL4g_66, align 8, !dbg !59
  store ptr %5, ptr @g_6, align 8, !dbg !59
  %6 = load ptr, ptr @_ZL4g_77, align 8, !dbg !59
  store ptr %6, ptr @g_7, align 8, !dbg !59
  %l = load float, ptr @g_float_undef, align 8, !dbg !59
  store float %l, ptr @g_2, align 8, !dbg !59
  ret void, !dbg !59
}

define dso_local noundef i32 @main() !dbg !77 {
entry:
  call void @_Z3barv(), !dbg !80
  ret i32 0, !dbg !82
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!39, !40, !41, !42, !43, !44}
!llvm.ident = !{!45}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g_1", scope: !2, file: !3, line: 1, type: !22, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 18.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4, globals: !7, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "")
!4 = !{!5}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!6 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!7 = !{!0, !8, !10, !13, !16, !19, !23, !25, !27, !29, !31, !33, !35, !37, !83}
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "g_2", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "g_3", scope: !2, file: !3, line: 3, type: !12, isLocal: false, isDefinition: true)
!12 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "g_4", scope: !2, file: !3, line: 4, type: !15, isLocal: false, isDefinition: true)
!15 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!16 = !DIGlobalVariableExpression(var: !17, expr: !DIExpression())
!17 = distinct !DIGlobalVariable(name: "g_5", scope: !2, file: !3, line: 5, type: !18, isLocal: false, isDefinition: true)
!18 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!19 = !DIGlobalVariableExpression(var: !20, expr: !DIExpression())
!20 = distinct !DIGlobalVariable(name: "g_6", scope: !2, file: !3, line: 6, type: !21, isLocal: false, isDefinition: true)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!23 = !DIGlobalVariableExpression(var: !24, expr: !DIExpression())
!24 = distinct !DIGlobalVariable(name: "g_7", scope: !2, file: !3, line: 7, type: !5, isLocal: false, isDefinition: true)
!25 = !DIGlobalVariableExpression(var: !26, expr: !DIExpression())
!26 = distinct !DIGlobalVariable(name: "g_11", linkageName: "_ZL4g_11", scope: !2, file: !3, line: 9, type: !22, isLocal: true, isDefinition: true)
!27 = !DIGlobalVariableExpression(var: !28, expr: !DIExpression())
!28 = distinct !DIGlobalVariable(name: "g_22", linkageName: "_ZL4g_22", scope: !2, file: !3, line: 10, type: !6, isLocal: true, isDefinition: true)
!29 = !DIGlobalVariableExpression(var: !30, expr: !DIExpression())
!30 = distinct !DIGlobalVariable(name: "g_33", linkageName: "_ZL4g_33", scope: !2, file: !3, line: 11, type: !12, isLocal: true, isDefinition: true)
!31 = !DIGlobalVariableExpression(var: !32, expr: !DIExpression())
!32 = distinct !DIGlobalVariable(name: "g_44", linkageName: "_ZL4g_44", scope: !2, file: !3, line: 12, type: !15, isLocal: true, isDefinition: true)
!33 = !DIGlobalVariableExpression(var: !34, expr: !DIExpression())
!34 = distinct !DIGlobalVariable(name: "g_55", linkageName: "_ZL4g_55", scope: !2, file: !3, line: 13, type: !18, isLocal: true, isDefinition: true)
!35 = !DIGlobalVariableExpression(var: !36, expr: !DIExpression())
!36 = distinct !DIGlobalVariable(name: "g_66", linkageName: "_ZL4g_66", scope: !2, file: !3, line: 14, type: !21, isLocal: true, isDefinition: true)
!37 = !DIGlobalVariableExpression(var: !38, expr: !DIExpression())
!38 = distinct !DIGlobalVariable(name: "g_77", linkageName: "_ZL4g_77", scope: !2, file: !3, line: 15, type: !5, isLocal: true, isDefinition: true)
!39 = !{i32 7, !"Dwarf Version", i32 5}
!40 = !{i32 2, !"Debug Info Version", i32 3}
!41 = !{i32 1, !"wchar_size", i32 4}
!42 = !{i32 8, !"PIC Level", i32 2}
!43 = !{i32 7, !"PIE Level", i32 2}
!44 = !{i32 7, !"uwtable", i32 2}
!45 = !{!"clang version 18.0.0"}
!46 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !3, file: !3, line: 17, type: !47, scopeLine: 17, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!47 = !DISubroutineType(types: !48)
!48 = !{null}
!59 = !DILocation(line: 20, column: 9, scope: !46)
!77 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 27, type: !78, scopeLine: 27, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!78 = !DISubroutineType(types: !79)
!79 = !{!22}
!80 = !DILocation(line: 29, column: 5, scope: !81)
!81 = distinct !DILexicalBlock(scope: !77, file: !3, line: 28, column: 3)
!82 = !DILocation(line: 31, column: 1, scope: !77)
!83 = !DIGlobalVariableExpression(var: !84, expr: !DIExpression())
!84 = distinct !DIGlobalVariable(name: "g_float_undef", linkageName: "g_float_undef", scope: !2, file: !3, line: 15, type: !6, isLocal: true, isDefinition: true)

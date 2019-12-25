; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s
;
; This test verifies global variables are emitted within the correct scope.
;
; -- global_visibility.cpp ----------------------------------------------------
;  1	
;  2	int global_int = 0;
;  3	
;  4	template <typename T> struct A {
;  5	  static T comdat_int;
;  6	  static T set(T value) { T r = comdat_int; comdat_int = value; return r; };
;  7	};
;  8	
;  9	template <typename T> T A<T>::comdat_int = 42;
; 10	
; 11	void foo() {
; 12	  static int local_int = 1;
; 13	  {
; 14	    static int nested_int = 2;
; 15	    local_int = nested_int;
; 16	  }
; 17	  local_int = A<int>::set(42);
; 18	}
; 19	
; 20	void bar() {
; 21	  static int local_int = 3;
; 22	  {
; 23	    static int nested_int = 4;
; 24	    local_int = nested_int;
; 25	  }
; 26	  local_int = A<unsigned>::set(42);
; 27	}
; 28	
; -----------------------------------------------------------------------------
;
; $ clang -S -emit-llvm -g -gcodeview global_visibility.cpp
;
; NOTE: The scope for both DIGlobalVariable's named "nested_int" should refer
;       to the appropriate DILexicalBlock, not a DISubprogram.
;

; CHECK: CodeViewDebugInfo [
; CHECK:   Section: .debug$S (9)

; CHECK:   Subsection [
; CHECK:     SubSectionType: Symbols (0xF1)
; CHECK:     GlobalProcIdSym {
; CHECK:       Kind: S_GPROC32_ID (0x1147)
; CHECK:       DisplayName: foo
; CHECK:       LinkageName: ?foo@@YAXXZ
; CHECK:     }
; CHECK:     DataSym {
; CHECK:       Kind: S_LDATA32 (0x110C)
; CHECK:       DisplayName: local_int
; CHECK:       LinkageName: ?local_int@?1??foo@@YAXXZ@4HA
; CHECK:     }
; CHECK:     BlockSym {
; CHECK:       Kind: S_BLOCK32 (0x1103)
; CHECK:     }
; CHECK:     DataSym {
; CHECK:       Kind: S_LDATA32 (0x110C)
; CHECK:       DisplayName: nested_int
; CHECK:       LinkageName: ?nested_int@?1??foo@@YAXXZ@4HA
; CHECK:     }
; CHECK:     ScopeEndSym {
; CHECK:       Kind: S_END (0x6)
; CHECK:     }
; CHECK:     ProcEnd {
; CHECK:       Kind: S_PROC_ID_END (0x114F)
; CHECK:     }
; CHECK:   ]
; CHECK:   Subsection [
; CHECK:     SubSectionType: Symbols (0xF1)
; CHECK:     GlobalProcIdSym {
; CHECK:       Kind: S_GPROC32_ID (0x1147)
; CHECK:       DisplayName: bar
; CHECK:       LinkageName: ?bar@@YAXXZ
; CHECK:     }
; CHECK:     DataSym {
; CHECK:       Kind: S_LDATA32 (0x110C)
; CHECK:       DisplayName: local_int
; CHECK:       LinkageName: ?local_int@?1??bar@@YAXXZ@4HA
; CHECK:     }
; CHECK:     BlockSym {
; CHECK:       Kind: S_BLOCK32 (0x1103)
; CHECK:     }
; CHECK:     DataSym {
; CHECK:       Kind: S_LDATA32 (0x110C)
; CHECK:       DisplayName: nested_int
; CHECK:       LinkageName: ?nested_int@?1??bar@@YAXXZ@4HA
; CHECK:     }
; CHECK:     ScopeEndSym {
; CHECK:       Kind: S_END (0x6)
; CHECK:     }
; CHECK:     ProcEnd {
; CHECK:       Kind: S_PROC_ID_END (0x114F)
; CHECK:     }
; CHECK:   ]
; CHECK:   Subsection [
; CHECK:     SubSectionType: Symbols (0xF1)
; CHECK:     GlobalData {
; CHECK:       Kind: S_GDATA32 (0x110D)
; CHECK:       DisplayName: global_int
; CHECK:       LinkageName: ?global_int@@3HA
; CHECK:     }
; CHECK:   ]
; CHECK: ]
; CHECK: CodeViewDebugInfo [
; CHECK:   Section: .debug$S (16)
; CHECK:   Subsection [
; CHECK:     SubSectionType: Symbols (0xF1)
; CHECK:     GlobalData {
; CHECK:       Kind: S_GDATA32 (0x110D)
; CHECK:       DisplayName: comdat_int
; CHECK:       LinkageName: ?comdat_int@?$A@H@@2HA
; CHECK:     }
; CHECK:   ]
; CHECK: ]
; CHECK: CodeViewDebugInfo [
; CHECK:   Section: .debug$S (17)
; CHECK:   Subsection [
; CHECK:     SubSectionType: Symbols (0xF1)
; CHECK:     GlobalData {
; CHECK:       Kind: S_GDATA32 (0x110D)
; CHECK:       DisplayName: comdat_int
; CHECK:       LinkageName: ?comdat_int@?$A@I@@2IA
; CHECK:     }
; CHECK:   ]
; CHECK: ]
;

; ModuleID = 'global_visibility.cpp'
source_filename = "global_visibility.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.15.26730"

$"?set@?$A@H@@SAHH@Z" = comdat any

$"?set@?$A@I@@SAII@Z" = comdat any

$"?comdat_int@?$A@H@@2HA" = comdat any

$"?comdat_int@?$A@I@@2IA" = comdat any

@"?global_int@@3HA" = dso_local global i32 0, align 4, !dbg !0
@"?local_int@?1??foo@@YAXXZ@4HA" = internal global i32 1, align 4, !dbg !6
@"?nested_int@?1??foo@@YAXXZ@4HA" = internal global i32 2, align 4, !dbg !12
@"?local_int@?1??bar@@YAXXZ@4HA" = internal global i32 3, align 4, !dbg !14
@"?nested_int@?1??bar@@YAXXZ@4HA" = internal global i32 4, align 4, !dbg !17
@"?comdat_int@?$A@H@@2HA" = linkonce_odr dso_local global i32 42, comdat, align 4, !dbg !19
@"?comdat_int@?$A@I@@2IA" = linkonce_odr dso_local global i32 42, comdat, align 4, !dbg !29

; Function Attrs: noinline optnone uwtable
define dso_local void @"?foo@@YAXXZ"() #0 !dbg !8 {
entry:
  %0 = load i32, i32* @"?nested_int@?1??foo@@YAXXZ@4HA", align 4, !dbg !45
  store i32 %0, i32* @"?local_int@?1??foo@@YAXXZ@4HA", align 4, !dbg !45
  %call = call i32 @"?set@?$A@H@@SAHH@Z"(i32 42), !dbg !47
  store i32 %call, i32* @"?local_int@?1??foo@@YAXXZ@4HA", align 4, !dbg !47
  ret void, !dbg !48
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local i32 @"?set@?$A@H@@SAHH@Z"(i32 %value) #1 comdat align 2 !dbg !49 {
entry:
  %value.addr = alloca i32, align 4
  %r = alloca i32, align 4
  store i32 %value, i32* %value.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %value.addr, metadata !50, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.declare(metadata i32* %r, metadata !52, metadata !DIExpression()), !dbg !51
  %0 = load i32, i32* @"?comdat_int@?$A@H@@2HA", align 4, !dbg !51
  store i32 %0, i32* %r, align 4, !dbg !51
  %1 = load i32, i32* %value.addr, align 4, !dbg !51
  store i32 %1, i32* @"?comdat_int@?$A@H@@2HA", align 4, !dbg !51
  %2 = load i32, i32* %r, align 4, !dbg !51
  ret i32 %2, !dbg !51
}

; Function Attrs: noinline optnone uwtable
define dso_local void @"?bar@@YAXXZ"() #0 !dbg !16 {
entry:
  %0 = load i32, i32* @"?nested_int@?1??bar@@YAXXZ@4HA", align 4, !dbg !53
  store i32 %0, i32* @"?local_int@?1??bar@@YAXXZ@4HA", align 4, !dbg !53
  %call = call i32 @"?set@?$A@I@@SAII@Z"(i32 42), !dbg !55
  store i32 %call, i32* @"?local_int@?1??bar@@YAXXZ@4HA", align 4, !dbg !55
  ret void, !dbg !56
}

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local i32 @"?set@?$A@I@@SAII@Z"(i32 %value) #1 comdat align 2 !dbg !57 {
entry:
  %value.addr = alloca i32, align 4
  %r = alloca i32, align 4
  store i32 %value, i32* %value.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %value.addr, metadata !58, metadata !DIExpression()), !dbg !59
  call void @llvm.dbg.declare(metadata i32* %r, metadata !60, metadata !DIExpression()), !dbg !59
  %0 = load i32, i32* @"?comdat_int@?$A@I@@2IA", align 4, !dbg !59
  store i32 %0, i32* %r, align 4, !dbg !59
  %1 = load i32, i32* %value.addr, align 4, !dbg !59
  store i32 %1, i32* @"?comdat_int@?$A@I@@2IA", align 4, !dbg !59
  %2 = load i32, i32* %r, align 4, !dbg !59
  ret i32 %2, !dbg !59
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

attributes #0 = { noinline optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!40, !41, !42, !43}
!llvm.ident = !{!44}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "global_int", linkageName: "?global_int@@3HA", scope: !2, file: !3, line: 2, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 8.0.0 (trunk)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "global_visibility.cpp", directory: "C:\5Cpath\5Cto\5Cdirectory", checksumkind: CSK_MD5, checksum: "f59b9e5de12391471b1a61888cb68a3e")
!4 = !{}
!5 = !{!0, !6, !12, !14, !17, !19, !29}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "local_int", scope: !8, file: !3, line: 12, type: !11, isLocal: true, isDefinition: true)
!8 = distinct !DISubprogram(name: "foo", linkageName: "?foo@@YAXXZ", scope: !3, file: !3, line: 11, type: !9, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression())
!13 = distinct !DIGlobalVariable(name: "nested_int", scope: !46, file: !3, line: 14, type: !11, isLocal: true, isDefinition: true)
!14 = !DIGlobalVariableExpression(var: !15, expr: !DIExpression())
!15 = distinct !DIGlobalVariable(name: "local_int", scope: !16, file: !3, line: 21, type: !11, isLocal: true, isDefinition: true)
!16 = distinct !DISubprogram(name: "bar", linkageName: "?bar@@YAXXZ", scope: !3, file: !3, line: 20, type: !9, scopeLine: 20, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = distinct !DIGlobalVariable(name: "nested_int", scope: !54, file: !3, line: 23, type: !11, isLocal: true, isDefinition: true)
!19 = !DIGlobalVariableExpression(var: !20, expr: !DIExpression())
!20 = distinct !DIGlobalVariable(name: "comdat_int", linkageName: "?comdat_int@?$A@H@@2HA", scope: !2, file: !3, line: 9, type: !11, isLocal: false, isDefinition: true, declaration: !21)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "comdat_int", scope: !22, file: !3, line: 5, baseType: !11, flags: DIFlagStaticMember)
!22 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A<int>", file: !3, line: 4, size: 8, flags: DIFlagTypePassByValue, elements: !23, templateParams: !27, identifier: ".?AU?$A@H@@")
!23 = !{!21, !24}
!24 = !DISubprogram(name: "set", linkageName: "?set@?$A@H@@SAHH@Z", scope: !22, file: !3, line: 6, type: !25, scopeLine: 6, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!25 = !DISubroutineType(types: !26)
!26 = !{!11, !11}
!27 = !{!28}
!28 = !DITemplateTypeParameter(name: "T", type: !11)
!29 = !DIGlobalVariableExpression(var: !30, expr: !DIExpression())
!30 = distinct !DIGlobalVariable(name: "comdat_int", linkageName: "?comdat_int@?$A@I@@2IA", scope: !2, file: !3, line: 9, type: !31, isLocal: false, isDefinition: true, declaration: !32)
!31 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!32 = !DIDerivedType(tag: DW_TAG_member, name: "comdat_int", scope: !33, file: !3, line: 5, baseType: !31, flags: DIFlagStaticMember)
!33 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A<unsigned int>", file: !3, line: 4, size: 8, flags: DIFlagTypePassByValue, elements: !34, templateParams: !38, identifier: ".?AU?$A@I@@")
!34 = !{!32, !35}
!35 = !DISubprogram(name: "set", linkageName: "?set@?$A@I@@SAII@Z", scope: !33, file: !3, line: 6, type: !36, scopeLine: 6, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!36 = !DISubroutineType(types: !37)
!37 = !{!31, !31}
!38 = !{!39}
!39 = !DITemplateTypeParameter(name: "T", type: !31)
!40 = !{i32 2, !"CodeView", i32 1}
!41 = !{i32 2, !"Debug Info Version", i32 3}
!42 = !{i32 1, !"wchar_size", i32 2}
!43 = !{i32 7, !"PIC Level", i32 2}
!44 = !{!"clang version 8.0.0 (trunk)"}
!45 = !DILocation(line: 15, scope: !46)
!46 = distinct !DILexicalBlock(scope: !8, file: !3, line: 13)
!47 = !DILocation(line: 17, scope: !8)
!48 = !DILocation(line: 18, scope: !8)
!49 = distinct !DISubprogram(name: "set", linkageName: "?set@?$A@H@@SAHH@Z", scope: !22, file: !3, line: 6, type: !25, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !24, retainedNodes: !4)
!50 = !DILocalVariable(name: "value", arg: 1, scope: !49, file: !3, line: 6, type: !11)
!51 = !DILocation(line: 6, scope: !49)
!52 = !DILocalVariable(name: "r", scope: !49, file: !3, line: 6, type: !11)
!53 = !DILocation(line: 24, scope: !54)
!54 = distinct !DILexicalBlock(scope: !16, file: !3, line: 22)
!55 = !DILocation(line: 26, scope: !16)
!56 = !DILocation(line: 27, scope: !16)
!57 = distinct !DISubprogram(name: "set", linkageName: "?set@?$A@I@@SAII@Z", scope: !33, file: !3, line: 6, type: !36, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !35, retainedNodes: !4)
!58 = !DILocalVariable(name: "value", arg: 1, scope: !57, file: !3, line: 6, type: !31)
!59 = !DILocation(line: 6, scope: !57)
!60 = !DILocalVariable(name: "r", scope: !57, file: !3, line: 6, type: !31)

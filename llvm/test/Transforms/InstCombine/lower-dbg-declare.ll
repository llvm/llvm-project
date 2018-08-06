; RUN: opt -instcombine < %s -S | FileCheck %s

; This tests dbg.declare lowering for CallInst users of an alloca. The
; resulting dbg.value expressions should add a deref to the declare's expression.

; Hand-reduced from this example (-g -Og -fsanitize=address):

;   static volatile int sink;
;   struct OneElementVector {
;     int Element;
;     OneElementVector(int Element) : Element(Element) { sink = Element; }
;     bool empty() const { return false; }
;   };
;   using container = OneElementVector;
;   static void escape(container &c) { sink = c.Element; }
;   int main() {
;     container d1 = {42};
;     while (!d1.empty())
;       escape(d1);
;     return 0;
;   }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

%struct.OneElementVector = type { i32 }

define i1 @escape(%struct.OneElementVector* %d1) {
  ret i1 false
}

; CHECK-LABEL: @main
define i32 @main() !dbg !15 {
entry:
  %d1 = alloca %struct.OneElementVector, align 4
  %0 = bitcast %struct.OneElementVector* %d1 to i8*, !dbg !34

; CHECK: dbg.value(metadata %struct.OneElementVector* [[var:%.*]], metadata !DIExpression(DW_OP_deref))
; CHECK-NEXT: call i1 @escape
  call void @llvm.dbg.declare(metadata %struct.OneElementVector* %d1, metadata !19, metadata !DIExpression()), !dbg !35
  call i1 @escape(%struct.OneElementVector* %d1)
  br label %while.cond, !dbg !37

while.cond:                                       ; preds = %while.body, %entry
; CHECK: dbg.value(metadata %struct.OneElementVector* [[var]], metadata !DIExpression(DW_OP_deref))
; CHECK-NEXT: call i1 @escape
  %call = call i1 @escape(%struct.OneElementVector* %d1), !dbg !38
  %lnot = xor i1 %call, true, !dbg !39
  br i1 %lnot, label %while.body, label %while.end, !dbg !37

while.body:                                       ; preds = %while.cond
; CHECK: dbg.value(metadata %struct.OneElementVector* [[var]], metadata !DIExpression(DW_OP_deref))
; CHECK-NEXT: call i1 @escape
  call i1 @escape(%struct.OneElementVector* %d1)
  br label %while.cond, !dbg !37, !llvm.loop !42

while.end:                                        ; preds = %while.cond
  ret i32 0, !dbg !45
}

; CHECK-LABEL: @main2
define i32 @main2() {
entry:
  %d1 = alloca %struct.OneElementVector, align 4
  %0 = bitcast %struct.OneElementVector* %d1 to i8*, !dbg !34

; CHECK: dbg.value(metadata %struct.OneElementVector* [[var:%.*]], metadata !DIExpression(DW_OP_lit0, DW_OP_mul, DW_OP_deref))
; CHECK-NEXT: call i1 @escape
  call void @llvm.dbg.declare(metadata %struct.OneElementVector* %d1, metadata !19, metadata !DIExpression(DW_OP_lit0, DW_OP_mul)), !dbg !35
  call i1 @escape(%struct.OneElementVector* %d1)
  br label %while.cond, !dbg !37

while.cond:                                       ; preds = %while.body, %entry
; CHECK: dbg.value(metadata %struct.OneElementVector* [[var]], metadata !DIExpression(DW_OP_lit0, DW_OP_mul, DW_OP_deref))
; CHECK-NEXT: call i1 @escape
  %call = call i1 @escape(%struct.OneElementVector* %d1), !dbg !38
  %lnot = xor i1 %call, true, !dbg !39
  br i1 %lnot, label %while.body, label %while.end, !dbg !37

while.body:                                       ; preds = %while.cond
; CHECK: dbg.value(metadata %struct.OneElementVector* [[var]], metadata !DIExpression(DW_OP_lit0, DW_OP_mul, DW_OP_deref))
; CHECK-NEXT: call i1 @escape
  call i1 @escape(%struct.OneElementVector* %d1)
  br label %while.cond, !dbg !37, !llvm.loop !42

while.end:                                        ; preds = %while.cond
  ret i32 0, !dbg !45
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.asan.globals = !{!8}
!llvm.module.flags = !{!10, !11, !12, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "sink", linkageName: "_ZL4sink", scope: !2, file: !3, line: 1, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 (trunk 337207) (llvm/trunk 337204)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "test.cc", directory: "/Users/vsk/src/builds/llvm.org-master-RA")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{}
!9 = !{!"test.cc", i32 1, i32 21}
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 7, !"PIC Level", i32 2}
!14 = !{!"clang version 7.0.0 (trunk 337207) (llvm/trunk 337204)"}
!15 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 18, type: !16, isLocal: false, isDefinition: true, scopeLine: 18, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !18)
!16 = !DISubroutineType(types: !17)
!17 = !{!7}
!18 = !{!19}
!19 = !DILocalVariable(name: "d1", scope: !15, file: !3, line: 21, type: !20)
!20 = !DIDerivedType(tag: DW_TAG_typedef, name: "container", file: !3, line: 12, baseType: !21)
!21 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "OneElementVector", file: !3, line: 3, size: 32, flags: DIFlagTypePassByValue, elements: !22, identifier: "_ZTS16OneElementVector")
!22 = !{!23, !24, !28}
!23 = !DIDerivedType(tag: DW_TAG_member, name: "Element", scope: !21, file: !3, line: 4, baseType: !7, size: 32)
!24 = !DISubprogram(name: "OneElementVector", scope: !21, file: !3, line: 6, type: !25, isLocal: false, isDefinition: false, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: true)
!25 = !DISubroutineType(types: !26)
!26 = !{null, !27, !7}
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!28 = !DISubprogram(name: "empty", linkageName: "_ZNK16OneElementVector5emptyEv", scope: !21, file: !3, line: 8, type: !29, isLocal: false, isDefinition: false, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true)
!29 = !DISubroutineType(types: !30)
!30 = !{!31, !32}
!31 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !33, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!33 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !21)
!34 = !DILocation(line: 21, column: 3, scope: !15)
!35 = !DILocation(line: 21, column: 13, scope: !15)
!36 = !DILocation(line: 21, column: 18, scope: !15)
!37 = !DILocation(line: 22, column: 3, scope: !15)
!38 = !DILocation(line: 22, column: 14, scope: !15)
!39 = !DILocation(line: 22, column: 10, scope: !15)
!40 = !DILocation(line: 23, column: 5, scope: !41)
!41 = distinct !DILexicalBlock(scope: !15, file: !3, line: 22, column: 23)
!42 = distinct !{!42, !37, !43}
!43 = !DILocation(line: 24, column: 3, scope: !15)
!44 = !DILocation(line: 26, column: 1, scope: !15)
!45 = !DILocation(line: 25, column: 3, scope: !15)
!46 = distinct !DISubprogram(name: "OneElementVector", linkageName: "_ZN16OneElementVectorC1Ei", scope: !21, file: !3, line: 6, type: !25, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, declaration: !24, retainedNodes: !47)
!47 = !{!48, !50}
!48 = !DILocalVariable(name: "this", arg: 1, scope: !46, type: !49, flags: DIFlagArtificial | DIFlagObjectPointer)
!49 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64)
!50 = !DILocalVariable(name: "Element", arg: 2, scope: !46, file: !3, line: 6, type: !7)
!51 = !DILocation(line: 0, scope: !46)
!52 = !DILocation(line: 6, column: 24, scope: !46)
!53 = !DILocation(line: 6, column: 52, scope: !46)
!54 = !DILocation(line: 6, column: 70, scope: !46)
!55 = distinct !DISubprogram(name: "empty", linkageName: "_ZNK16OneElementVector5emptyEv", scope: !21, file: !3, line: 8, type: !29, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, unit: !2, declaration: !28, retainedNodes: !56)
!56 = !{!57}
!57 = !DILocalVariable(name: "this", arg: 1, scope: !55, type: !58, flags: DIFlagArtificial | DIFlagObjectPointer)
!58 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !33, size: 64)
!59 = !DILocation(line: 0, scope: !55)
!60 = !DILocation(line: 8, column: 24, scope: !55)
!61 = distinct !DISubprogram(name: "escape", linkageName: "_ZL6escapeR16OneElementVector", scope: !3, file: !3, line: 14, type: !62, isLocal: true, isDefinition: true, scopeLine: 14, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !65)
!62 = !DISubroutineType(types: !63)
!63 = !{null, !64}
!64 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !20, size: 64)
!65 = !{!66}
!66 = !DILocalVariable(name: "c", arg: 1, scope: !61, file: !3, line: 14, type: !64)
!67 = !DILocation(line: 14, column: 31, scope: !61)
!68 = !DILocation(line: 15, column: 12, scope: !61)
!69 = !{!70, !71, i64 0}
!70 = !{!"_ZTS16OneElementVector", !71, i64 0}
!71 = !{!"int", !72, i64 0}
!72 = !{!"omnipotent char", !73, i64 0}
!73 = !{!"Simple C++ TBAA"}
!74 = !DILocation(line: 15, column: 8, scope: !61)
!75 = !{!71, !71, i64 0}
!76 = !DILocation(line: 16, column: 1, scope: !61)
!77 = distinct !DISubprogram(name: "OneElementVector", linkageName: "_ZN16OneElementVectorC2Ei", scope: !21, file: !3, line: 6, type: !25, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, declaration: !24, retainedNodes: !78)
!78 = !{!79, !80}
!79 = !DILocalVariable(name: "this", arg: 1, scope: !77, type: !49, flags: DIFlagArtificial | DIFlagObjectPointer)
!80 = !DILocalVariable(name: "Element", arg: 2, scope: !77, file: !3, line: 6, type: !7)
!81 = !DILocation(line: 0, scope: !77)
!82 = !DILocation(line: 6, column: 24, scope: !77)
!83 = !DILocation(line: 6, column: 35, scope: !77)
!84 = !DILocation(line: 6, column: 59, scope: !85)
!85 = distinct !DILexicalBlock(scope: !77, file: !3, line: 6, column: 52)
!86 = !DILocation(line: 6, column: 70, scope: !77)

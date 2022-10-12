; RUN: opt %s -simplifycfg -verify -S -o - | FileCheck %s
;
; Verify that SimplifyCFG does not invalidate operands for
; llvm.dbg.value intrinsics.
;
; Reduced from the following example, compiled with
; -O1 -g -fno-exceptions -fno-discard-value-names:
;
; void test() {
;   bool a = false;
;   bool b = false;
;   bool c = false;
;   [&a, &b, &c] { b = true; }();
;
;   if (a || b || c) {
;     __builtin_debugtrap();
;   }
; }

; CHECK: _Z4testv
; CHECK-NOT: llvm.dbg.value(metadata !
; CHECK: ret void

%class.anon = type { ptr, ptr, ptr }

; Function Attrs: nounwind uwtable
define dso_local void @_Z4testv() local_unnamed_addr !dbg !7 {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  %c = alloca i8, align 1
  %ref.tmp = alloca %class.anon, align 8
  call void @llvm.dbg.value(metadata ptr %a, metadata !12, metadata !DIExpression(DW_OP_deref)), !dbg !16
  call void @llvm.dbg.value(metadata i8 0, metadata !12, metadata !DIExpression()), !dbg !16
  store i8 0, ptr %a, align 1, !dbg !16
  call void @llvm.dbg.value(metadata ptr %b, metadata !14, metadata !DIExpression(DW_OP_deref)), !dbg !16
  call void @llvm.dbg.value(metadata i8 0, metadata !14, metadata !DIExpression()), !dbg !16
  store i8 0, ptr %b, align 1, !dbg !16
  call void @llvm.dbg.value(metadata ptr %c, metadata !15, metadata !DIExpression(DW_OP_deref)), !dbg !16
  call void @llvm.dbg.value(metadata i8 0, metadata !15, metadata !DIExpression()), !dbg !16
  store i8 0, ptr %c, align 1, !dbg !16
  store ptr %a, ptr %ref.tmp, align 8, !dbg !16
  %0 = getelementptr inbounds %class.anon, ptr %ref.tmp, i64 0, i32 1, !dbg !16
  store ptr %b, ptr %0, align 8, !dbg !16
  %1 = getelementptr inbounds %class.anon, ptr %ref.tmp, i64 0, i32 2, !dbg !16
  store ptr %c, ptr %1, align 8, !dbg !16
  call fastcc void @"_ZZ4testvENK3$_0clEv"(ptr nonnull %ref.tmp), !dbg !16
  %2 = load i8, ptr %a, align 1, !dbg !17
  call void @llvm.dbg.value(metadata i8 %2, metadata !12, metadata !DIExpression()), !dbg !16
  %tobool = icmp eq i8 %2, 0, !dbg !17
  br i1 %tobool, label %lor.lhs.false, label %if.then, !dbg !17

lor.lhs.false:                                    ; preds = %entry
  %3 = load i8, ptr %b, align 1, !dbg !17
  call void @llvm.dbg.value(metadata i8 %3, metadata !14, metadata !DIExpression()), !dbg !16
  %tobool1 = icmp eq i8 %3, 0, !dbg !17
  br i1 %tobool1, label %lor.lhs.false2, label %if.then, !dbg !17

lor.lhs.false2:                                   ; preds = %lor.lhs.false
  %4 = load i8, ptr %c, align 1, !dbg !17
  call void @llvm.dbg.value(metadata i8 %4, metadata !15, metadata !DIExpression()), !dbg !16
  %tobool3 = icmp eq i8 %4, 0, !dbg !17
  br i1 %tobool3, label %if.end, label %if.then, !dbg !16

if.then:                                          ; preds = %lor.lhs.false2, %lor.lhs.false, %entry
  call void @llvm.debugtrap(), !dbg !19
  br label %if.end, !dbg !19

if.end:                                           ; preds = %lor.lhs.false2, %if.then
  call void @llvm.dbg.value(metadata ptr %c, metadata !15, metadata !DIExpression(DW_OP_deref)), !dbg !16
  call void @llvm.dbg.value(metadata ptr %b, metadata !14, metadata !DIExpression(DW_OP_deref)), !dbg !16
  call void @llvm.dbg.value(metadata ptr %a, metadata !12, metadata !DIExpression(DW_OP_deref)), !dbg !16
  ret void, !dbg !16
}

; Function Attrs: inlinehint nounwind uwtable
define internal fastcc void @"_ZZ4testvENK3$_0clEv"(ptr %this) unnamed_addr align 2 !dbg !21 {
entry:
  call void @llvm.dbg.value(metadata ptr %this, metadata !34, metadata !DIExpression()), !dbg !36
  %0 = getelementptr inbounds %class.anon, ptr %this, i32 0, i32 1, !dbg !36
  %1 = load ptr, ptr %0, align 8, !dbg !36
  store i8 1, ptr %1, align 1, !dbg !36
  ret void, !dbg !36
}

; Function Attrs: nounwind
declare void @llvm.debugtrap()

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.1-4 (tags/RELEASE_801/final)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.cc", directory: "/dir")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.1-4 (tags/RELEASE_801/final)"}
!7 = distinct !DISubprogram(name: "test", linkageName: "_Z4testv", scope: !8, file: !8, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DIFile(filename: "test.cc", directory: "/dir")
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !{!12, !14, !15}
!12 = !DILocalVariable(name: "a", scope: !7, file: !8, line: 2, type: !13)
!13 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!14 = !DILocalVariable(name: "b", scope: !7, file: !8, line: 3, type: !13)
!15 = !DILocalVariable(name: "c", scope: !7, file: !8, line: 4, type: !13)
!16 = !DILocation(line: 2, scope: !7)
!17 = !DILocation(line: 7, scope: !18)
!18 = distinct !DILexicalBlock(scope: !7, file: !8, line: 7)
!19 = !DILocation(line: 8, scope: !20)
!20 = distinct !DILexicalBlock(scope: !18, file: !8, line: 7)
!21 = distinct !DISubprogram(name: "operator()", linkageName: "_ZZ4testvENK3$_0clEv", scope: !22, file: !8, line: 5, type: !28, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !32, retainedNodes: !33)
!22 = distinct !DICompositeType(tag: DW_TAG_class_type, scope: !7, file: !8, line: 5, size: 192, flags: DIFlagTypePassByValue, elements: !23)
!23 = !{!24, !26, !27}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !22, file: !8, line: 5, baseType: !25, size: 64)
!25 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !13, size: 64)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !22, file: !8, line: 5, baseType: !25, size: 64, offset: 64)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !22, file: !8, line: 5, baseType: !25, size: 64, offset: 128)
!28 = !DISubroutineType(types: !29)
!29 = !{null, !30}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !31, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!31 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !22)
!32 = !DISubprogram(name: "operator()", scope: !22, file: !8, line: 5, type: !28, scopeLine: 5, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!33 = !{!34}
!34 = !DILocalVariable(name: "this", arg: 1, scope: !21, type: !35, flags: DIFlagArtificial | DIFlagObjectPointer)
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !31, size: 64)
!36 = !DILocation(line: 0, scope: !21)

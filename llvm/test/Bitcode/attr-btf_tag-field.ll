; REQUIRES: x86-registered-target
; RUN: llvm-as < %s | llvm-dis | FileCheck %s

%struct.t1 = type { i32 }
%struct.t2 = type { i8, [3 x i8] }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @foo(%struct.t1* %arg) #0 !dbg !9 {
entry:
  %arg.addr = alloca %struct.t1*, align 8
  store %struct.t1* %arg, %struct.t1** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.t1** %arg.addr, metadata !20, metadata !DIExpression()), !dbg !21
  %0 = load %struct.t1*, %struct.t1** %arg.addr, align 8, !dbg !22
  %a = getelementptr inbounds %struct.t1, %struct.t1* %0, i32 0, i32 0, !dbg !23
  %1 = load i32, i32* %a, align 4, !dbg !23
  ret i32 %1, !dbg !24
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @foo2(%struct.t2* %arg) #0 !dbg !25 {
entry:
  %arg.addr = alloca %struct.t2*, align 8
  store %struct.t2* %arg, %struct.t2** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.t2** %arg.addr, metadata !32, metadata !DIExpression()), !dbg !33
  %0 = load %struct.t2*, %struct.t2** %arg.addr, align 8, !dbg !34
  %1 = bitcast %struct.t2* %0 to i8*, !dbg !35
  %bf.load = load i8, i8* %1, align 4, !dbg !35
  %bf.shl = shl i8 %bf.load, 7, !dbg !35
  %bf.ashr = ashr i8 %bf.shl, 7, !dbg !35
  %bf.cast = sext i8 %bf.ashr to i32, !dbg !35
  ret i32 %bf.cast, !dbg !36
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0 (https://github.com/llvm/llvm-project.git 4cbaee98885ead226304e8836090069db6596965)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "attr-btf_tag-field.c", directory: "/home/yhs/work/tests/llvm/btf_tag")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"uwtable", i32 1}
!7 = !{i32 7, !"frame-pointer", i32 2}
!8 = !{!"clang version 14.0.0 (https://github.com/llvm/llvm-project.git 4cbaee98885ead226304e8836090069db6596965)"}
!9 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 11, type: !10, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !13}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", file: !1, line: 7, size: 32, elements: !15)
!15 = !{!16}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !14, file: !1, line: 8, baseType: !12, size: 32, annotations: !17)
!17 = !{!18, !19}
!18 = !{!"btf_decl_tag", !"tag1"}
!19 = !{!"btf_decl_tag", !"tag2"}

; CHECK:        !DIDerivedType(tag: DW_TAG_member, name: "a"
; CHECK-SAME:   annotations: ![[ANNOT:[0-9]+]]
; CHECK:        ![[ANNOT]] = !{![[TAG1:[0-9]+]], ![[TAG2:[0-9]+]]}
; CHECK:        ![[TAG1]] = !{!"btf_decl_tag", !"tag1"}
; CHECK:        ![[TAG2]] = !{!"btf_decl_tag", !"tag2"}

!20 = !DILocalVariable(name: "arg", arg: 1, scope: !9, file: !1, line: 11, type: !13)
!21 = !DILocation(line: 11, column: 20, scope: !9)
!22 = !DILocation(line: 12, column: 10, scope: !9)
!23 = !DILocation(line: 12, column: 15, scope: !9)
!24 = !DILocation(line: 12, column: 3, scope: !9)
!25 = distinct !DISubprogram(name: "foo2", scope: !1, file: !1, line: 19, type: !26, scopeLine: 19, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!26 = !DISubroutineType(types: !27)
!27 = !{!12, !28}
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !29, size: 64)
!29 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t2", file: !1, line: 15, size: 32, elements: !30)
!30 = !{!31}
!31 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !29, file: !1, line: 16, baseType: !12, size: 1, flags: DIFlagBitField, extraData: i64 0, annotations: !17)

; CHECK:        !DIDerivedType(tag: DW_TAG_member, name: "b"
; CHECK-SAME:   annotations: ![[ANNOT]]

!32 = !DILocalVariable(name: "arg", arg: 1, scope: !25, file: !1, line: 19, type: !28)
!33 = !DILocation(line: 19, column: 21, scope: !25)
!34 = !DILocation(line: 20, column: 10, scope: !25)
!35 = !DILocation(line: 20, column: 15, scope: !25)
!36 = !DILocation(line: 20, column: 3, scope: !25)

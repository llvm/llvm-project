; RUN: llc -mtriple=hexagon < %s
; REQUIRES: asserts

; Check that we no longer get this error:
; void llvm::ScheduleDAGMILive::scheduleMI(llvm::SUnit *, bool):
; Assertion `TopRPTracker.getPos() == CurrentTop && "out of sync"' failed.

target triple = "hexagon"

%struct.A = type { ptr, ptr }
%struct.B = type { ptr, ptr, ptr }

@.str.4 = external hidden unnamed_addr constant [41 x i8], align 1
@__func__.fred = external hidden unnamed_addr constant [16 x i8], align 1
@.str.5 = external hidden unnamed_addr constant [43 x i8], align 1

; Function Attrs: nounwind
declare void @_Assert(ptr, ptr) #0

; Function Attrs: nounwind
define void @fred(ptr %pA, ptr %p) #0 !dbg !6 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %pA, i64 0, metadata !26, metadata !28), !dbg !29
  tail call void @llvm.dbg.value(metadata ptr %p, i64 0, metadata !27, metadata !28), !dbg !30
  %cmp = icmp eq ptr %p, null, !dbg !31
  br i1 %cmp, label %cond.false, label %cond.end, !dbg !31

cond.false:                                       ; preds = %entry
  tail call void @_Assert(ptr @.str.4, ptr @__func__.fred) #0, !dbg !32
  br label %cond.end, !dbg !32

cond.end:                                         ; preds = %cond.false, %entry
  %cmp1 = icmp eq ptr %pA, null, !dbg !34
  br i1 %cmp1, label %cond.false3, label %cond.end4, !dbg !34

cond.false3:                                      ; preds = %cond.end
  tail call void @_Assert(ptr @.str.5, ptr @__func__.fred) #0, !dbg !35
  br label %cond.end4, !dbg !35

cond.end4:                                        ; preds = %cond.false3, %cond.end
  %0 = load ptr, ptr %pA, align 4, !dbg !38, !tbaa !39
  %cmp5 = icmp eq ptr %0, null, !dbg !44
  br i1 %cmp5, label %if.then, label %if.end, !dbg !45

if.then:                                          ; preds = %cond.end4
  %p1 = getelementptr inbounds %struct.A, ptr %pA, i32 0, i32 1, !dbg !46
  store ptr %p, ptr %p1, align 4, !dbg !48, !tbaa !49
  store ptr %p, ptr %pA, align 4, !dbg !50, !tbaa !39
  %p4 = getelementptr inbounds %struct.B, ptr %p, i32 0, i32 1, !dbg !51
  store ptr null, ptr %p4, align 4, !dbg !52, !tbaa !53
  %p5 = getelementptr inbounds %struct.B, ptr %p, i32 0, i32 2, !dbg !55
  store ptr null, ptr %p5, align 4, !dbg !56, !tbaa !57
  br label %return, !dbg !58

if.end:                                           ; preds = %cond.end4
  %1 = ptrtoint ptr %0 to i32, !dbg !59
  %p57 = getelementptr inbounds %struct.B, ptr %p, i32 0, i32 2, !dbg !60
  store ptr null, ptr %p57, align 4, !dbg !61, !tbaa !57
  %p49 = getelementptr inbounds %struct.B, ptr %p, i32 0, i32 1, !dbg !62
  store i32 %1, ptr %p49, align 4, !dbg !63, !tbaa !53
  %p511 = getelementptr inbounds %struct.B, ptr %0, i32 0, i32 2, !dbg !64
  store ptr %p, ptr %p511, align 4, !dbg !65, !tbaa !57
  store ptr %p, ptr %pA, align 4, !dbg !66, !tbaa !39
  br label %return, !dbg !67

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !68
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "bug.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.0"}
!6 = distinct !DISubprogram(name: "fred", scope: !1, file: !1, line: 138, type: !7, isLocal: false, isDefinition: true, scopeLine: 139, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !25)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9, !15}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 32, align: 32)
!10 = !DIDerivedType(tag: DW_TAG_typedef, name: "A", file: !11, line: 57, baseType: !12)
!11 = !DIFile(filename: "bug.h", directory: "/")
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !11, line: 54, size: 64, align: 32, elements: !13)
!13 = !{!14, !24}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "p2", scope: !12, file: !11, line: 55, baseType: !15, size: 32, align: 32)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 32, align: 32)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "B", file: !11, line: 50, baseType: !17)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "B", file: !11, line: 45, size: 96, align: 32, elements: !18)
!18 = !{!19, !21, !23}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "p3", scope: !17, file: !11, line: 47, baseType: !20, size: 32, align: 32)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 32, align: 32)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "p4", scope: !17, file: !11, line: 48, baseType: !22, size: 32, align: 32, offset: 32)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 32, align: 32)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "p5", scope: !17, file: !11, line: 49, baseType: !22, size: 32, align: 32, offset: 64)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "p1", scope: !12, file: !11, line: 56, baseType: !15, size: 32, align: 32, offset: 32)
!25 = !{!26, !27}
!26 = !DILocalVariable(name: "pA", arg: 1, scope: !6, file: !1, line: 138, type: !9)
!27 = !DILocalVariable(name: "p", arg: 2, scope: !6, file: !1, line: 138, type: !15)
!28 = !DIExpression()
!29 = !DILocation(line: 138, column: 34, scope: !6)
!30 = !DILocation(line: 138, column: 57, scope: !6)
!31 = !DILocation(line: 140, column: 5, scope: !6)
!32 = !DILocation(line: 140, column: 5, scope: !33)
!33 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 2)
!34 = !DILocation(line: 141, column: 5, scope: !6)
!35 = !DILocation(line: 141, column: 5, scope: !33)
!36 = !DILocation(line: 143, column: 30, scope: !37)
!37 = distinct !DILexicalBlock(scope: !6, file: !1, line: 143, column: 9)
!38 = !DILocation(line: 155, column: 18, scope: !6)
!39 = !{!40, !41, i64 0}
!40 = !{!"", !41, i64 0, !41, i64 4}
!41 = !{!"any pointer", !42, i64 0}
!42 = !{!"omnipotent char", !43, i64 0}
!43 = !{!"Simple C/C++ TBAA"}
!44 = !DILocation(line: 143, column: 14, scope: !37)
!45 = !DILocation(line: 143, column: 9, scope: !6)
!46 = !DILocation(line: 146, column: 26, scope: !47)
!47 = distinct !DILexicalBlock(scope: !37, file: !1, line: 143, column: 41)
!48 = !DILocation(line: 146, column: 36, scope: !47)
!49 = !{!40, !41, i64 4}
!50 = !DILocation(line: 145, column: 32, scope: !47)
!51 = !DILocation(line: 147, column: 20, scope: !47)
!52 = !DILocation(line: 147, column: 29, scope: !47)
!53 = !{!54, !41, i64 4}
!54 = !{!"B", !41, i64 0, !41, i64 4, !41, i64 8}
!55 = !DILocation(line: 148, column: 20, scope: !47)
!56 = !DILocation(line: 148, column: 29, scope: !47)
!57 = !{!54, !41, i64 8}
!58 = !DILocation(line: 149, column: 9, scope: !47)
!59 = !DILocation(line: 154, column: 41, scope: !6)
!60 = !DILocation(line: 153, column: 16, scope: !6)
!61 = !DILocation(line: 153, column: 25, scope: !6)
!62 = !DILocation(line: 154, column: 16, scope: !6)
!63 = !DILocation(line: 154, column: 26, scope: !6)
!64 = !DILocation(line: 155, column: 29, scope: !6)
!65 = !DILocation(line: 155, column: 39, scope: !6)
!66 = !DILocation(line: 156, column: 28, scope: !6)
!67 = !DILocation(line: 157, column: 1, scope: !6)
!68 = !DILocation(line: 157, column: 1, scope: !69)
!69 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 1)

; RUN: llc %s -o /dev/null
; PR 2613.

%struct.__class_type_info_pseudo = type { %struct.__type_info_pseudo }
%struct.__type_info_pseudo = type { ptr, ptr }
%struct.test1 = type { ptr }

@_ZTV5test1 = weak_odr constant [4 x ptr] [ptr null, ptr @_ZTI5test1, ptr @_ZN5test1D1Ev, ptr @_ZN5test1D0Ev], align 32 ; <ptr> [#uses=1]
@_ZTI5test1 = weak_odr constant %struct.__class_type_info_pseudo { %struct.__type_info_pseudo { ptr getelementptr (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS5test1 } }, align 16 ; <ptr> [#uses=1]
@_ZTVN10__cxxabiv117__class_type_infoE = external constant [0 x ptr] ; <ptr> [#uses=1]
@_ZTS5test1 = weak_odr constant [7 x i8] c"5test1\00" ; <ptr> [#uses=2]

define i32 @main() nounwind ssp {
entry:
  %retval = alloca i32                            ; <ptr> [#uses=2]
  %0 = alloca i32                                 ; <ptr> [#uses=2]
  %tst = alloca %struct.test1                     ; <ptr> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata ptr %tst, metadata !0, metadata !DIExpression()), !dbg !21
  call void @_ZN5test1C1Ev(ptr %tst) nounwind, !dbg !22
  store i32 0, ptr %0, align 4, !dbg !23
  %1 = load i32, ptr %0, align 4, !dbg !23            ; <i32> [#uses=1]
  store i32 %1, ptr %retval, align 4, !dbg !23
  br label %return, !dbg !23

return:                                           ; preds = %entry
  %retval1 = load i32, ptr %retval, !dbg !23          ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !23
}

define linkonce_odr void @_ZN5test1C1Ev(ptr %this) nounwind ssp align 2 {
entry:
  %this_addr = alloca ptr              ; <ptr> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata ptr %this_addr, metadata !24, metadata !DIExpression()), !dbg !28
  store ptr %this, ptr %this_addr
  %0 = load ptr, ptr %this_addr, align 8, !dbg !28 ; <ptr> [#uses=1]
  store ptr getelementptr inbounds ([4 x ptr], ptr @_ZTV5test1, i64 0, i64 2), ptr %0, align 8, !dbg !28
  br label %return, !dbg !28

return:                                           ; preds = %entry
  ret void, !dbg !29
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define linkonce_odr void @_ZN5test1D1Ev(ptr %this) nounwind ssp align 2 {
entry:
  %this_addr = alloca ptr              ; <ptr> [#uses=3]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata ptr %this_addr, metadata !32, metadata !DIExpression()), !dbg !34
  store ptr %this, ptr %this_addr
  %0 = load ptr, ptr %this_addr, align 8, !dbg !35 ; <ptr> [#uses=1]
  store ptr getelementptr inbounds ([4 x ptr], ptr @_ZTV5test1, i64 0, i64 2), ptr %0, align 8, !dbg !35
  br label %bb, !dbg !37

bb:                                               ; preds = %entry
  %1 = trunc i32 0 to i8, !dbg !37                ; <i8> [#uses=1]
  %toBool = icmp ne i8 %1, 0, !dbg !37            ; <i1> [#uses=1]
  br i1 %toBool, label %bb1, label %bb2, !dbg !37

bb1:                                              ; preds = %bb
  %2 = load ptr, ptr %this_addr, align 8, !dbg !37 ; <ptr> [#uses=1]
  call void @_ZdlPv(ptr %2) nounwind, !dbg !37
  br label %bb2, !dbg !37

bb2:                                              ; preds = %bb1, %bb
  br label %return, !dbg !37

return:                                           ; preds = %bb2
  ret void, !dbg !37
}

define linkonce_odr void @_ZN5test1D0Ev(ptr %this) nounwind ssp align 2 {
entry:
  %this_addr = alloca ptr              ; <ptr> [#uses=3]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata ptr %this_addr, metadata !38, metadata !DIExpression()), !dbg !40
  store ptr %this, ptr %this_addr
  %0 = load ptr, ptr %this_addr, align 8, !dbg !41 ; <ptr> [#uses=1]
  store ptr getelementptr inbounds ([4 x ptr], ptr @_ZTV5test1, i64 0, i64 2), ptr %0, align 8, !dbg !41
  br label %bb, !dbg !43

bb:                                               ; preds = %entry
  %1 = trunc i32 1 to i8, !dbg !43                ; <i8> [#uses=1]
  %toBool = icmp ne i8 %1, 0, !dbg !43            ; <i1> [#uses=1]
  br i1 %toBool, label %bb1, label %bb2, !dbg !43

bb1:                                              ; preds = %bb
  %2 = load ptr, ptr %this_addr, align 8, !dbg !43 ; <ptr> [#uses=1]
  call void @_ZdlPv(ptr %2) nounwind, !dbg !43
  br label %bb2, !dbg !43

bb2:                                              ; preds = %bb1, %bb
  br label %return, !dbg !43

return:                                           ; preds = %bb2
  ret void, !dbg !43
}

declare void @_ZdlPv(ptr) nounwind

!llvm.dbg.cu = !{!4}
!0 = !DILocalVariable(name: "tst", line: 13, scope: !1, file: !4, type: !8)
!1 = distinct !DILexicalBlock(line: 0, column: 0, file: !44, scope: !2)
!2 = distinct !DILexicalBlock(line: 0, column: 0, file: !44, scope: !3)
!3 = distinct !DISubprogram(name: "main", linkageName: "main", line: 11, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !4, scope: !4, type: !5)
!4 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, emissionKind: FullDebug, file: !44, enums: !45, retainedTypes: !45)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "test1", line: 1, size: 64, align: 64, file: !44, scope: !4, elements: !9, vtableHolder: !8)
!9 = !{!10, !14, !18}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$test1", line: 1, size: 64, align: 64, file: !44, scope: !8, baseType: !11)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !4, baseType: !12)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", scope: !4, baseType: !5)
!13 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: false, emissionKind: FullDebug, file: !46, enums: !45, retainedTypes: !45)
!14 = !DISubprogram(name: "test1", line: 1, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrivate, isOptimized: false, scope: !8, type: !15)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, file: !4, baseType: !8)
!18 = !DISubprogram(name: "~test1", line: 4, isLocal: false, isDefinition: false, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 6, isOptimized: false, scope: !8, type: !19, containingType: !8)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !17, !7}
!21 = !DILocation(line: 11, scope: !1)
!22 = !DILocation(line: 13, scope: !1)
!23 = !DILocation(line: 14, scope: !1)
!24 = !DILocalVariable(name: "this", line: 13, arg: 1, scope: !25, file: !4, type: !26)
!25 = distinct !DISubprogram(name: "test1", linkageName: "_ZN5test1C1Ev", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !4, scope: !4, type: !15)
!26 = !DIDerivedType(tag: DW_TAG_const_type, size: 64, align: 64, flags: DIFlagArtificial, file: !4, baseType: !27)
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !4, baseType: !8)
!28 = !DILocation(line: 1, scope: !25)
!29 = !DILocation(line: 1, scope: !30)
!30 = distinct !DILexicalBlock(line: 0, column: 0, file: !44, scope: !31)
!31 = distinct !DILexicalBlock(line: 0, column: 0, file: !44, scope: !25)
!32 = !DILocalVariable(name: "this", line: 4, arg: 1, scope: !33, file: !4, type: !26)
!33 = distinct !DISubprogram(name: "~test1", linkageName: "_ZN5test1D1Ev", line: 4, isLocal: false, isDefinition: true, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 6, isOptimized: false, unit: !4, scope: !8, type: !15, containingType: !8)
!34 = !DILocation(line: 4, scope: !33)
!35 = !DILocation(line: 5, scope: !36)
!36 = distinct !DILexicalBlock(line: 0, column: 0, file: !44, scope: !33)
!37 = !DILocation(line: 6, scope: !36)
!38 = !DILocalVariable(name: "this", line: 4, arg: 1, scope: !39, file: !4, type: !26)
!39 = distinct !DISubprogram(name: "~test1", linkageName: "_ZN5test1D0Ev", line: 4, isLocal: false, isDefinition: true, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 6, isOptimized: false, unit: !4, scope: !8, type: !15, containingType: !8)
!40 = !DILocation(line: 4, scope: !39)
!41 = !DILocation(line: 5, scope: !42)
!42 = distinct !DILexicalBlock(line: 0, column: 0, file: !44, scope: !39)
!43 = !DILocation(line: 6, scope: !42)
!44 = !DIFile(filename: "inheritance.cpp", directory: "/tmp/")
!45 = !{i32 0}
!46 = !DIFile(filename: "<built-in>", directory: "/tmp/")

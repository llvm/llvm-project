; RUN: %llc_dwarf -O0 < %s | grep AT_decl_file |  grep 2
; Here _ZN1S3fooEv is defined in header file identified as AT_decl_file no. 2 in debug info.
%struct.S = type <{ i8 }>

define i32 @_Z3barv() nounwind ssp !dbg !3 {
entry:
  %retval = alloca i32                            ; <ptr> [#uses=2]
  %0 = alloca i32                                 ; <ptr> [#uses=2]
  %s1 = alloca %struct.S                          ; <ptr> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata ptr %s1, metadata !0, metadata !DIExpression()), !dbg !16
  %1 = call i32 @_ZN1S3fooEv(ptr %s1) nounwind, !dbg !17 ; <i32> [#uses=1]
  store i32 %1, ptr %0, align 4, !dbg !17
  %2 = load i32, ptr %0, align 4, !dbg !17            ; <i32> [#uses=1]
  store i32 %2, ptr %retval, align 4, !dbg !17
  br label %return, !dbg !17

return:                                           ; preds = %entry
  %retval1 = load i32, ptr %retval, !dbg !17          ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !16
}

define linkonce_odr i32 @_ZN1S3fooEv(ptr %this) nounwind ssp align 2 !dbg !12 {
entry:
  %this_addr = alloca ptr                  ; <ptr> [#uses=1]
  %retval = alloca i32                            ; <ptr> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata ptr %this_addr, metadata !18, metadata !DIExpression(DW_OP_deref)), !dbg !21
  store ptr %this, ptr %this_addr
  br label %return, !dbg !21

return:                                           ; preds = %entry
  %retval1 = load i32, ptr %retval, !dbg !21          ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !22
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!28}

!0 = !DILocalVariable(name: "s1", line: 3, scope: !1, file: !4, type: !9)
!1 = distinct !DILexicalBlock(line: 3, column: 0, file: !25, scope: !2)
!2 = distinct !DILexicalBlock(line: 3, column: 0, file: !25, scope: !3)
!3 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !5, scopeLine: 3, file: !25, scope: !4, type: !6)
!4 = !DIFile(filename: "one.cc", directory: "/tmp/")
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: false, emissionKind: FullDebug, file: !25, enums: !27, retainedTypes: !27, imports:  null)
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", line: 2, size: 8, align: 8, file: !26, scope: !4, elements: !11)
!10 = !DIFile(filename: "one.h", directory: "/tmp/")
!11 = !{!12}
!12 = distinct !DISubprogram(name: "foo", linkageName: "_ZN1S3fooEv", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !5, scopeLine: 3, file: !26, scope: !9, type: !13)
!13 = !DISubroutineType(types: !14)
!14 = !{!8, !15}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, file: !25, scope: !4, baseType: !9)
!16 = !DILocation(line: 3, scope: !1)
!17 = !DILocation(line: 3, scope: !3)
; Modified from being a pointer, to make this testcase independent of target pointer size
!18 = !DILocalVariable(name: "this", line: 3, arg: 1, scope: !12, file: !10, type: !9)
!19 = !DIDerivedType(tag: DW_TAG_const_type, size: 64, align: 64, flags: DIFlagArtificial, file: !25, scope: !4, baseType: !20)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !25, scope: !4, baseType: !9)
!21 = !DILocation(line: 3, scope: !12)
!22 = !DILocation(line: 3, scope: !23)
!23 = distinct !DILexicalBlock(line: 3, column: 0, file: !26, scope: !12)
!25 = !DIFile(filename: "one.cc", directory: "/tmp/")
!26 = !DIFile(filename: "one.h", directory: "/tmp/")
!27 = !{}
!28 = !{i32 1, !"Debug Info Version", i32 3}

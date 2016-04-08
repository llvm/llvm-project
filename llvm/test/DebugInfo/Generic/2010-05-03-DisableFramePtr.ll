; RUN: llc  -o /dev/null < %s
; Radar 7937664
%struct.AppleEvent = type opaque

define void @DisposeDMNotificationUPP(void (%struct.AppleEvent*)* %userUPP) "no-frame-pointer-elim-non-leaf" nounwind ssp {
entry:
  %userUPP_addr = alloca void (%struct.AppleEvent*)* ; <void (%struct.AppleEvent*)**> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata void (%struct.AppleEvent*)** %userUPP_addr, metadata !0, metadata !DIExpression(DW_OP_deref)), !dbg !13
  store void (%struct.AppleEvent*)* %userUPP, void (%struct.AppleEvent*)** %userUPP_addr
  br label %return, !dbg !14

return:                                           ; preds = %entry
  ret void, !dbg !14
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!19}
!0 = !DILocalVariable(name: "userUPP", line: 7, arg: 1, scope: !1, file: !2, type: !6)
!1 = distinct !DISubprogram(name: "DisposeDMNotificationUPP", linkageName: "DisposeDMNotificationUPP", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !16, scope: null, type: !4)
!2 = !DIFile(filename: "t.c", directory: "/Users/echeng/LLVM/radars/r7937664/")
!3 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build 9999)", isOptimized: true, emissionKind: FullDebug, file: !16, enums: !17, retainedTypes: !17, subprograms: !18)
!4 = !DISubroutineType(types: !5)
!5 = !{null, !6}
; Manually modified to avoid dependence on pointer size in generic test
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "DMNotificationUPP", line: 6, file: !16, scope: !2, baseType: !8)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !16, scope: !2, baseType: !8)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !16, scope: !2, baseType: !11)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "AppleEvent", line: 4, file: !16, scope: !2, baseType: !12)
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "AEDesc", line: 1, flags: DIFlagFwdDecl, file: !16, scope: !2)
!13 = !DILocation(line: 7, scope: !1)
!14 = !DILocation(line: 8, scope: !15)
!15 = distinct !DILexicalBlock(line: 7, column: 0, file: !16, scope: !1)
!16 = !DIFile(filename: "t.c", directory: "/Users/echeng/LLVM/radars/r7937664/")
!17 = !{}
!18 = !{!1}
!19 = !{i32 1, !"Debug Info Version", i32 3}

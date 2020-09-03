; RUN: llc -O1 -filetype=obj -emit-call-site-info -debug-entry-values -o - < %s | llvm-dwarfdump -verify - -o /dev/null

; TODO: This test should be made more targeted by converting to MIR and reducing,
; however at the moment conversion to MIR fails with:
; Assertion failed: (!NameRef.empty() && "Normal symbols cannot be unnamed!")

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

%struct.e = type opaque
%"class.aa::aq" = type { i8 }
%"class.aa::ah" = type { i8 }
%"class.aa::y" = type { i8 }
%"class.aa::y.0" = type { i8 }
%struct.j = type opaque
%struct.h = type opaque
%struct.r = type opaque

@o = local_unnamed_addr global i32 0, align 4, !dbg !0
@p = local_unnamed_addr global %struct.e* null, align 8, !dbg !42

; Function Attrs: optsize ssp uwtable
define void @_ZN2aa2aq2arEv(%"class.aa::aq"* %this) local_unnamed_addr #0 align 2 !dbg !50 {
entry:
  call void @llvm.dbg.value(metadata %"class.aa::aq"* %this, metadata !71, metadata !DIExpression()), !dbg !75
  %0 = bitcast %"class.aa::aq"* %this to %"class.aa::ah"*, !dbg !76
  tail call void @_ZN2aa2ah2aiEiib(%"class.aa::ah"* %0, i32 undef, i32 undef, i1 zeroext true) #5, !dbg !76
  ret void, !dbg !77
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: optsize ssp uwtable
define linkonce_odr void @_ZN2aa2ah2aiEiib(%"class.aa::ah"* %this, i32 %aj, i32 %0, i1 zeroext %1) local_unnamed_addr #0 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) !dbg !78 {
entry:
  %ao = alloca %"class.aa::y", align 1
  %ap = alloca %"class.aa::y.0", align 1
  call void @llvm.dbg.value(metadata %"class.aa::ah"* %this, metadata !80, metadata !DIExpression()), !dbg !126
  call void @llvm.dbg.value(metadata i32 %aj, metadata !82, metadata !DIExpression()), !dbg !126
  call void @llvm.dbg.value(metadata i32 %0, metadata !83, metadata !DIExpression()), !dbg !126
  call void @llvm.dbg.value(metadata i1 %1, metadata !84, metadata !DIExpression()), !dbg !126
  call void @llvm.dbg.value(metadata i32 %aj, metadata !85, metadata !DIExpression()), !dbg !126
  %2 = getelementptr inbounds %"class.aa::y", %"class.aa::y"* %ao, i64 0, i32 0, !dbg !127
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %2) #6, !dbg !127
  call void @llvm.dbg.declare(metadata %"class.aa::y"* %ao, metadata !91, metadata !DIExpression()), !dbg !128
  %call = tail call %struct.j* @_Z1mPvS_lPFvS_PKvlE(i8* undef, i8* undef, i64 0, void (i8*, i8*, i64)* nonnull @_ZN2aa12_GLOBAL__N_12agEPvPKvl) #5, !dbg !129
  call void @_ZN2aa1yIP1jNS_2ac1zI1eEEEC1ES2_(%"class.aa::y"* nonnull %ao, %struct.j* %call) #5, !dbg !128
  %3 = getelementptr inbounds %"class.aa::y.0", %"class.aa::y.0"* %ap, i64 0, i32 0, !dbg !130
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %3) #6, !dbg !130
  call void @llvm.dbg.declare(metadata %"class.aa::y.0"* %ap, metadata !110, metadata !DIExpression()), !dbg !131
  %4 = load %struct.e*, %struct.e** @p, align 8, !dbg !132, !tbaa !133
  %call3 = invoke %struct.h* @_Z1qP1e(%struct.e* %4) #5
          to label %invoke.cont unwind label %lpad, !dbg !137

invoke.cont:                                      ; preds = %entry
  invoke void @_ZN2aa1yIP1hNS_2ac1zI1eEEEC1ES2_(%"class.aa::y.0"* nonnull %ap, %struct.h* %call3) #5
          to label %invoke.cont4 unwind label %lpad, !dbg !131

invoke.cont4:                                     ; preds = %invoke.cont
  %conv = sext i32 %aj to i64, !dbg !138
  %mul = shl nsw i32 %aj, 2, !dbg !139
  %conv6 = sext i32 %mul to i64, !dbg !140
  %call9 = invoke %struct.h* @_ZN2aa1yIP1hNS_2ac1zI1eEEE2abEv(%"class.aa::y.0"* nonnull %ap) #5
          to label %invoke.cont8 unwind label %lpad7, !dbg !141

invoke.cont8:                                     ; preds = %invoke.cont4
  %call11 = invoke %struct.j* @_ZN2aa1yIP1jNS_2ac1zI1eEEE2abEv(%"class.aa::y"* nonnull %ao) #5
          to label %invoke.cont10 unwind label %lpad7, !dbg !142

invoke.cont10:                                    ; preds = %invoke.cont8
  %5 = load i32, i32* @o, align 4, !dbg !143, !tbaa !144
  %call13 = invoke %struct.r* @_Z1vlllllP1hiP1jPdb1n(i64 %conv, i64 0, i64 8, i64 2, i64 %conv6, %struct.h* %call9, i32 0, %struct.j* %call11, double* null, i1 zeroext false, i32 %5) #5
          to label %invoke.cont12 unwind label %lpad7, !dbg !146

invoke.cont12:                                    ; preds = %invoke.cont10
  unreachable, !dbg !146

lpad:                                             ; preds = %invoke.cont, %entry
  %6 = landingpad { i8*, i32 }
          cleanup, !dbg !147
  %7 = extractvalue { i8*, i32 } %6, 0, !dbg !147
  %8 = extractvalue { i8*, i32 } %6, 1, !dbg !147
  br label %ehcleanup, !dbg !147

lpad7:                                            ; preds = %invoke.cont10, %invoke.cont8, %invoke.cont4
  %9 = landingpad { i8*, i32 }
          cleanup, !dbg !147
  %10 = extractvalue { i8*, i32 } %9, 0, !dbg !147
  %11 = extractvalue { i8*, i32 } %9, 1, !dbg !147
  call void @_ZN2aa1yIP1hNS_2ac1zI1eEEED1Ev(%"class.aa::y.0"* nonnull %ap) #7, !dbg !147
  br label %ehcleanup, !dbg !147

ehcleanup:                                        ; preds = %lpad7, %lpad
  %exn.slot.0 = phi i8* [ %10, %lpad7 ], [ %7, %lpad ], !dbg !147
  %ehselector.slot.0 = phi i32 [ %11, %lpad7 ], [ %8, %lpad ], !dbg !147
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %3) #6, !dbg !147
  call void @_ZN2aa1yIP1jNS_2ac1zI1eEEED1Ev(%"class.aa::y"* nonnull %ao) #7, !dbg !147
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %2) #6, !dbg !147
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.0, 0, !dbg !147
  %lpad.val19 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.0, 1, !dbg !147
  resume { i8*, i32 } %lpad.val19, !dbg !147
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: optsize
declare !dbg !11 %struct.j* @_Z1mPvS_lPFvS_PKvlE(i8*, i8*, i64, void (i8*, i8*, i64)*) local_unnamed_addr #3

; Function Attrs: optsize
declare void @_ZN2aa12_GLOBAL__N_12agEPvPKvl(i8*, i8*, i64) #3

; Function Attrs: optsize
declare void @_ZN2aa1yIP1jNS_2ac1zI1eEEEC1ES2_(%"class.aa::y"*, %struct.j*) unnamed_addr #3

; Function Attrs: optsize
declare !dbg !24 %struct.h* @_Z1qP1e(%struct.e*) local_unnamed_addr #3

declare i32 @__gxx_personality_v0(...)

; Function Attrs: optsize
declare void @_ZN2aa1yIP1hNS_2ac1zI1eEEEC1ES2_(%"class.aa::y.0"*, %struct.h*) unnamed_addr #3

; Function Attrs: optsize
declare !dbg !31 %struct.r* @_Z1vlllllP1hiP1jPdb1n(i64, i64, i64, i64, i64, %struct.h*, i32, %struct.j*, double*, i1 zeroext, i32) local_unnamed_addr #3

; Function Attrs: optsize
declare %struct.h* @_ZN2aa1yIP1hNS_2ac1zI1eEEE2abEv(%"class.aa::y.0"*) local_unnamed_addr #3

; Function Attrs: optsize
declare %struct.j* @_ZN2aa1yIP1jNS_2ac1zI1eEEE2abEv(%"class.aa::y"*) local_unnamed_addr #3

; Function Attrs: nounwind optsize
declare void @_ZN2aa1yIP1hNS_2ac1zI1eEEED1Ev(%"class.aa::y.0"*) unnamed_addr #4

; Function Attrs: nounwind optsize
declare void @_ZN2aa1yIP1jNS_2ac1zI1eEEED1Ev(%"class.aa::y"*) unnamed_addr #4

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { optsize ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { argmemonly nounwind willreturn }
attributes #3 = { optsize "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind optsize "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { optsize }
attributes #6 = { nounwind }
attributes #7 = { nounwind optsize }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!45, !46, !47, !48}
!llvm.ident = !{!49}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "o", scope: !2, file: !6, line: 11, type: !40, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 11.0.0 (git@github.com:llvm/llvm-project.git 0fecdcd1628999a1900d9cf84cd33dacf1319fa6)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !10, globals: !41, nameTableKind: None, sysroot: "/")
!3 = !DIFile(filename: "/Users/vsk/tmp/x.cc", directory: "/Users/vsk/src/llvm-backup-master")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !6, line: 16, baseType: !7, size: 32, elements: !8)
!6 = !DIFile(filename: "tmp/x.cc", directory: "/Users/vsk")
!7 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!8 = !{!9}
!9 = !DIEnumerator(name: "u", value: 0, isUnsigned: true)
!10 = !{!11, !24, !31}
!11 = !DISubprogram(name: "m", linkageName: "_Z1mPvS_lPFvS_PKvlE", scope: !6, file: !6, line: 10, type: !12, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !23)
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !16, !16, !17, !18}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DICompositeType(tag: DW_TAG_structure_type, name: "j", file: !6, line: 8, flags: DIFlagFwdDecl, identifier: "_ZTS1j")
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!17 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !16, !21, !17}
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!22 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!23 = !{}
!24 = !DISubprogram(name: "q", linkageName: "_Z1qP1e", scope: !6, file: !6, line: 13, type: !25, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !23)
!25 = !DISubroutineType(types: !26)
!26 = !{!27, !29}
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !28, size: 64)
!28 = !DICompositeType(tag: DW_TAG_structure_type, name: "h", file: !6, line: 7, flags: DIFlagFwdDecl, identifier: "_ZTS1h")
!29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !30, size: 64)
!30 = !DICompositeType(tag: DW_TAG_structure_type, name: "e", file: !6, line: 5, flags: DIFlagFwdDecl, identifier: "_ZTS1e")
!31 = !DISubprogram(name: "v", linkageName: "_Z1vlllllP1hiP1jPdb1n", scope: !6, file: !6, line: 17, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !23)
!32 = !DISubroutineType(types: !33)
!33 = !{!34, !17, !17, !17, !17, !17, !27, !36, !14, !37, !39, !40}
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !35, size: 64)
!35 = !DICompositeType(tag: DW_TAG_structure_type, name: "r", file: !6, line: 14, flags: DIFlagFwdDecl, identifier: "_ZTS1r")
!36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !38, size: 64)
!38 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!39 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!40 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "n", file: !6, line: 11, size: 32, flags: DIFlagFwdDecl, identifier: "_ZTS1n")
!41 = !{!0, !42}
!42 = !DIGlobalVariableExpression(var: !43, expr: !DIExpression())
!43 = distinct !DIGlobalVariable(name: "p", scope: !2, file: !6, line: 12, type: !44, isLocal: false, isDefinition: true)
!44 = !DIDerivedType(tag: DW_TAG_typedef, name: "f", file: !6, line: 5, baseType: !29)
!45 = !{i32 7, !"Dwarf Version", i32 4}
!46 = !{i32 2, !"Debug Info Version", i32 3}
!47 = !{i32 1, !"wchar_size", i32 4}
!48 = !{i32 7, !"PIC Level", i32 2}
!49 = !{!"clang version 11.0.0 (git@github.com:llvm/llvm-project.git 0fecdcd1628999a1900d9cf84cd33dacf1319fa6)"}
!50 = distinct !DISubprogram(name: "ar", linkageName: "_ZN2aa2aq2arEv", scope: !51, file: !6, line: 48, type: !67, scopeLine: 48, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, declaration: !66, retainedNodes: !70)
!51 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "aq", scope: !52, file: !6, line: 45, size: 8, flags: DIFlagTypePassByValue, elements: !53, identifier: "_ZTSN2aa2aqE")
!52 = !DINamespace(name: "aa", scope: null)
!53 = !{!54, !66}
!54 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !51, baseType: !55, extraData: i32 0)
!55 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "ah", scope: !52, file: !6, line: 34, size: 8, flags: DIFlagTypePassByValue, elements: !56, identifier: "_ZTSN2aa2ahE")
!56 = !{!57}
!57 = !DISubprogram(name: "ai", linkageName: "_ZN2aa2ah2aiEiib", scope: !55, file: !6, line: 36, type: !58, scopeLine: 36, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!58 = !DISubroutineType(types: !59)
!59 = !{!60, !64, !65, !65, !39}
!60 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "af<w>", scope: !52, file: !6, line: 30, size: 8, flags: DIFlagTypePassByValue, elements: !23, templateParams: !61, identifier: "_ZTSN2aa2afI1wEE")
!61 = !{!62}
!62 = !DITemplateTypeParameter(type: !63)
!63 = !DICompositeType(tag: DW_TAG_class_type, name: "w", file: !6, line: 18, flags: DIFlagFwdDecl, identifier: "_ZTS1w")
!64 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !55, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!65 = !DIDerivedType(tag: DW_TAG_typedef, name: "b", file: !6, line: 2, baseType: !36)
!66 = !DISubprogram(name: "ar", linkageName: "_ZN2aa2aq2arEv", scope: !51, file: !6, line: 46, type: !67, scopeLine: 46, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!67 = !DISubroutineType(types: !68)
!68 = !{null, !69}
!69 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !51, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!70 = !{!71, !73, !74}
!71 = !DILocalVariable(name: "this", arg: 1, scope: !50, type: !72, flags: DIFlagArtificial | DIFlagObjectPointer)
!72 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !51, size: 64)
!73 = !DILocalVariable(name: "aj", scope: !50, file: !6, line: 49, type: !65)
!74 = !DILocalVariable(name: "am", scope: !50, file: !6, line: 50, type: !65)
!75 = !DILocation(line: 0, scope: !50)
!76 = !DILocation(line: 51, column: 3, scope: !50)
!77 = !DILocation(line: 52, column: 1, scope: !50)
!78 = distinct !DISubprogram(name: "ai", linkageName: "_ZN2aa2ah2aiEiib", scope: !55, file: !6, line: 36, type: !58, scopeLine: 36, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, declaration: !57, retainedNodes: !79)
!79 = !{!80, !82, !83, !84, !85, !86, !87, !91, !110}
!80 = !DILocalVariable(name: "this", arg: 1, scope: !78, type: !81, flags: DIFlagArtificial | DIFlagObjectPointer)
!81 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !55, size: 64)
!82 = !DILocalVariable(name: "aj", arg: 2, scope: !78, file: !6, line: 36, type: !65)
!83 = !DILocalVariable(arg: 3, scope: !78, file: !6, line: 36, type: !65)
!84 = !DILocalVariable(arg: 4, scope: !78, file: !6, line: 36, type: !39)
!85 = !DILocalVariable(name: "ak", scope: !78, file: !6, line: 37, type: !65)
!86 = !DILocalVariable(name: "al", scope: !78, file: !6, line: 38, type: !65)
!87 = !DILocalVariable(name: "an", scope: !78, file: !6, line: 39, type: !88)
!88 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !89, size: 64)
!89 = !DIDerivedType(tag: DW_TAG_typedef, name: "c", file: !6, line: 3, baseType: !90)
!90 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!91 = !DILocalVariable(name: "ao", scope: !78, file: !6, line: 40, type: !92)
!92 = !DIDerivedType(tag: DW_TAG_typedef, name: "ae<k>", scope: !52, file: !6, line: 29, baseType: !93)
!93 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "y<j *, aa::ac::z<e> >", scope: !52, file: !6, line: 20, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !94, templateParams: !105, identifier: "_ZTSN2aa1yIP1jNS_2ac1zI1eEEEE")
!94 = !{!95, !99, !102}
!95 = !DISubprogram(name: "y", scope: !93, file: !6, line: 22, type: !96, scopeLine: 22, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!96 = !DISubroutineType(types: !97)
!97 = !{null, !98, !14}
!98 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !93, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!99 = !DISubprogram(name: "~y", scope: !93, file: !6, line: 23, type: !100, scopeLine: 23, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!100 = !DISubroutineType(types: !101)
!101 = !{null, !98}
!102 = !DISubprogram(name: "ab", linkageName: "_ZN2aa1yIP1jNS_2ac1zI1eEEE2abEv", scope: !93, file: !6, line: 24, type: !103, scopeLine: 24, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!103 = !DISubroutineType(types: !104)
!104 = !{!14, !98}
!105 = !{!106, !107}
!106 = !DITemplateTypeParameter(name: "x", type: !14)
!107 = !DITemplateTypeParameter(type: !108)
!108 = !DICompositeType(tag: DW_TAG_structure_type, name: "z<e>", scope: !109, file: !6, line: 27, flags: DIFlagFwdDecl, identifier: "_ZTSN2aa2ac1zI1eEE")
!109 = !DINamespace(name: "ac", scope: !52)
!110 = !DILocalVariable(name: "ap", scope: !78, file: !6, line: 41, type: !111)
!111 = !DIDerivedType(tag: DW_TAG_typedef, name: "ae<i>", scope: !52, file: !6, line: 29, baseType: !112)
!112 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "y<h *, aa::ac::z<e> >", scope: !52, file: !6, line: 20, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !113, templateParams: !124, identifier: "_ZTSN2aa1yIP1hNS_2ac1zI1eEEEE")
!113 = !{!114, !118, !121}
!114 = !DISubprogram(name: "y", scope: !112, file: !6, line: 22, type: !115, scopeLine: 22, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!115 = !DISubroutineType(types: !116)
!116 = !{null, !117, !27}
!117 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !112, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!118 = !DISubprogram(name: "~y", scope: !112, file: !6, line: 23, type: !119, scopeLine: 23, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!119 = !DISubroutineType(types: !120)
!120 = !{null, !117}
!121 = !DISubprogram(name: "ab", linkageName: "_ZN2aa1yIP1hNS_2ac1zI1eEEE2abEv", scope: !112, file: !6, line: 24, type: !122, scopeLine: 24, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!122 = !DISubroutineType(types: !123)
!123 = !{!27, !117}
!124 = !{!125, !107}
!125 = !DITemplateTypeParameter(name: "x", type: !27)
!126 = !DILocation(line: 0, scope: !78)
!127 = !DILocation(line: 40, column: 5, scope: !78)
!128 = !DILocation(line: 40, column: 11, scope: !78)
!129 = !DILocation(line: 40, column: 14, scope: !78)
!130 = !DILocation(line: 41, column: 5, scope: !78)
!131 = !DILocation(line: 41, column: 11, scope: !78)
!132 = !DILocation(line: 41, column: 16, scope: !78)
!133 = !{!134, !134, i64 0}
!134 = !{!"any pointer", !135, i64 0}
!135 = !{!"omnipotent char", !136, i64 0}
!136 = !{!"Simple C++ TBAA"}
!137 = !DILocation(line: 41, column: 14, scope: !78)
!138 = !DILocation(line: 42, column: 7, scope: !78)
!139 = !DILocation(line: 42, column: 23, scope: !78)
!140 = !DILocation(line: 42, column: 21, scope: !78)
!141 = !DILocation(line: 42, column: 32, scope: !78)
!142 = !DILocation(line: 42, column: 44, scope: !78)
!143 = !DILocation(line: 42, column: 70, scope: !78)
!144 = !{!145, !145, i64 0}
!145 = !{!"_ZTS1n", !135, i64 0}
!146 = !DILocation(line: 42, column: 5, scope: !78)
!147 = !DILocation(line: 43, column: 3, scope: !78)

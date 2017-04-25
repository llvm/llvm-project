; RUN: llc -filetype=obj -o - %s | llvm-dwarfdump - | FileCheck %s
; Note that it would be even better to avoid emitting the empty piece.
; CHECK:  Location description: 93 08
;                               piece 0x00000008
source_filename = "bugpoint-reduced-simplified.ll"
target triple = "i386-apple-ios7.0.0"

%class.K = type { %class.N, %struct.F, i32 }
%class.N = type { %struct.B }
%struct.B = type { i8 }
%struct.F = type { i8 }
%struct.ConditionPlatformHelper = type { i8 }
%"struct.J<K::L, false, int>::Node" = type { %"class.K::L" }
%"class.K::L" = type { %class.__thread_id }
%class.__thread_id = type { i32 }
%"struct.K::M" = type { %class.I, %class.H }
%class.I = type { i32 }
%class.H = type { i32 }

; Function Attrs: nounwind
define void @_Z34swift_getGenericMetadata_argumentsv() #0 !dbg !5 {
entry:
  %ref.tmp = alloca %class.K, align 8
  %0 = getelementptr inbounds %class.K, %class.K* %ref.tmp, i32 0, i32 0, i32 0, i32 0
  %call = tail call i64 @_Z8getCacheP23ConditionPlatformHelper(%struct.ConditionPlatformHelper* undef)
  %1 = bitcast %class.K* %ref.tmp to i64*
  %LastSearch.i.i = getelementptr inbounds %class.K, %class.K* %ref.tmp, i32 0, i32 0, i32 0
  %call.i.i = call %"struct.J<K::L, false, int>::Node"* @_ZN1BIPN1JIN1K1LELb0EiE4NodeEE4loadEv(%struct.B* nonnull %LastSearch.i.i)
  %tobool.i.i = icmp eq %"struct.J<K::L, false, int>::Node"* %call.i.i, null
  br i1 %tobool.i.i, label %_ZN1NIN1K1LELi0EE11getOrInsertIiEE1AIPS1_ET_.exit.i, label %if.then.i.i

if.then.i.i:
  %2 = lshr i64 %call, 32
  %3 = trunc i64 %2 to i32
  %Payload.i.i = getelementptr inbounds %"struct.J<K::L, false, int>::Node", %"struct.J<K::L, false, int>::Node"* %call.i.i, i32 0, i32 0
  br label %_ZN1NIN1K1LELi0EE11getOrInsertIiEE1AIPS1_ET_.exit.i

_ZN1NIN1K1LELi0EE11getOrInsertIiEE1AIPS1_ET_.exit.i: ; preds = %if.then.i.i, %entry
  %retval.sroa.0.0.i.i = phi %"class.K::L"* [ %Payload.i.i, %if.then.i.i ], [ undef, %entry ]
  %call4.i = call %"struct.K::M"* @_ZN1FIN1K1MEE3getEv(%struct.F* undef)
  call void @llvm.dbg.value(metadata %"struct.K::M"* %call4.i, i64 0, metadata !7, metadata !11), !dbg !12
  call void @llvm.dbg.value(metadata %"struct.K::M"* %call4.i, i64 0, metadata !7, metadata !18), !dbg !12
  %Handle2.i.i.i.i.i = getelementptr inbounds %"struct.K::M", %"struct.K::M"* %call4.i, i32 0, i32 0, i32 0
  %Handle.i.i.i.i.i = getelementptr inbounds %"struct.K::M", %"struct.K::M"* %call4.i, i32 0, i32 1, i32 0
  %4 = getelementptr inbounds %"class.K::L", %"class.K::L"* %retval.sroa.0.0.i.i, i32 0, i32 0, i32 0
  br label %while.body.i.i.i.i

while.body.i.i.i.i:
  %5 = load i32, i32* %4, align 4
  %call.i.i.i.i.i.i = call i32 @_Z6get_idv(), !dbg !12
  %call.i.i.i.i.i.i.i = call zeroext i1 @_Z24__libcpp_thread_id_equalii(i32 %5, i32 %call.i.i.i.i.i.i)
  %6 = load i32, i32* %Handle2.i.i.i.i.i, align 4
  call void @_ZN23ConditionPlatformHelper4waitERii(i32* nonnull dereferenceable(4) %Handle.i.i.i.i.i, i32 %6)
  br label %while.body.i.i.i.i
}

declare i64 @_Z8getCacheP23ConditionPlatformHelper(%struct.ConditionPlatformHelper*) local_unnamed_addr

declare %"struct.K::M"* @_ZN1FIN1K1MEE3getEv(%struct.F*) local_unnamed_addr

declare %"struct.J<K::L, false, int>::Node"* @_ZN1BIPN1JIN1K1LELb0EiE4NodeEE4loadEv(%struct.B*) local_unnamed_addr

declare i32 @_Z6get_idv() local_unnamed_addr

declare zeroext i1 @_Z24__libcpp_thread_id_equalii(i32, i32) local_unnamed_addr

declare void @_ZN23ConditionPlatformHelper4waitERii(i32* dereferenceable(4), i32) local_unnamed_addr

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.ii", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "swift_getGenericMetadata_arguments", linkageName: "_Z34swift_getGenericMetadata_argumentsv", scope: !0, file: !1, line: 95, type: !6, isLocal: false, isDefinition: true, scopeLine: 95, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!6 = !DISubroutineType(types: !2)
!7 = !DILocalVariable(name: "criticalSection", arg: 2, scope: !8, file: !1, line: 23, type: !10)
!8 = distinct !DISubprogram(name: "<(lambda at test.ii:28:14)>", scope: !0, file: !1, line: 23, type: !6, isLocal: false, isDefinition: true, scopeLine: 23, flags: DIFlagPrototyped, isOptimized: true, unit: !0, templateParams: !2, declaration: !9, variables: !2)
!9 = !DISubprogram(name: "<(lambda at test.ii:28:14)>", scope: !0, file: !1, line: 23, type: !6, isLocal: false, isDefinition: false, scopeLine: 23, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: true, templateParams: !2)
!10 = distinct !DICompositeType(tag: DW_TAG_class_type, scope: !0, file: !1, line: 28, size: 96, elements: !2)
!11 = !DIExpression(DW_OP_stack_value, DW_OP_LLVM_fragment, 0, 32)
!12 = !DILocation(line: 23, column: 33, scope: !8, inlinedAt: !13)
!13 = distinct !DILocation(line: 28, column: 5, scope: !14, inlinedAt: !16)
!14 = distinct !DISubprogram(name: "<(lambda at test.ii:87:58)>", scope: !0, file: !1, line: 27, type: !6, isLocal: false, isDefinition: true, scopeLine: 27, flags: DIFlagPrototyped, isOptimized: true, unit: !0, templateParams: !2, declaration: !15, variables: !2)
!15 = !DISubprogram(name: "<(lambda at test.ii:87:58)>", scope: !0, file: !1, line: 27, type: !6, isLocal: false, isDefinition: false, scopeLine: 27, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: true, templateParams: !2)
!16 = distinct !DILocation(line: 99, column: 21, scope: !17)
!17 = !DILexicalBlockFile(scope: !5, file: !1, discriminator: 2)
!18 = !DIExpression(DW_OP_plus, 4, DW_OP_stack_value, DW_OP_LLVM_fragment, 64, 32)

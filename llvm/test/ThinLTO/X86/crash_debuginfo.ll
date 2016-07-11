; RUN: opt -module-summary -o %t-dst.bc %s
; RUN: opt -module-summary -o %t-src.bc %p/Inputs/crash_debuginfo.ll 
; RUN: llvm-lto -thinlto -o %t-index %t-dst.bc %t-src.bc
; RUN: opt -function-import -summary-file %t-index.thinlto.bc %t-dst.bc -o /dev/null

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.0"

%"class.boost::python::api::slice_nil.60.2301.31" = type { %"class.boost::python::api::object.59.2300.30" }
%"class.boost::python::api::object.59.2300.30" = type { %"struct.boost::python::api::object_base.58.2299.29" }
%"struct.boost::python::api::object_base.58.2299.29" = type { %struct._object.57.2259.28* }
%struct._object.57.2259.28 = type { i64, %struct._typeobject.56.2258.27* }
%struct._typeobject.56.2258.27 = type { i64, %struct._typeobject.56.2258.27*, i64, i8*, i64, i64, void (%struct._object.57.2259.28*)*, i32 (%struct._object.57.2259.28*, %struct.__sFILE.47.2249.18*, i32)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, i8*)*, i32 (%struct._object.57.2259.28*, i8*, %struct._object.57.2259.28*)*, i32 (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*)*, %struct.PyNumberMethods.48.2250.19*, %struct.PySequenceMethods.49.2251.20*, %struct.PyMappingMethods.50.2252.21*, i64 (%struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, i32 (%struct._object.57.2259.28*, %struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct.PyBufferProcs.52.2254.23*, i64, i8*, i32 (%struct._object.57.2259.28*, i32 (%struct._object.57.2259.28*, i8*)*, i8*)*, i32 (%struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*, i32)*, i64, %struct._object.57.2259.28* (%struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*)*, %struct.PyMethodDef.53.2255.24*, %struct.PyMemberDef.54.2256.25*, %struct.PyGetSetDef.55.2257.26*, %struct._typeobject.56.2258.27*, %struct._object.57.2259.28*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*, %struct._object.57.2259.28*)*, i32 (%struct._object.57.2259.28*, %struct._object.57.2259.28*, %struct._object.57.2259.28*)*, i64, i32 (%struct._object.57.2259.28*, %struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._typeobject.56.2258.27*, i64)*, %struct._object.57.2259.28* (%struct._typeobject.56.2258.27*, %struct._object.57.2259.28*, %struct._object.57.2259.28*)*, void (i8*)*, i32 (%struct._object.57.2259.28*)*, %struct._object.57.2259.28*, %struct._object.57.2259.28*, %struct._object.57.2259.28*, %struct._object.57.2259.28*, %struct._object.57.2259.28*, void (%struct._object.57.2259.28*)*, i32 }
%struct.__sFILE.47.2249.18 = type { i8*, i32, i32, i16, i16, %struct.__sbuf.45.2247.16, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf.45.2247.16, %struct.__sFILEX.46.2248.17*, i32, [3 x i8], [1 x i8], %struct.__sbuf.45.2247.16, i32, i64 }
%struct.__sFILEX.46.2248.17 = type opaque
%struct.__sbuf.45.2247.16 = type { i8*, i32 }
%struct.PyNumberMethods.48.2250.19 = type { %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*)*, i32 (%struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, i32 (%struct._object.57.2259.28**, %struct._object.57.2259.28**)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*)* }
%struct.PySequenceMethods.49.2251.20 = type { i64 (%struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, i64)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, i64)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, i64, i64)*, i32 (%struct._object.57.2259.28*, i64, %struct._object.57.2259.28*)*, i32 (%struct._object.57.2259.28*, i64, i64, %struct._object.57.2259.28*)*, i32 (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, i64)* }
%struct.PyMappingMethods.50.2252.21 = type { i64 (%struct._object.57.2259.28*)*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, i32 (%struct._object.57.2259.28*, %struct._object.57.2259.28*, %struct._object.57.2259.28*)* }
%struct.PyBufferProcs.52.2254.23 = type { i64 (%struct._object.57.2259.28*, i64, i8**)*, i64 (%struct._object.57.2259.28*, i64, i8**)*, i64 (%struct._object.57.2259.28*, i64*)*, i64 (%struct._object.57.2259.28*, i64, i8**)*, i32 (%struct._object.57.2259.28*, %struct.bufferinfo.51.2253.22*, i32)*, void (%struct._object.57.2259.28*, %struct.bufferinfo.51.2253.22*)* }
%struct.bufferinfo.51.2253.22 = type { i8*, %struct._object.57.2259.28*, i64, i64, i32, i32, i8*, i64*, i64*, i64*, [2 x i64], i8* }
%struct.PyMethodDef.53.2255.24 = type { i8*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, %struct._object.57.2259.28*)*, i32, i8* }
%struct.PyMemberDef.54.2256.25 = type opaque
%struct.PyGetSetDef.55.2257.26 = type { i8*, %struct._object.57.2259.28* (%struct._object.57.2259.28*, i8*)*, i32 (%struct._object.57.2259.28*, %struct._object.57.2259.28*, i8*)*, i8*, i8* }

define void @_ZN22LPush2MidiRemoteScript36NewIpcChannelAndLogAndCrashFileNamesEv() {
  call void @_ZN13ADebugOptions9GetOptionEPKc()
  %tmp47 = add i32 0, 0, !dbg !14
  unreachable
}

declare void @_ZN13ADebugOptions9GetOptionEPKc() #0

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!60}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "Apple LLVM version 8.0.0 (clang-800.0.24.1)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, imports: !2)
!1 = !DIFile(filename: "/Stk/Ableton/img/Live/master/Projects/IceSong/Unity/IceSong_Unity_Src_cpp-23.cpp", directory: "/Stk/Ableton/img/Live/master")
!2 = !{!3}
!3 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !8, line: 39)
!4 = !DINamespace(name: "python", scope: !6, file: !5, line: 60)
!5 = !DIFile(filename: "Projects/IceSong/Exp/MidiRemoteScript.h", directory: "/Stk/Ableton/img/Live/master")
!6 = !DINamespace(name: "boost", scope: null, file: !7, line: 482)
!7 = !DIFile(filename: "modules/boost/include/boost/config/suffix.hpp", directory: "/Stk/Ableton/img/Live/master")
!8 = distinct !DIGlobalVariable(name: "_", linkageName: "_ZN5boost6python3apiL1_E", scope: !9, file: !10, line: 20, type: !11, isLocal: true, isDefinition: true, variable: %"class.boost::python::api::slice_nil.60.2301.31"* undef)
!9 = !DINamespace(name: "api", scope: !4, file: !5, line: 61)
!10 = !DIFile(filename: "modules/boost/include/boost/python/slice_nil.hpp", directory: "/Stk/Ableton/img/Live/master")
!11 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !12)
!12 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "slice_nil", scope: !9, file: !10, line: 13, size: 64, align: 64, elements: !13, identifier: "_ZTSN5boost6python3api9slice_nilE")
!13 = !{}
!14 = !DILocation(line: 139, column: 42, scope: !15, inlinedAt: !21)
!15 = distinct !DISubprogram(name: "address", linkageName: "_ZN5boost15optional_detail15aligned_storageI12TDebugOptionE7addressEv", scope: !17, file: !16, line: 139, type: !19, isLocal: false, isDefinition: true, scopeLine: 139, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !20, variables: !13)
!16 = !DIFile(filename: "modules/boost/include/boost/optional/optional.hpp", directory: "/Stk/Ableton/img/Live/master")
!17 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "aligned_storage<TDebugOption>", scope: !18, file: !16, line: 120, size: 128, align: 64, elements: !13, templateParams: !13, identifier: "_ZTSN5boost15optional_detail15aligned_storageI12TDebugOptionEE")
!18 = !DINamespace(name: "optional_detail", scope: !6, file: !16, line: 114)
!19 = !DISubroutineType(types: !13)
!20 = !DISubprogram(name: "address", linkageName: "_ZN5boost15optional_detail15aligned_storageI12TDebugOptionE7addressEv", scope: !17, file: !16, line: 139, type: !19, isLocal: false, isDefinition: false, scopeLine: 139, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: true)
!21 = distinct !DILocation(line: 728, column: 71, scope: !22, inlinedAt: !28)
!22 = distinct !DISubprogram(name: "get_object", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionE10get_objectEv", scope: !23, file: !16, line: 726, type: !19, isLocal: false, isDefinition: true, scopeLine: 727, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !24, variables: !25)
!23 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "optional_base<TDebugOption>", scope: !18, file: !16, line: 197, size: 192, align: 64, elements: !13, templateParams: !13, identifier: "_ZTSN5boost15optional_detail13optional_baseI12TDebugOptionEE")
!24 = !DISubprogram(name: "get_object", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionE10get_objectEv", scope: !23, file: !16, line: 726, type: !19, isLocal: false, isDefinition: false, scopeLine: 726, flags: DIFlagPrototyped, isOptimized: true)
!25 = !{!26}
!26 = !DILocalVariable(name: "caster", scope: !22, file: !16, line: 728, type: !27)
!27 = distinct !DICompositeType(tag: DW_TAG_union_type, scope: !22, file: !16, line: 728, size: 64, align: 64, elements: !13, identifier: "_ZTSZN5boost15optional_detail13optional_baseI12TDebugOptionE10get_objectEvEUt_")
!28 = distinct !DILocation(line: 714, column: 63, scope: !29, inlinedAt: !31)
!29 = distinct !DISubprogram(name: "get_ptr_impl", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionE12get_ptr_implEv", scope: !23, file: !16, line: 714, type: !19, isLocal: false, isDefinition: true, scopeLine: 714, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !30, variables: !13)
!30 = !DISubprogram(name: "get_ptr_impl", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionE12get_ptr_implEv", scope: !23, file: !16, line: 714, type: !19, isLocal: false, isDefinition: false, scopeLine: 714, flags: DIFlagProtected | DIFlagPrototyped, isOptimized: true)
!31 = distinct !DILocation(line: 745, column: 50, scope: !32, inlinedAt: !34)
!32 = distinct !DISubprogram(name: "destroy_impl", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionE12destroy_implEN4mpl_5bool_ILb0EEE", scope: !23, file: !16, line: 745, type: !19, isLocal: false, isDefinition: true, scopeLine: 745, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !33, variables: !13)
!33 = !DISubprogram(name: "destroy_impl", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionE12destroy_implEN4mpl_5bool_ILb0EEE", scope: !23, file: !16, line: 745, type: !19, isLocal: false, isDefinition: false, scopeLine: 745, flags: DIFlagPrototyped, isOptimized: true)
!34 = distinct !DILocation(line: 707, column: 9, scope: !35, inlinedAt: !38)
!35 = distinct !DILexicalBlock(scope: !36, file: !16, line: 706, column: 12)
!36 = distinct !DISubprogram(name: "destroy", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionE7destroyEv", scope: !23, file: !16, line: 704, type: !19, isLocal: false, isDefinition: true, scopeLine: 705, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !37, variables: !13)
!37 = !DISubprogram(name: "destroy", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionE7destroyEv", scope: !23, file: !16, line: 704, type: !19, isLocal: false, isDefinition: false, scopeLine: 704, flags: DIFlagProtected | DIFlagPrototyped, isOptimized: true)
!38 = distinct !DILocation(line: 327, column: 24, scope: !39, inlinedAt: !42)
!39 = distinct !DILexicalBlock(scope: !40, file: !16, line: 327, column: 22)
!40 = distinct !DISubprogram(name: "~optional_base", linkageName: "_ZN5boost15optional_detail13optional_baseI12TDebugOptionED2Ev", scope: !23, file: !16, line: 327, type: !19, isLocal: false, isDefinition: true, scopeLine: 327, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !41, variables: !13)
!41 = !DISubprogram(name: "~optional_base", scope: !23, file: !16, line: 327, type: !19, isLocal: false, isDefinition: false, scopeLine: 327, flags: DIFlagProtected | DIFlagPrototyped, isOptimized: true)
!42 = distinct !DILocation(line: 877, column: 18, scope: !43, inlinedAt: !47)
!43 = distinct !DILexicalBlock(scope: !44, file: !16, line: 877, column: 17)
!44 = distinct !DISubprogram(name: "~optional", linkageName: "_ZN5boost8optionalI12TDebugOptionED2Ev", scope: !45, file: !16, line: 877, type: !19, isLocal: false, isDefinition: true, scopeLine: 877, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !46, variables: !13)
!45 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "optional<TDebugOption>", scope: !6, file: !16, line: 765, size: 192, align: 64, elements: !13, templateParams: !13, identifier: "_ZTSN5boost8optionalI12TDebugOptionEE")
!46 = !DISubprogram(name: "~optional", scope: !45, file: !16, line: 877, type: !19, isLocal: false, isDefinition: false, scopeLine: 877, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: true)
!47 = distinct !DILocation(line: 877, column: 17, scope: !48, inlinedAt: !49)
!48 = distinct !DISubprogram(name: "~optional", linkageName: "_ZN5boost8optionalI12TDebugOptionED1Ev", scope: !45, file: !16, line: 877, type: !19, isLocal: false, isDefinition: true, scopeLine: 877, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !46, variables: !13)
!49 = distinct !DILocation(line: 87, column: 9, scope: !50)
!50 = distinct !DILexicalBlock(scope: !52, file: !51, line: 87, column: 9)
!51 = !DIFile(filename: "Projects/IceSong/Src/Push2MidiRemoteScript.cpp", directory: "/Stk/Ableton/img/Live/master")
!52 = distinct !DILexicalBlock(scope: !53, file: !51, line: 82, column: 3)
!53 = distinct !DILexicalBlock(scope: !54, file: !51, line: 81, column: 7)
!54 = distinct !DISubprogram(name: "NewIpcChannelAndLogAndCrashFileNames", linkageName: "_ZN22LPush2MidiRemoteScript36NewIpcChannelAndLogAndCrashFileNamesEv", scope: !55, file: !51, line: 73, type: !19, isLocal: false, isDefinition: true, scopeLine: 74, flags: DIFlagPrototyped, isOptimized: true, unit: !0, declaration: !59, variables: !13)
!55 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "LPush2MidiRemoteScript", file: !56, line: 53, size: 5440, align: 64, elements: !13, vtableHolder: !57, identifier: "_ZTS22LPush2MidiRemoteScript")
!56 = !DIFile(filename: "Projects/IceSong/Exp/Push2MidiRemoteScript.h", directory: "/Stk/Ableton/img/Live/master")
!57 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "AObject", file: !58, line: 207, size: 64, align: 64, elements: !13, vtableHolder: !57, identifier: "_ZTS7AObject")
!58 = !DIFile(filename: "Projects/CreamOs/Exp/NewWithoutES.h", directory: "/Stk/Ableton/img/Live/master")
!59 = !DISubprogram(name: "NewIpcChannelAndLogAndCrashFileNames", linkageName: "_ZN22LPush2MidiRemoteScript36NewIpcChannelAndLogAndCrashFileNamesEv", scope: !55, file: !56, line: 83, type: !19, isLocal: false, isDefinition: false, scopeLine: 83, flags: DIFlagPrototyped, isOptimized: true)
!60 = !{i32 2, !"Debug Info Version", i32 700000003}

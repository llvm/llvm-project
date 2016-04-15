; RUN: opt -objc-arc -S < %s | FileCheck %s
; rdar://11744105
; bugzilla://14584

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%0 = type opaque
%struct._class_t = type { %struct._class_t*, %struct._class_t*, %struct._objc_cache*, i8* (i8*, i8*)**, %struct._class_ro_t* }
%struct._objc_cache = type opaque
%struct._class_ro_t = type { i32, i32, i32, i8*, i8*, %struct.__method_list_t*, %struct._objc_protocol_list*, %struct._ivar_list_t*, i8*, %struct._prop_list_t* }
%struct.__method_list_t = type { i32, i32, [0 x %struct._objc_method] }
%struct._objc_method = type { i8*, i8*, i8* }
%struct._objc_protocol_list = type { i64, [0 x %struct._protocol_t*] }
%struct._protocol_t = type { i8*, i8*, %struct._objc_protocol_list*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct._prop_list_t*, i32, i32, i8** }
%struct._prop_list_t = type { i32, i32, [0 x %struct._prop_t] }
%struct._prop_t = type { i8*, i8* }
%struct._ivar_list_t = type { i32, i32, [0 x %struct._ivar_t] }
%struct._ivar_t = type { i64*, i8*, i8*, i32, i32 }
%struct.NSConstantString = type { i32*, i32, i8*, i64 }

@"OBJC_CLASS_$_NSObject" = external global %struct._class_t
@"\01L_OBJC_CLASSLIST_REFERENCES_$_" = internal global %struct._class_t* @"OBJC_CLASS_$_NSObject", section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_METH_VAR_NAME_" = internal global [4 x i8] c"new\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_SELECTOR_REFERENCES_" = internal global i8* getelementptr inbounds ([4 x i8], [4 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i64 0, i64 0), section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@__CFConstantStringClassReference = external global [0 x i32]
@.str = private unnamed_addr constant [11 x i8] c"Failed: %@\00", align 1
@_unnamed_cfstring_ = private constant %struct.NSConstantString { i32* getelementptr inbounds ([0 x i32], [0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i32 0, i32 0), i64 10 }, section "__DATA,__cfstring"
@"OBJC_CLASS_$_NSException" = external global %struct._class_t
@"\01L_OBJC_CLASSLIST_REFERENCES_$_1" = internal global %struct._class_t* @"OBJC_CLASS_$_NSException", section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@.str2 = private unnamed_addr constant [4 x i8] c"Foo\00", align 1
@_unnamed_cfstring_3 = private constant %struct.NSConstantString { i32* getelementptr inbounds ([0 x i32], [0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str2, i32 0, i32 0), i64 3 }, section "__DATA,__cfstring"
@"\01L_OBJC_METH_VAR_NAME_4" = internal global [14 x i8] c"raise:format:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_SELECTOR_REFERENCES_5" = internal global i8* getelementptr inbounds ([14 x i8], [14 x i8]* @"\01L_OBJC_METH_VAR_NAME_4", i64 0, i64 0), section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@llvm.used = appending global [6 x i8*] [i8* bitcast (%struct._class_t** @"\01L_OBJC_CLASSLIST_REFERENCES_$_" to i8*), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), i8* bitcast (i8** @"\01L_OBJC_SELECTOR_REFERENCES_" to i8*), i8* bitcast (%struct._class_t** @"\01L_OBJC_CLASSLIST_REFERENCES_$_1" to i8*), i8* getelementptr inbounds ([14 x i8], [14 x i8]* @"\01L_OBJC_METH_VAR_NAME_4", i32 0, i32 0), i8* bitcast (i8** @"\01L_OBJC_SELECTOR_REFERENCES_5" to i8*)], section "llvm.metadata"

define i32 @main() uwtable ssp personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*) !dbg !5 {
entry:
  %tmp = load %struct._class_t*, %struct._class_t** @"\01L_OBJC_CLASSLIST_REFERENCES_$_", align 8, !dbg !37
  %tmp1 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8, !dbg !37, !invariant.load !38
  %tmp2 = bitcast %struct._class_t* %tmp to i8*, !dbg !37
; CHECK: call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* %tmp2, i8* %tmp1)
  %call = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* %tmp2, i8* %tmp1), !dbg !37, !clang.arc.no_objc_arc_exceptions !38
  call void @llvm.dbg.value(metadata i8* %call, i64 0, metadata !25, metadata !DIExpression()), !dbg !37
; CHECK: call i8* @objc_retain(i8* %call) [[NUW:#[0-9]+]]
  %tmp3 = call i8* @objc_retain(i8* %call) nounwind, !dbg !39
  call void @llvm.dbg.value(metadata i8* %call, i64 0, metadata !25, metadata !DIExpression()), !dbg !39
  invoke fastcc void @ThrowFunc(i8* %call)
          to label %eh.cont unwind label %lpad, !dbg !40, !clang.arc.no_objc_arc_exceptions !38

eh.cont:                                          ; preds = %entry
; CHECK: call void @objc_release(i8* %call)
  call void @objc_release(i8* %call) nounwind, !dbg !42, !clang.imprecise_release !38
  br label %if.end, !dbg !43

lpad:                                             ; preds = %entry
  %tmp4 = landingpad { i8*, i32 }
          catch i8* null, !dbg !40
  %tmp5 = extractvalue { i8*, i32 } %tmp4, 0, !dbg !40
  %exn.adjusted = call i8* @objc_begin_catch(i8* %tmp5) nounwind, !dbg !44
  call void @llvm.dbg.value(metadata i8 0, i64 0, metadata !21, metadata !DIExpression()), !dbg !46
  call void @objc_end_catch(), !dbg !49, !clang.arc.no_objc_arc_exceptions !38
; CHECK: call void @objc_release(i8* %call)
  call void @objc_release(i8* %call) nounwind, !dbg !42, !clang.imprecise_release !38
  call void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring_ to i8*), i8* %call), !dbg !50, !clang.arc.no_objc_arc_exceptions !38
  br label %if.end, !dbg !52

if.end:                                           ; preds = %lpad, %eh.cont
  call void (i8*, ...) @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring_ to i8*), i8* %call), !dbg !53, !clang.arc.no_objc_arc_exceptions !38
; CHECK: call void @objc_release(i8* %call)
  call void @objc_release(i8* %call) nounwind, !dbg !54, !clang.imprecise_release !38
  ret i32 0, !dbg !54
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i8* @objc_msgSend(i8*, i8*, ...) nonlazybind

declare i8* @objc_retain(i8*) nonlazybind

declare i8* @objc_begin_catch(i8*)

declare void @objc_end_catch()

declare void @objc_exception_rethrow()

define internal fastcc void @ThrowFunc(i8* %obj) uwtable noinline ssp !dbg !27 {
entry:
  %tmp = call i8* @objc_retain(i8* %obj) nounwind
  call void @llvm.dbg.value(metadata i8* %obj, i64 0, metadata !32, metadata !DIExpression()), !dbg !55
  %tmp1 = load %struct._class_t*, %struct._class_t** @"\01L_OBJC_CLASSLIST_REFERENCES_$_1", align 8, !dbg !56
  %tmp2 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_5", align 8, !dbg !56, !invariant.load !38
  %tmp3 = bitcast %struct._class_t* %tmp1 to i8*, !dbg !56
  call void (i8*, i8*, %0*, %0*, ...) bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, %0*, %0*, ...)*)(i8* %tmp3, i8* %tmp2, %0* bitcast (%struct.NSConstantString* @_unnamed_cfstring_3 to %0*), %0* bitcast (%struct.NSConstantString* @_unnamed_cfstring_3 to %0*)), !dbg !56, !clang.arc.no_objc_arc_exceptions !38
  call void @objc_release(i8* %obj) nounwind, !dbg !58, !clang.imprecise_release !38
  ret void, !dbg !58
}

declare i32 @__objc_personality_v0(...)

declare void @objc_release(i8*) nonlazybind

declare void @NSLog(i8*, ...)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

; CHECK: attributes #0 = { ssp uwtable }
; CHECK: attributes #1 = { nounwind readnone }
; CHECK: attributes #2 = { nonlazybind }
; CHECK: attributes #3 = { noinline ssp uwtable }
; CHECK: attributes [[NUW]] = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33, !34, !35, !36, !61}

!0 = distinct !DICompileUnit(language: DW_LANG_ObjC, producer: "clang version 3.3 ", isOptimized: true, runtimeVersion: 2, emissionKind: FullDebug, file: !60, enums: !1, retainedTypes: !1, globals: !1)
!1 = !{}
!5 = distinct !DISubprogram(name: "main", line: 9, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !0, scopeLine: 10, file: !60, scope: !6, type: !7, variables: !11)
!6 = !DIFile(filename: "test.m", directory: "/Volumes/Files/gottesmmcab/Radar/12906997")
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12, !21, !25}
!12 = !DILocalVariable(name: "obj", line: 11, scope: !13, file: !6, type: !14)
!13 = distinct !DILexicalBlock(line: 10, column: 0, file: !60, scope: !5)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "id", line: 11, file: !60, baseType: !15)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !60, baseType: !16)
!16 = !DICompositeType(tag: DW_TAG_structure_type, name: "objc_object", file: !60, elements: !17)
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "isa", size: 64, file: !60, scope: !16, baseType: !19)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, baseType: !20)
!20 = !DICompositeType(tag: DW_TAG_structure_type, name: "objc_class", flags: DIFlagFwdDecl, file: !60)
!21 = !DILocalVariable(name: "ok", line: 13, scope: !22, file: !6, type: !23)
!22 = distinct !DILexicalBlock(line: 12, column: 0, file: !60, scope: !13)
!23 = !DIDerivedType(tag: DW_TAG_typedef, name: "BOOL", line: 62, file: !60, baseType: !24)
!24 = !DIBasicType(tag: DW_TAG_base_type, name: "signed char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!25 = !DILocalVariable(name: "obj2", line: 15, scope: !26, file: !6, type: !14)
!26 = distinct !DILexicalBlock(line: 14, column: 0, file: !60, scope: !22)
!27 = distinct !DISubprogram(name: "ThrowFunc", line: 4, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 5, file: !60, scope: !6, type: !28, variables: !31)
!28 = !DISubroutineType(types: !29)
!29 = !{null, !14}
!31 = !{!32}
!32 = !DILocalVariable(name: "obj", line: 4, arg: 1, scope: !27, file: !6, type: !14)
!33 = !{i32 1, !"Objective-C Version", i32 2}
!34 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!35 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!36 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
!37 = !DILocation(line: 11, scope: !13)
!38 = !{}
!39 = !DILocation(line: 15, scope: !26)
!40 = !DILocation(line: 17, scope: !41)
!41 = distinct !DILexicalBlock(line: 16, column: 0, file: !60, scope: !26)
!42 = !DILocation(line: 22, scope: !26)
!43 = !DILocation(line: 23, scope: !22)
!44 = !DILocation(line: 19, scope: !41)
!45 = !{i8 0}
!46 = !DILocation(line: 20, scope: !47)
!47 = distinct !DILexicalBlock(line: 19, column: 0, file: !60, scope: !48)
!48 = distinct !DILexicalBlock(line: 19, column: 0, file: !60, scope: !26)
!49 = !DILocation(line: 21, scope: !47)
!50 = !DILocation(line: 24, scope: !51)
!51 = distinct !DILexicalBlock(line: 23, column: 0, file: !60, scope: !22)
!52 = !DILocation(line: 25, scope: !51)
!53 = !DILocation(line: 27, scope: !13)
!54 = !DILocation(line: 28, scope: !13)
!55 = !DILocation(line: 4, scope: !27)
!56 = !DILocation(line: 6, scope: !57)
!57 = distinct !DILexicalBlock(line: 5, column: 0, file: !60, scope: !27)
!58 = !DILocation(line: 7, scope: !57)
!60 = !DIFile(filename: "test.m", directory: "/Volumes/Files/gottesmmcab/Radar/12906997")
!61 = !{i32 1, !"Debug Info Version", i32 3}

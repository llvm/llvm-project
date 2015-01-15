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
@"\01L_OBJC_SELECTOR_REFERENCES_" = internal global i8* getelementptr inbounds ([4 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i64 0, i64 0), section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@__CFConstantStringClassReference = external global [0 x i32]
@.str = private unnamed_addr constant [11 x i8] c"Failed: %@\00", align 1
@_unnamed_cfstring_ = private constant %struct.NSConstantString { i32* getelementptr inbounds ([0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i64 10 }, section "__DATA,__cfstring"
@"OBJC_CLASS_$_NSException" = external global %struct._class_t
@"\01L_OBJC_CLASSLIST_REFERENCES_$_1" = internal global %struct._class_t* @"OBJC_CLASS_$_NSException", section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@.str2 = private unnamed_addr constant [4 x i8] c"Foo\00", align 1
@_unnamed_cfstring_3 = private constant %struct.NSConstantString { i32* getelementptr inbounds ([0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr inbounds ([4 x i8]* @.str2, i32 0, i32 0), i64 3 }, section "__DATA,__cfstring"
@"\01L_OBJC_METH_VAR_NAME_4" = internal global [14 x i8] c"raise:format:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_SELECTOR_REFERENCES_5" = internal global i8* getelementptr inbounds ([14 x i8]* @"\01L_OBJC_METH_VAR_NAME_4", i64 0, i64 0), section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@llvm.used = appending global [6 x i8*] [i8* bitcast (%struct._class_t** @"\01L_OBJC_CLASSLIST_REFERENCES_$_" to i8*), i8* getelementptr inbounds ([4 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), i8* bitcast (i8** @"\01L_OBJC_SELECTOR_REFERENCES_" to i8*), i8* bitcast (%struct._class_t** @"\01L_OBJC_CLASSLIST_REFERENCES_$_1" to i8*), i8* getelementptr inbounds ([14 x i8]* @"\01L_OBJC_METH_VAR_NAME_4", i32 0, i32 0), i8* bitcast (i8** @"\01L_OBJC_SELECTOR_REFERENCES_5" to i8*)], section "llvm.metadata"

define i32 @main() uwtable ssp {
entry:
  %tmp = load %struct._class_t** @"\01L_OBJC_CLASSLIST_REFERENCES_$_", align 8, !dbg !37
  %tmp1 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", align 8, !dbg !37, !invariant.load !38
  %tmp2 = bitcast %struct._class_t* %tmp to i8*, !dbg !37
; CHECK: call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* %tmp2, i8* %tmp1)
  %call = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* %tmp2, i8* %tmp1), !dbg !37, !clang.arc.no_objc_arc_exceptions !38
  call void @llvm.dbg.value(metadata i8* %call, i64 0, metadata i32 02, metadata !{}), !dbg !37
; CHECK: call i8* @objc_retain(i8* %call) [[NUW:#[0-9]+]]
  %tmp3 = call i8* @objc_retain(i8* %call) nounwind, !dbg !39
  call void @llvm.dbg.value(metadata i8* %call, i64 0, metadata !25, metadata !{}), !dbg !39
  invoke fastcc void @ThrowFunc(i8* %call)
          to label %eh.cont unwind label %lpad, !dbg !40, !clang.arc.no_objc_arc_exceptions !38

eh.cont:                                          ; preds = %entry
; CHECK: call void @objc_release(i8* %call)
  call void @objc_release(i8* %call) nounwind, !dbg !42, !clang.imprecise_release !38
  br label %if.end, !dbg !43

lpad:                                             ; preds = %entry
  %tmp4 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*)
          catch i8* null, !dbg !40
  %tmp5 = extractvalue { i8*, i32 } %tmp4, 0, !dbg !40
  %exn.adjusted = call i8* @objc_begin_catch(i8* %tmp5) nounwind, !dbg !44
  call void @llvm.dbg.value(metadata i8 0, i64 0, metadata !21, metadata !{}), !dbg !46
  call void @objc_end_catch(), !dbg !49, !clang.arc.no_objc_arc_exceptions !38
; CHECK: call void @objc_release(i8* %call)
  call void @objc_release(i8* %call) nounwind, !dbg !42, !clang.imprecise_release !38
  call void (i8*, ...)* @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring_ to i8*), i8* %call), !dbg !50, !clang.arc.no_objc_arc_exceptions !38
  br label %if.end, !dbg !52

if.end:                                           ; preds = %lpad, %eh.cont
  call void (i8*, ...)* @NSLog(i8* bitcast (%struct.NSConstantString* @_unnamed_cfstring_ to i8*), i8* %call), !dbg !53, !clang.arc.no_objc_arc_exceptions !38
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

define internal fastcc void @ThrowFunc(i8* %obj) uwtable noinline ssp {
entry:
  %tmp = call i8* @objc_retain(i8* %obj) nounwind
  call void @llvm.dbg.value(metadata i8* %obj, i64 0, metadata !32, metadata !{}), !dbg !55
  %tmp1 = load %struct._class_t** @"\01L_OBJC_CLASSLIST_REFERENCES_$_1", align 8, !dbg !56
  %tmp2 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_5", align 8, !dbg !56, !invariant.load !38
  %tmp3 = bitcast %struct._class_t* %tmp1 to i8*, !dbg !56
  call void (i8*, i8*, %0*, %0*, ...)* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, %0*, %0*, ...)*)(i8* %tmp3, i8* %tmp2, %0* bitcast (%struct.NSConstantString* @_unnamed_cfstring_3 to %0*), %0* bitcast (%struct.NSConstantString* @_unnamed_cfstring_3 to %0*)), !dbg !56, !clang.arc.no_objc_arc_exceptions !38
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

!0 = !{!"0x11\0016\00clang version 3.3 \001\00\002\00\000", !60, !1, !1, !3, !1, null} ; [ DW_TAG_compile_unit ] [/Volumes/Files/gottesmmcab/Radar/12906997/test.m] [DW_LANG_ObjC]
!1 = !{i32 0}
!3 = !{!5, !27}
!5 = !{!"0x2e\00main\00main\00\009\000\001\000\006\000\001\0010", !60, !6, !7, null, i32 ()* @main, null, null, !10} ; [ DW_TAG_subprogram ] [line 9] [def] [scope 10] [main]
!6 = !{!"0x29", !60} ; [ DW_TAG_file_type ]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{!9}
!9 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = !{!11}
!11 = !{!12, !21, !25}
!12 = !{!"0x100\00obj\0011\000", !13, !6, !14} ; [ DW_TAG_auto_variable ] [obj] [line 11]
!13 = !{!"0xb\0010\000\000", !60, !5} ; [ DW_TAG_lexical_block ] [/Volumes/Files/gottesmmcab/Radar/12906997/test.m]
!14 = !{!"0x16\00id\0011\000\000\000\000", !60, null, !15} ; [ DW_TAG_typedef ] [id] [line 11, size 0, align 0, offset 0] [from ]
!15 = !{!"0xf\00\000\0064\0064\000\000", !60, null, !16} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from objc_object]
!16 = !{!"0x13\00objc_object\000\000\000\000\000\000", !60, null, null, !17, null, i32 0, null} ; [ DW_TAG_structure_type ] [objc_object] [line 0, size 0, align 0, offset 0] [def] [from ]
!17 = !{!18}
!18 = !{!"0xd\00isa\000\0064\000\000\000", !60, !16, !19} ; [ DW_TAG_member ] [isa] [line 0, size 64, align 0, offset 0] [from ]
!19 = !{!"0xf\00\000\0064\000\000\000", null, null, !20} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from objc_class]
!20 = !{!"0x13\00objc_class\000\000\000\000\004\000", !60, null, null, null, null, null, null} ; [ DW_TAG_structure_type ] [objc_class] [line 0, size 0, align 0, offset 0] [decl] [from ]
!21 = !{!"0x100\00ok\0013\000", !22, !6, !23} ; [ DW_TAG_auto_variable ] [ok] [line 13]
!22 = !{!"0xb\0012\000\001", !60, !13} ; [ DW_TAG_lexical_block ] [/Volumes/Files/gottesmmcab/Radar/12906997/test.m]
!23 = !{!"0x16\00BOOL\0062\000\000\000\000", !60, null, !24} ; [ DW_TAG_typedef ] [BOOL] [line 62, size 0, align 0, offset 0] [from signed char]
!24 = !{!"0x24\00signed char\000\008\008\000\000\006", null, null} ; [ DW_TAG_base_type ] [signed char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!25 = !{!"0x100\00obj2\0015\000", !26, !6, !14} ; [ DW_TAG_auto_variable ] [obj2] [line 15]
!26 = !{!"0xb\0014\000\002", !60, !22} ; [ DW_TAG_lexical_block ] [/Volumes/Files/gottesmmcab/Radar/12906997/test.m]
!27 = !{!"0x2e\00ThrowFunc\00ThrowFunc\00\004\001\001\000\006\00256\001\005", !60, !6, !28, null, void (i8*)* @ThrowFunc, null, null, !30} ; [ DW_TAG_subprogram ] [line 4] [local] [def] [scope 5] [ThrowFunc]
!28 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !29, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!29 = !{null, !14}
!30 = !{!31}
!31 = !{!32}
!32 = !{!"0x101\00obj\0016777220\000", !27, !6, !14} ; [ DW_TAG_arg_variable ] [obj] [line 4]
!33 = !{i32 1, !"Objective-C Version", i32 2}
!34 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!35 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!36 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
!37 = !MDLocation(line: 11, scope: !13)
!38 = !{}
!39 = !MDLocation(line: 15, scope: !26)
!40 = !MDLocation(line: 17, scope: !41)
!41 = !{!"0xb\0016\000\003", !60, !26} ; [ DW_TAG_lexical_block ] [/Volumes/Files/gottesmmcab/Radar/12906997/test.m]
!42 = !MDLocation(line: 22, scope: !26)
!43 = !MDLocation(line: 23, scope: !22)
!44 = !MDLocation(line: 19, scope: !41)
!45 = !{i8 0}
!46 = !MDLocation(line: 20, scope: !47)
!47 = !{!"0xb\0019\000\005", !60, !48} ; [ DW_TAG_lexical_block ] [/Volumes/Files/gottesmmcab/Radar/12906997/test.m]
!48 = !{!"0xb\0019\000\004", !60, !26} ; [ DW_TAG_lexical_block ] [/Volumes/Files/gottesmmcab/Radar/12906997/test.m]
!49 = !MDLocation(line: 21, scope: !47)
!50 = !MDLocation(line: 24, scope: !51)
!51 = !{!"0xb\0023\000\006", !60, !22} ; [ DW_TAG_lexical_block ] [/Volumes/Files/gottesmmcab/Radar/12906997/test.m]
!52 = !MDLocation(line: 25, scope: !51)
!53 = !MDLocation(line: 27, scope: !13)
!54 = !MDLocation(line: 28, scope: !13)
!55 = !MDLocation(line: 4, scope: !27)
!56 = !MDLocation(line: 6, scope: !57)
!57 = !{!"0xb\005\000\007", !60, !27} ; [ DW_TAG_lexical_block ] [/Volumes/Files/gottesmmcab/Radar/12906997/test.m]
!58 = !MDLocation(line: 7, scope: !57)
!60 = !{!"test.m", !"/Volumes/Files/gottesmmcab/Radar/12906997"}
!61 = !{i32 1, !"Debug Info Version", i32 2}

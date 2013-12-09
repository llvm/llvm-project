; RUN: llc  -mtriple=x86_64-apple-macosx10.8.0 -O0 -filetype=obj -o %t %s
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; Test that we generate debug info for by-value struct args that are not used.
;
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_name {{.*}} "info"
;
; generated from
;
; typedef unsigned long NSUInteger;
; typedef struct
; {
;  NSUInteger width;
;  NSUInteger height;
;  double pixelAspect;
; } ImageInfo;
; @implementation Bitmap
; - (id)initWithCopy:(Bitmap *)otherBitmap
;            andInfo:(ImageInfo)info
;        andLength:(NSUInteger)length
; {
; }
; @end

; ModuleID = 't.mm'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%0 = type opaque
%struct._objc_cache = type opaque
%struct._class_t = type { %struct._class_t*, %struct._class_t*, %struct._objc_cache*, i8* (i8*, i8*)**, %struct._class_ro_t* }
%struct._class_ro_t = type { i32, i32, i32, i8*, i8*, %struct.__method_list_t*, %struct._objc_protocol_list*, %struct._ivar_list_t*, i8*, %struct._prop_list_t* }
%struct.__method_list_t = type { i32, i32, [0 x %struct._objc_method] }
%struct._objc_method = type { i8*, i8*, i8* }
%struct._objc_protocol_list = type { i64, [0 x %struct._protocol_t*] }
%struct._protocol_t = type { i8*, i8*, %struct._objc_protocol_list*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct._prop_list_t*, i32, i32, i8** }
%struct._prop_list_t = type { i32, i32, [0 x %struct._prop_t] }
%struct._prop_t = type { i8*, i8* }
%struct._ivar_list_t = type { i32, i32, [0 x %struct._ivar_t] }
%struct._ivar_t = type { i64*, i8*, i8*, i32, i32 }
%struct.ImageInfo = type { i64, i64, double }

@_objc_empty_cache = external global %struct._objc_cache
@_objc_empty_vtable = external global i8* (i8*, i8*)*
@"OBJC_CLASS_$_Bitmap" = global %struct._class_t { %struct._class_t* @"OBJC_METACLASS_$_Bitmap", %struct._class_t* null, %struct._objc_cache* @_objc_empty_cache, i8* (i8*, i8*)** @_objc_empty_vtable, %struct._class_ro_t* @"\01l_OBJC_CLASS_RO_$_Bitmap" }, section "__DATA, __objc_data", align 8
@"OBJC_METACLASS_$_Bitmap" = global %struct._class_t { %struct._class_t* @"OBJC_METACLASS_$_Bitmap", %struct._class_t* @"OBJC_CLASS_$_Bitmap", %struct._objc_cache* @_objc_empty_cache, i8* (i8*, i8*)** @_objc_empty_vtable, %struct._class_ro_t* @"\01l_OBJC_METACLASS_RO_$_Bitmap" }, section "__DATA, __objc_data", align 8
@"\01L_OBJC_CLASS_NAME_" = internal global [7 x i8] c"Bitmap\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@"\01l_OBJC_METACLASS_RO_$_Bitmap" = internal global %struct._class_ro_t { i32 3, i32 40, i32 40, i8* null, i8* getelementptr inbounds ([7 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), %struct.__method_list_t* null, %struct._objc_protocol_list* null, %struct._ivar_list_t* null, i8* null, %struct._prop_list_t* null }, section "__DATA, __objc_const", align 8
@"\01L_OBJC_METH_VAR_NAME_" = internal global [32 x i8] c"initWithCopy:andInfo:andLength:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_METH_VAR_TYPE_" = internal global [23 x i8] c"@56@0:8@16{?=QQd}24Q48\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@"\01l_OBJC_$_INSTANCE_METHODS_Bitmap" = internal global { i32, i32, [1 x %struct._objc_method] } { i32 24, i32 1, [1 x %struct._objc_method] [%struct._objc_method { i8* getelementptr inbounds ([32 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), i8* getelementptr inbounds ([23 x i8]* @"\01L_OBJC_METH_VAR_TYPE_", i32 0, i32 0), i8* bitcast (i8* (%0*, i8*, %0*, %struct.ImageInfo*, i64)* @"\01-[Bitmap initWithCopy:andInfo:andLength:]" to i8*) }] }, section "__DATA, __objc_const", align 8
@"\01l_OBJC_CLASS_RO_$_Bitmap" = internal global %struct._class_ro_t { i32 2, i32 0, i32 0, i8* null, i8* getelementptr inbounds ([7 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), %struct.__method_list_t* bitcast ({ i32, i32, [1 x %struct._objc_method] }* @"\01l_OBJC_$_INSTANCE_METHODS_Bitmap" to %struct.__method_list_t*), %struct._objc_protocol_list* null, %struct._ivar_list_t* null, i8* null, %struct._prop_list_t* null }, section "__DATA, __objc_const", align 8
@"\01L_OBJC_LABEL_CLASS_$" = internal global [1 x i8*] [i8* bitcast (%struct._class_t* @"OBJC_CLASS_$_Bitmap" to i8*)], section "__DATA, __objc_classlist, regular, no_dead_strip", align 8
@llvm.used = appending global [5 x i8*] [i8* getelementptr inbounds ([7 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), i8* getelementptr inbounds ([32 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), i8* getelementptr inbounds ([23 x i8]* @"\01L_OBJC_METH_VAR_TYPE_", i32 0, i32 0), i8* bitcast ({ i32, i32, [1 x %struct._objc_method] }* @"\01l_OBJC_$_INSTANCE_METHODS_Bitmap" to i8*), i8* bitcast ([1 x i8*]* @"\01L_OBJC_LABEL_CLASS_$" to i8*)], section "llvm.metadata"

; Function Attrs: ssp uwtable
define internal i8* @"\01-[Bitmap initWithCopy:andInfo:andLength:]"(%0* %self, i8* %_cmd, %0* %otherBitmap, %struct.ImageInfo* byval align 8 %info, i64 %length) #0 {
entry:
  %retval = alloca i8*, align 8
  %self.addr = alloca %0*, align 8
  %_cmd.addr = alloca i8*, align 8
  %otherBitmap.addr = alloca %0*, align 8
  %length.addr = alloca i64, align 8
  store %0* %self, %0** %self.addr, align 8
  call void @llvm.dbg.declare(metadata !{%0** %self.addr}, metadata !28), !dbg !29
  store i8* %_cmd, i8** %_cmd.addr, align 8
  call void @llvm.dbg.declare(metadata !{i8** %_cmd.addr}, metadata !30), !dbg !29
  store %0* %otherBitmap, %0** %otherBitmap.addr, align 8
  call void @llvm.dbg.declare(metadata !{%0** %otherBitmap.addr}, metadata !32), !dbg !29
  call void @llvm.dbg.declare(metadata !{%struct.ImageInfo* %info}, metadata !33), !dbg !34
  store i64 %length, i64* %length.addr, align 8
  call void @llvm.dbg.declare(metadata !{i64* %length.addr}, metadata !35), !dbg !36
  %0 = load i8** %retval, !dbg !37
  ret i8* %0, !dbg !37
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!24, !25, !26, !27, !38}

!0 = metadata !{i32 786449, metadata !1, i32 17, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 2, metadata !2, metadata !3, metadata !6, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/t.mm] [DW_LANG_ObjC_plus_plus]
!1 = metadata !{metadata !"t.mm", metadata !""}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786451, metadata !1, metadata !5, metadata !"Bitmap", i32 8, i64 8, i64 8, i32 0, i32 512, null, metadata !2, i32 17, null, null, null} ; [ DW_TAG_structure_type ] [Bitmap] [line 8, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/t.mm]
!6 = metadata !{metadata !7}
!7 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"-[Bitmap initWithCopy:andInfo:andLength:]", metadata !"-[Bitmap initWithCopy:andInfo:andLength:]", metadata !"", i32 9, metadata !8, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, i8* (%0*, i8*, %0*, %struct.ImageInfo*, i64)* @"\01-[Bitmap initWithCopy:andInfo:andLength:]", null, null, metadata !2, i32 9} ; [ DW_TAG_subprogram ] [line 9] [local] [def] [-[Bitmap initWithCopy:andInfo:andLength:]]
!8 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !9, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!9 = metadata !{metadata !4, metadata !10, metadata !11, metadata !14, metadata !15, metadata !19}
!10 = metadata !{i32 786447, i32 0, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !4} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from Bitmap]
!11 = metadata !{i32 786454, metadata !1, null, metadata !"SEL", i32 9, i64 0, i64 0, i64 0, i32 64, metadata !12} ; [ DW_TAG_typedef ] [SEL] [line 9, size 0, align 0, offset 0] [artificial] [from ]
!12 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !13} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from objc_selector]
!13 = metadata !{i32 786451, metadata !1, null, metadata !"objc_selector", i32 0, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [objc_selector] [line 0, size 0, align 0, offset 0] [decl] [from ]
!14 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !4} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from Bitmap]
!15 = metadata !{i32 786454, metadata !1, null, metadata !"ImageInfo", i32 7, i64 0, i64 0, i64 0, i32 0, metadata !16} ; [ DW_TAG_typedef ] [ImageInfo] [line 7, size 0, align 0, offset 0] [from ]
!16 = metadata !{i32 786451, metadata !1, null, metadata !"", i32 2, i64 192, i64 64, i32 0, i32 0, null, metadata !17, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [line 2, size 192, align 64, offset 0] [def] [from ]
!17 = metadata !{metadata !18, metadata !21, metadata !22}
!18 = metadata !{i32 786445, metadata !1, metadata !16, metadata !"width", i32 4, i64 64, i64 64, i64 0, i32 0, metadata !19} ; [ DW_TAG_member ] [width] [line 4, size 64, align 64, offset 0] [from NSUInteger]
!19 = metadata !{i32 786454, metadata !1, null, metadata !"NSUInteger", i32 1, i64 0, i64 0, i64 0, i32 0, metadata !20} ; [ DW_TAG_typedef ] [NSUInteger] [line 1, size 0, align 0, offset 0] [from long unsigned int]
!20 = metadata !{i32 786468, null, null, metadata !"long unsigned int", i32 0, i64 64, i64 64, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ] [long unsigned int] [line 0, size 64, align 64, offset 0, enc DW_ATE_unsigned]
!21 = metadata !{i32 786445, metadata !1, metadata !16, metadata !"height", i32 5, i64 64, i64 64, i64 64, i32 0, metadata !19} ; [ DW_TAG_member ] [height] [line 5, size 64, align 64, offset 64] [from NSUInteger]
!22 = metadata !{i32 786445, metadata !1, metadata !16, metadata !"pixelAspect", i32 6, i64 64, i64 64, i64 128, i32 0, metadata !23} ; [ DW_TAG_member ] [pixelAspect] [line 6, size 64, align 64, offset 128] [from double]
!23 = metadata !{i32 786468, null, null, metadata !"double", i32 0, i64 64, i64 64, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ] [double] [line 0, size 64, align 64, offset 0, enc DW_ATE_float]
!24 = metadata !{i32 1, metadata !"Objective-C Version", i32 2}
!25 = metadata !{i32 1, metadata !"Objective-C Image Info Version", i32 0}
!26 = metadata !{i32 1, metadata !"Objective-C Image Info Section", metadata !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!27 = metadata !{i32 4, metadata !"Objective-C Garbage Collection", i32 0}
!28 = metadata !{i32 786689, metadata !7, metadata !"self", metadata !5, i32 16777225, metadata !14, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [self] [line 9]
!29 = metadata !{i32 9, i32 0, metadata !7, null}
!30 = metadata !{i32 786689, metadata !7, metadata !"_cmd", metadata !5, i32 33554441, metadata !31, i32 64, i32 0} ; [ DW_TAG_arg_variable ] [_cmd] [line 9]
!31 = metadata !{i32 786454, metadata !1, null, metadata !"SEL", i32 9, i64 0, i64 0, i64 0, i32 0, metadata !12} ; [ DW_TAG_typedef ] [SEL] [line 9, size 0, align 0, offset 0] [from ]
!32 = metadata !{i32 786689, metadata !7, metadata !"otherBitmap", metadata !5, i32 50331657, metadata !14, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [otherBitmap] [line 9]
!33 = metadata !{i32 786689, metadata !7, metadata !"info", metadata !5, i32 67108874, metadata !15, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [info] [line 10]
!34 = metadata !{i32 10, i32 0, metadata !7, null}
!35 = metadata !{i32 786689, metadata !7, metadata !"length", metadata !5, i32 83886091, metadata !19, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [length] [line 11]
!36 = metadata !{i32 11, i32 0, metadata !7, null}
!37 = metadata !{i32 13, i32 0, metadata !7, null}
!38 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}

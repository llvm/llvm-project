; RUN: llc  -mtriple=x86_64-apple-macosx10.8.0 -O0 -filetype=obj -o %t %s
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s
; Test that we generate debug info for by-value struct args that are not used.
;
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}} "info"
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
%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }
%struct._class_ro_t = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct.__method_list_t = type { i32, i32, [0 x %struct._objc_method] }
%struct._objc_method = type { ptr, ptr, ptr }
%struct._objc_protocol_list = type { i64, [0 x ptr] }
%struct._protocol_t = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, ptr }
%struct._prop_list_t = type { i32, i32, [0 x %struct._prop_t] }
%struct._prop_t = type { ptr, ptr }
%struct._ivar_list_t = type { i32, i32, [0 x %struct._ivar_t] }
%struct._ivar_t = type { ptr, ptr, ptr, i32, i32 }
%struct.ImageInfo = type { i64, i64, double }

@_objc_empty_cache = external global %struct._objc_cache
@_objc_empty_vtable = external global ptr
@"OBJC_CLASS_$_Bitmap" = global %struct._class_t { ptr @"OBJC_METACLASS_$_Bitmap", ptr null, ptr @_objc_empty_cache, ptr @_objc_empty_vtable, ptr @"\01l_OBJC_CLASS_RO_$_Bitmap" }, section "__DATA, __objc_data", align 8
@"OBJC_METACLASS_$_Bitmap" = global %struct._class_t { ptr @"OBJC_METACLASS_$_Bitmap", ptr @"OBJC_CLASS_$_Bitmap", ptr @_objc_empty_cache, ptr @_objc_empty_vtable, ptr @"\01l_OBJC_METACLASS_RO_$_Bitmap" }, section "__DATA, __objc_data", align 8
@"\01L_OBJC_CLASS_NAME_" = internal global [7 x i8] c"Bitmap\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@"\01l_OBJC_METACLASS_RO_$_Bitmap" = internal global %struct._class_ro_t { i32 3, i32 40, i32 40, ptr null, ptr @"\01L_OBJC_CLASS_NAME_", ptr null, ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
@"\01L_OBJC_METH_VAR_NAME_" = internal global [32 x i8] c"initWithCopy:andInfo:andLength:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_METH_VAR_TYPE_" = internal global [23 x i8] c"@56@0:8@16{?=QQd}24Q48\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@"\01l_OBJC_$_INSTANCE_METHODS_Bitmap" = internal global { i32, i32, [1 x %struct._objc_method] } { i32 24, i32 1, [1 x %struct._objc_method] [%struct._objc_method { ptr @"\01L_OBJC_METH_VAR_NAME_", ptr @"\01L_OBJC_METH_VAR_TYPE_", ptr @"\01-[Bitmap initWithCopy:andInfo:andLength:]" }] }, section "__DATA, __objc_const", align 8
@"\01l_OBJC_CLASS_RO_$_Bitmap" = internal global %struct._class_ro_t { i32 2, i32 0, i32 0, ptr null, ptr @"\01L_OBJC_CLASS_NAME_", ptr @"\01l_OBJC_$_INSTANCE_METHODS_Bitmap", ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
@"\01L_OBJC_LABEL_CLASS_$" = internal global [1 x ptr] [ptr @"OBJC_CLASS_$_Bitmap"], section "__DATA, __objc_classlist, regular, no_dead_strip", align 8
@llvm.used = appending global [5 x ptr] [ptr @"\01L_OBJC_CLASS_NAME_", ptr @"\01L_OBJC_METH_VAR_NAME_", ptr @"\01L_OBJC_METH_VAR_TYPE_", ptr @"\01l_OBJC_$_INSTANCE_METHODS_Bitmap", ptr @"\01L_OBJC_LABEL_CLASS_$"], section "llvm.metadata"

; Function Attrs: ssp uwtable
define internal ptr @"\01-[Bitmap initWithCopy:andInfo:andLength:]"(ptr %self, ptr %_cmd, ptr %otherBitmap, ptr byval(%struct.ImageInfo) align 8 %info, i64 %length) #0 !dbg !7 {
entry:
  %retval = alloca ptr, align 8
  %self.addr = alloca ptr, align 8
  %_cmd.addr = alloca ptr, align 8
  %otherBitmap.addr = alloca ptr, align 8
  %length.addr = alloca i64, align 8
  store ptr %self, ptr %self.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %self.addr, metadata !28, metadata !DIExpression()), !dbg !29
  store ptr %_cmd, ptr %_cmd.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %_cmd.addr, metadata !30, metadata !DIExpression()), !dbg !29
  store ptr %otherBitmap, ptr %otherBitmap.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %otherBitmap.addr, metadata !32, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.declare(metadata ptr %info, metadata !33, metadata !DIExpression()), !dbg !34
  store i64 %length, ptr %length.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %length.addr, metadata !35, metadata !DIExpression()), !dbg !36
  %0 = load ptr, ptr %retval, !dbg !37
  ret ptr %0, !dbg !37
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!24, !25, !26, !27, !38}

!0 = distinct !DICompileUnit(language: DW_LANG_ObjC_plus_plus, producer: "clang version 3.4 ", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "t.mm", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "Bitmap", line: 8, size: 8, align: 8, flags: DIFlagObjcClassComplete, runtimeLang: DW_LANG_ObjC_plus_plus, file: !1, scope: !5, elements: !2)
!5 = !DIFile(filename: "t.mm", directory: "")
!7 = distinct !DISubprogram(name: "-[Bitmap initWithCopy:andInfo:andLength:]", line: 9, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 9, file: !1, scope: !5, type: !8, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!4, !10, !11, !14, !15, !19}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !4)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "SEL", line: 9, flags: DIFlagArtificial, file: !1, baseType: !12)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !13)
!13 = !DICompositeType(tag: DW_TAG_structure_type, name: "objc_selector", flags: DIFlagFwdDecl, file: !1)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !4)
!15 = !DIDerivedType(tag: DW_TAG_typedef, name: "ImageInfo", line: 7, file: !1, baseType: !16)
!16 = !DICompositeType(tag: DW_TAG_structure_type, line: 2, size: 192, align: 64, file: !1, elements: !17)
!17 = !{!18, !21, !22}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "width", line: 4, size: 64, align: 64, file: !1, scope: !16, baseType: !19)
!19 = !DIDerivedType(tag: DW_TAG_typedef, name: "NSUInteger", line: 1, file: !1, baseType: !20)
!20 = !DIBasicType(tag: DW_TAG_base_type, name: "long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "height", line: 5, size: 64, align: 64, offset: 64, file: !1, scope: !16, baseType: !19)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "pixelAspect", line: 6, size: 64, align: 64, offset: 128, file: !1, scope: !16, baseType: !23)
!23 = !DIBasicType(tag: DW_TAG_base_type, name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!24 = !{i32 1, !"Objective-C Version", i32 2}
!25 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!26 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!27 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
!28 = !DILocalVariable(name: "self", line: 9, arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !7, file: !5, type: !14)
!29 = !DILocation(line: 9, scope: !7)
!30 = !DILocalVariable(name: "_cmd", line: 9, arg: 2, flags: DIFlagArtificial, scope: !7, file: !5, type: !31)
!31 = !DIDerivedType(tag: DW_TAG_typedef, name: "SEL", line: 9, file: !1, baseType: !12)
!32 = !DILocalVariable(name: "otherBitmap", line: 9, arg: 3, scope: !7, file: !5, type: !14)
!33 = !DILocalVariable(name: "info", line: 10, arg: 4, scope: !7, file: !5, type: !15)
!34 = !DILocation(line: 10, scope: !7)
!35 = !DILocalVariable(name: "length", line: 11, arg: 5, scope: !7, file: !5, type: !19)
!36 = !DILocation(line: 11, scope: !7)
!37 = !DILocation(line: 13, scope: !7)
!38 = !{i32 1, !"Debug Info Version", i32 3}

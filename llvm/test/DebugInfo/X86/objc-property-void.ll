; RUN: llc -filetype=obj -o %t.o < %s >/dev/null 2>&1
; RUN: llvm-dwarfdump -v -debug-info %t.o | FileCheck %s

; CHECK: DW_TAG_structure_type
; CHECK:  DW_AT_APPLE_objc_complete_type
; CHECK:  DW_AT_name [DW_FORM_strp] ( .debug_str[0x{{[0-9a-fA-F]+}}] = "Foo")
; CHECK   DW_AT_APPLE_runtime_class [DW_FORM_data1]       (DW_LANG_ObjC)
; CHECK: DW_TAG_APPLE_property
; CHECK:  DW_AT_APPLE_property_name [DW_FORM_strp] ( .debug_str[0x{{[0-9a-fA-F]+}}] = "foo")

; generated from:
; @interface Foo
; @property (nonatomic,assign,readonly) void foo;
; @end
; @implementation Foo
; - (void)foo {}
; @end
;
; with:
; clang -S -emit-llvm -O0 -g

; ModuleID = '-'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%0 = type opaque
%struct._objc_cache = type opaque
%struct._class_t = type { %struct._class_t*, %struct._class_t*, %struct._objc_cache*, i8* (i8*, i8*)**, %struct._class_ro_t* }
%struct._class_ro_t = type { i32, i32, i32, i8*, i8*, %struct.__method_list_t*, %struct._objc_protocol_list*, %struct._ivar_list_t*, i8*, %struct._prop_list_t* }
%struct.__method_list_t = type { i32, i32, [0 x %struct._objc_method] }
%struct._objc_method = type { i8*, i8*, i8* }
%struct._objc_protocol_list = type { i64, [0 x %struct._protocol_t*] }
%struct._protocol_t = type { i8*, i8*, %struct._objc_protocol_list*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct._prop_list_t*, i32, i32, i8** }
%struct._ivar_list_t = type { i32, i32, [0 x %struct._ivar_t] }
%struct._ivar_t = type { i64*, i8*, i8*, i32, i32 }
%struct._prop_list_t = type { i32, i32, [0 x %struct._prop_t] }
%struct._prop_t = type { i8*, i8* }

@_objc_empty_cache = external global %struct._objc_cache
@"OBJC_CLASS_$_Foo" = global %struct._class_t { %struct._class_t* @"OBJC_METACLASS_$_Foo", %struct._class_t* null, %struct._objc_cache* @_objc_empty_cache, i8* (i8*, i8*)** null, %struct._class_ro_t* @"\01l_OBJC_CLASS_RO_$_Foo" }, section "__DATA, __objc_data", align 8
@"OBJC_METACLASS_$_Foo" = global %struct._class_t { %struct._class_t* @"OBJC_METACLASS_$_Foo", %struct._class_t* @"OBJC_CLASS_$_Foo", %struct._objc_cache* @_objc_empty_cache, i8* (i8*, i8*)** null, %struct._class_ro_t* @"\01l_OBJC_METACLASS_RO_$_Foo" }, section "__DATA, __objc_data", align 8
@"\01L_OBJC_CLASS_NAME_" = internal global [4 x i8] c"Foo\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@"\01l_OBJC_METACLASS_RO_$_Foo" = internal global %struct._class_ro_t { i32 3, i32 40, i32 40, i8* null, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), %struct.__method_list_t* null, %struct._objc_protocol_list* null, %struct._ivar_list_t* null, i8* null, %struct._prop_list_t* null }, section "__DATA, __objc_const", align 8
@"\01L_OBJC_METH_VAR_NAME_" = internal global [4 x i8] c"foo\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_METH_VAR_TYPE_" = internal global [8 x i8] c"v16@0:8\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@"\01l_OBJC_$_INSTANCE_METHODS_Foo" = internal global { i32, i32, [1 x %struct._objc_method] } { i32 24, i32 1, [1 x %struct._objc_method] [%struct._objc_method { i8* getelementptr inbounds ([4 x i8], [4 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @"\01L_OBJC_METH_VAR_TYPE_", i32 0, i32 0), i8* bitcast (void (%0*, i8*)* @"\01-[Foo foo]" to i8*) }] }, section "__DATA, __objc_const", align 8
@"\01L_OBJC_PROP_NAME_ATTR_" = internal global [4 x i8] c"foo\00", section "__TEXT,__cstring,cstring_literals", align 1
@"\01L_OBJC_PROP_NAME_ATTR_1" = internal global [7 x i8] c"Tv,R,N\00", section "__TEXT,__cstring,cstring_literals", align 1
@"\01l_OBJC_$_PROP_LIST_Foo" = internal global { i32, i32, [1 x %struct._prop_t] } { i32 16, i32 1, [1 x %struct._prop_t] [%struct._prop_t { i8* getelementptr inbounds ([4 x i8], [4 x i8]* @"\01L_OBJC_PROP_NAME_ATTR_", i32 0, i32 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @"\01L_OBJC_PROP_NAME_ATTR_1", i32 0, i32 0) }] }, section "__DATA, __objc_const", align 8
@"\01l_OBJC_CLASS_RO_$_Foo" = internal global %struct._class_ro_t { i32 2, i32 0, i32 0, i8* null, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), %struct.__method_list_t* bitcast ({ i32, i32, [1 x %struct._objc_method] }* @"\01l_OBJC_$_INSTANCE_METHODS_Foo" to %struct.__method_list_t*), %struct._objc_protocol_list* null, %struct._ivar_list_t* null, i8* null, %struct._prop_list_t* bitcast ({ i32, i32, [1 x %struct._prop_t] }* @"\01l_OBJC_$_PROP_LIST_Foo" to %struct._prop_list_t*) }, section "__DATA, __objc_const", align 8
@"\01L_OBJC_LABEL_CLASS_$" = internal global [1 x i8*] [i8* bitcast (%struct._class_t* @"OBJC_CLASS_$_Foo" to i8*)], section "__DATA, __objc_classlist, regular, no_dead_strip", align 8
@llvm.used = appending global [8 x i8*] [i8* getelementptr inbounds ([4 x i8], [4 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @"\01L_OBJC_METH_VAR_TYPE_", i32 0, i32 0), i8* bitcast ({ i32, i32, [1 x %struct._objc_method] }* @"\01l_OBJC_$_INSTANCE_METHODS_Foo" to i8*), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @"\01L_OBJC_PROP_NAME_ATTR_", i32 0, i32 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @"\01L_OBJC_PROP_NAME_ATTR_1", i32 0, i32 0), i8* bitcast ({ i32, i32, [1 x %struct._prop_t] }* @"\01l_OBJC_$_PROP_LIST_Foo" to i8*), i8* bitcast ([1 x i8*]* @"\01L_OBJC_LABEL_CLASS_$" to i8*)], section "llvm.metadata"

; Function Attrs: ssp uwtable
define internal void @"\01-[Foo foo]"(%0* %self, i8* %_cmd) #0 !dbg !10 {
entry:
  %self.addr = alloca %0*, align 8
  %_cmd.addr = alloca i8*, align 8
  store %0* %self, %0** %self.addr, align 8
  call void @llvm.dbg.declare(metadata %0** %self.addr, metadata !24, metadata !DIExpression()), !dbg !26
  store i8* %_cmd, i8** %_cmd.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %_cmd.addr, metadata !27, metadata !DIExpression()), !dbg !26
  ret void, !dbg !29
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18, !19, !20, !21, !22}
!llvm.ident = !{!23}

!0 = distinct !DICompileUnit(language: DW_LANG_ObjC, isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "-", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", line: 1, align: 8, flags: DIFlagObjcClassComplete, runtimeLang: DW_LANG_ObjC, file: !5, scope: !6, elements: !7)
!5 = !DIFile(filename: "<stdin>", directory: "")
!6 = !DIFile(filename: "<stdin>", directory: "")
!7 = !{!8}
!8 = !DIObjCProperty(name: "foo", line: 2, attributes: 2117, file: !6)
!10 = distinct !DISubprogram(name: "-[Foo foo]", line: 5, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 5, file: !5, scope: !6, type: !11, retainedNodes: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !14}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !4)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "SEL", line: 5, flags: DIFlagArtificial, file: !5, baseType: !15)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !16)
!16 = !DICompositeType(tag: DW_TAG_structure_type, name: "objc_selector", flags: DIFlagFwdDecl, file: !1)
!17 = !{i32 1, !"Objective-C Version", i32 2}
!18 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!19 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!20 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
!21 = !{i32 2, !"Dwarf Version", i32 2}
!22 = !{i32 1, !"Debug Info Version", i32 3}
!23 = !{!""}
!24 = !DILocalVariable(name: "self", arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !10, type: !25)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !4)
!26 = !DILocation(line: 0, scope: !10)
!27 = !DILocalVariable(name: "_cmd", arg: 2, flags: DIFlagArtificial, scope: !10, type: !28)
!28 = !DIDerivedType(tag: DW_TAG_typedef, name: "SEL", line: 5, file: !5, baseType: !15)
!29 = !DILocation(line: 5, scope: !10)

; RUN: llc -O0 -fast-isel -filetype=obj -o - %s | llvm-dwarfdump -v - | FileCheck %s
;
; Derived from (clang -O0 -g -fsanitize=address -fobjc-arc)
;   @protocol NSObject
;   @end
;   @interface NSObject<NSObject>{}
;   + (instancetype)alloc;
;   @end
;   struct CGSize {
;     double width;
;     double height;
;   };
;   typedef struct CGSize CGSize;
;   @interface Object : NSObject
;   - (instancetype)initWithSize:(CGSize)size;
;   - (id)aMessage;
;   @end
;   @implementation MyObject
;   + (id)doWithSize:(CGSize)imageSize andObject:(id)object {
;     return [object aMessage];
;   }
;   @end
;
; CHECK: .debug_info contents:
; CHECK: DW_TAG_subprogram
; CHECK-NEXT:   DW_AT_low_pc [DW_FORM_addr]     (0x0000000000000000)
; CHECK-NEXT:   DW_AT_high_pc [DW_FORM_addr]    ([[FN_END:.*]])
; CHECK: "_cmd"
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_location
; CHECK-NEXT:   [0x{{0*}}, 0x{{.*}}):
; CHECK-NOT:    DW_AT_
; CHECK:        [0x{{.*}}, [[FN_END]]):
; CHECK-NEXT: DW_AT_name {{.*}}"imageSize"

; ModuleID = 'm.m'
source_filename = "m.m"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios"

%0 = type opaque
%struct._class_t = type { %struct._class_t*, %struct._class_t*, %struct._objc_cache*, i8* (i8*, i8*)**, %struct._class_ro_t* }
%struct._objc_cache = type opaque
%struct._class_ro_t = type { i32, i32, i32, i8*, i8*, %struct.__method_list_t*, %struct._objc_protocol_list*, %struct._ivar_list_t*, i8*, %struct._prop_list_t* }
%struct.__method_list_t = type { i32, i32, [0 x %struct._objc_method] }
%struct._objc_method = type { i8*, i8*, i8* }
%struct._objc_protocol_list = type { i64, [0 x %struct._protocol_t*] }
%struct._protocol_t = type { i8*, i8*, %struct._objc_protocol_list*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct._prop_list_t*, i32, i32, i8**, i8*, %struct._prop_list_t* }
%struct._ivar_list_t = type { i32, i32, [0 x %struct._ivar_t] }
%struct._ivar_t = type { i32*, i8*, i8*, i32, i32 }
%struct._prop_list_t = type { i32, i32, [0 x %struct._prop_t] }
%struct._prop_t = type { i8*, i8* }
%struct.CGSize = type { double, double }

@"OBJC_CLASS_$_Object" = external global %struct._class_t
@"OBJC_CLASSLIST_REFERENCES_$_" = private global %struct._class_t* @"OBJC_CLASS_$_Object", section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@OBJC_METH_VAR_NAME_ = private unnamed_addr constant [6 x i8] c"alloc\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_ = private externally_initialized global i8* getelementptr inbounds ([6 x i8], [6 x i8]* @OBJC_METH_VAR_NAME_, i32 0, i32 0), section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip", align 8
@OBJC_METH_VAR_NAME_.1 = private unnamed_addr constant [14 x i8] c"initWithSize:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_.2 = private externally_initialized global i8* getelementptr inbounds ([14 x i8], [14 x i8]* @OBJC_METH_VAR_NAME_.1, i32 0, i32 0), section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip", align 8
@OBJC_METH_VAR_NAME_.3 = private unnamed_addr constant [9 x i8] c"aMessage\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_.4 = private externally_initialized global i8* getelementptr inbounds ([9 x i8], [9 x i8]* @OBJC_METH_VAR_NAME_.3, i32 0, i32 0), section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip", align 8
@_objc_empty_cache = external global %struct._objc_cache
@"OBJC_CLASS_$_MyObject" = global %struct._class_t { %struct._class_t* @"OBJC_METACLASS_$_MyObject", %struct._class_t* null, %struct._objc_cache* @_objc_empty_cache, i8* (i8*, i8*)** null, %struct._class_ro_t* @"\01l_OBJC_CLASS_RO_$_MyObject" }, section "__DATA, __objc_data", align 8
@"OBJC_METACLASS_$_MyObject" = global %struct._class_t { %struct._class_t* @"OBJC_METACLASS_$_MyObject", %struct._class_t* @"OBJC_CLASS_$_MyObject", %struct._objc_cache* @_objc_empty_cache, i8* (i8*, i8*)** null, %struct._class_ro_t* @"\01l_OBJC_METACLASS_RO_$_MyObject" }, section "__DATA, __objc_data", align 8
@OBJC_CLASS_NAME_ = private unnamed_addr constant [9 x i8] c"MyObject\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@OBJC_METH_VAR_NAME_.5 = private unnamed_addr constant [12 x i8] c"doWithSize:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_METH_VAR_TYPE_ = private unnamed_addr constant [21 x i8] c"@32@0:8{CGSize=dd}16\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@"\01l_OBJC_$_CLASS_METHODS_MyObject" = private global { i32, i32, [1 x %struct._objc_method] } { i32 24, i32 1, [1 x %struct._objc_method] [%struct._objc_method { i8* getelementptr inbounds ([12 x i8], [12 x i8]* @OBJC_METH_VAR_NAME_.5, i32 0, i32 0), i8* getelementptr inbounds ([21 x i8], [21 x i8]* @OBJC_METH_VAR_TYPE_, i32 0, i32 0), i8* bitcast (i8* (i8*, i8*, [2 x double])* @"\01+[MyObject doWithSize:]" to i8*) }] }, section "__DATA, __objc_const", align 8
@"\01l_OBJC_METACLASS_RO_$_MyObject" = private global %struct._class_ro_t { i32 131, i32 40, i32 40, i8* null, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @OBJC_CLASS_NAME_, i32 0, i32 0), %struct.__method_list_t* bitcast ({ i32, i32, [1 x %struct._objc_method] }* @"\01l_OBJC_$_CLASS_METHODS_MyObject" to %struct.__method_list_t*), %struct._objc_protocol_list* null, %struct._ivar_list_t* null, i8* null, %struct._prop_list_t* null }, section "__DATA, __objc_const", align 8
@"\01l_OBJC_CLASS_RO_$_MyObject" = private global %struct._class_ro_t { i32 130, i32 0, i32 0, i8* null, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @OBJC_CLASS_NAME_, i32 0, i32 0), %struct.__method_list_t* null, %struct._objc_protocol_list* null, %struct._ivar_list_t* null, i8* null, %struct._prop_list_t* null }, section "__DATA, __objc_const", align 8
@"OBJC_LABEL_CLASS_$" = private global [1 x i8*] [i8* bitcast (%struct._class_t* @"OBJC_CLASS_$_MyObject" to i8*)], section "__DATA, __objc_classlist, regular, no_dead_strip", align 8
@llvm.compiler.used = appending global [12 x i8*] [i8* bitcast (%struct._class_t** @"OBJC_CLASSLIST_REFERENCES_$_" to i8*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @OBJC_METH_VAR_NAME_, i32 0, i32 0), i8* bitcast (i8** @OBJC_SELECTOR_REFERENCES_ to i8*), i8* getelementptr inbounds ([14 x i8], [14 x i8]* @OBJC_METH_VAR_NAME_.1, i32 0, i32 0), i8* bitcast (i8** @OBJC_SELECTOR_REFERENCES_.2 to i8*), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @OBJC_METH_VAR_NAME_.3, i32 0, i32 0), i8* bitcast (i8** @OBJC_SELECTOR_REFERENCES_.4 to i8*), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @OBJC_CLASS_NAME_, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8], [12 x i8]* @OBJC_METH_VAR_NAME_.5, i32 0, i32 0), i8* getelementptr inbounds ([21 x i8], [21 x i8]* @OBJC_METH_VAR_TYPE_, i32 0, i32 0), i8* bitcast ({ i32, i32, [1 x %struct._objc_method] }* @"\01l_OBJC_$_CLASS_METHODS_MyObject" to i8*), i8* bitcast ([1 x i8*]* @"OBJC_LABEL_CLASS_$" to i8*)], section "llvm.metadata"
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 1, void ()* @asan.module_ctor, i8* null }]
@__asan_shadow_memory_dynamic_address = external global i64
@__asan_gen_ = private unnamed_addr constant [34 x i8] c"2 32 16 9 imageSize 64 8 6 object\00", align 1

; Function Attrs: noinline sanitize_address ssp uwtable
define internal i8* @"\01+[MyObject doWithSize:]"(i8* %self, i8* %_cmd, [2 x double] %imageSize.coerce) #0 !dbg !14 {
entry:
  %0 = load i64, i64* @__asan_shadow_memory_dynamic_address
  %self.addr = alloca i8*, align 8
  %_cmd.addr = alloca i8*, align 8
  %MyAlloca = alloca [96 x i8], align 32, !dbg !35
  %1 = ptrtoint [96 x i8]* %MyAlloca to i64, !dbg !35
  %2 = add i64 %1, 32, !dbg !35
  %3 = inttoptr i64 %2 to %struct.CGSize*, !dbg !35
  %4 = add i64 %1, 64, !dbg !35
  %5 = inttoptr i64 %4 to %0**, !dbg !35
  %6 = inttoptr i64 %1 to i64*, !dbg !35
  store i64 1102416563, i64* %6, !dbg !35
  %7 = add i64 %1, 8, !dbg !35
  %8 = inttoptr i64 %7 to i64*, !dbg !35
  store i64 ptrtoint ([34 x i8]* @__asan_gen_ to i64), i64* %8, !dbg !35
  %9 = add i64 %1, 16, !dbg !35
  %10 = inttoptr i64 %9 to i64*, !dbg !35
  store i64 ptrtoint (i8* (i8*, i8*, [2 x double])* @"\01+[MyObject doWithSize:]" to i64), i64* %10, !dbg !35
  %11 = lshr i64 %1, 3, !dbg !35
  %12 = add i64 %11, %0, !dbg !35
  %13 = add i64 %12, 0, !dbg !35
  %14 = inttoptr i64 %13 to i64*, !dbg !35
  store i64 -940689368107847183, i64* %14, align 1, !dbg !35
  %15 = add i64 %12, 9, !dbg !35
  %16 = inttoptr i64 %15 to i16*, !dbg !35
  store i16 -3085, i16* %16, align 1, !dbg !35
  %17 = add i64 %12, 11, !dbg !35
  %18 = inttoptr i64 %17 to i8*, !dbg !35
  store i8 -13, i8* %18, align 1, !dbg !35
  call void @llvm.dbg.declare(metadata %struct.CGSize* %3, metadata !36, metadata !37), !dbg !38
  call void @llvm.dbg.declare(metadata %0** %5, metadata !39, metadata !37), !dbg !45
  %19 = bitcast %struct.CGSize* %3 to [2 x double]*
  %20 = ptrtoint [2 x double]* %19 to i64
  %21 = lshr i64 %20, 3
  %22 = add i64 %21, %0
  %23 = inttoptr i64 %22 to i16*
  %24 = load i16, i16* %23
  %25 = icmp ne i16 %24, 0
  br i1 %25, label %26, label %27

; <label>:26:                                     ; preds = %entry
  call void @__asan_report_store16(i64 %20)
  call void asm sideeffect "", ""()
  unreachable

; <label>:27:                                     ; preds = %entry
  store [2 x double] %imageSize.coerce, [2 x double]* %19, align 8
  store i8* %self, i8** %self.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %self.addr, metadata !46, metadata !48), !dbg !49
  store i8* %_cmd, i8** %_cmd.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %_cmd.addr, metadata !50, metadata !48), !dbg !49
  %28 = load %struct._class_t*, %struct._class_t** @"OBJC_CLASSLIST_REFERENCES_$_", align 8, !dbg !52
  %29 = add i64 lshr (i64 ptrtoint (i8** @OBJC_SELECTOR_REFERENCES_ to i64), i64 3), %0, !dbg !52
  %30 = inttoptr i64 %29 to i8*, !dbg !52
  %31 = load i8, i8* %30, !dbg !52
  %32 = icmp ne i8 %31, 0, !dbg !52
  br i1 %32, label %33, label %34, !dbg !52

; <label>:33:                                     ; preds = %27
  call void @__asan_report_load8(i64 ptrtoint (i8** @OBJC_SELECTOR_REFERENCES_ to i64)), !dbg !52
  call void asm sideeffect "", ""(), !dbg !52
  unreachable, !dbg !52

; <label>:34:                                     ; preds = %27
  %35 = load i8*, i8** @OBJC_SELECTOR_REFERENCES_, align 8, !dbg !52, !invariant.load !2
  %36 = bitcast %struct._class_t* %28 to i8*, !dbg !52
  %call = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* %36, i8* %35), !dbg !52
  %37 = bitcast i8* %call to %0*, !dbg !52
  %38 = add i64 lshr (i64 ptrtoint (i8** @OBJC_SELECTOR_REFERENCES_.2 to i64), i64 3), %0, !dbg !53
  %39 = inttoptr i64 %38 to i8*, !dbg !53
  %40 = load i8, i8* %39, !dbg !53
  %41 = icmp ne i8 %40, 0, !dbg !53
  br i1 %41, label %42, label %43, !dbg !53

; <label>:42:                                     ; preds = %34
  call void @__asan_report_load8(i64 ptrtoint (i8** @OBJC_SELECTOR_REFERENCES_.2 to i64)), !dbg !53
  call void asm sideeffect "", ""(), !dbg !53
  unreachable, !dbg !53

; <label>:43:                                     ; preds = %34
  %44 = load i8*, i8** @OBJC_SELECTOR_REFERENCES_.2, align 8, !dbg !53, !invariant.load !2
  %45 = bitcast %0* %37 to i8*, !dbg !53
  %46 = bitcast %struct.CGSize* %3 to [2 x double]*, !dbg !53
  %47 = ptrtoint [2 x double]* %46 to i64, !dbg !53
  %48 = lshr i64 %47, 3, !dbg !53
  %49 = add i64 %48, %0, !dbg !53
  %50 = inttoptr i64 %49 to i16*, !dbg !53
  %51 = load i16, i16* %50, !dbg !53
  %52 = icmp ne i16 %51, 0, !dbg !53
  br i1 %52, label %53, label %54, !dbg !53

; <label>:53:                                     ; preds = %43
  call void @__asan_report_load16(i64 %47), !dbg !53
  call void asm sideeffect "", ""(), !dbg !53
  unreachable, !dbg !53

; <label>:54:                                     ; preds = %43
  %55 = load [2 x double], [2 x double]* %46, align 8, !dbg !53
  %call1 = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, [2 x double])*)(i8* %45, i8* %44, [2 x double] %55), !dbg !53
  %56 = bitcast i8* %call1 to %0*, !dbg !53
  %57 = ptrtoint %0** %5 to i64, !dbg !45
  %58 = lshr i64 %57, 3, !dbg !45
  %59 = add i64 %58, %0, !dbg !45
  %60 = inttoptr i64 %59 to i8*, !dbg !45
  %61 = load i8, i8* %60, !dbg !45
  %62 = icmp ne i8 %61, 0, !dbg !45
  br i1 %62, label %63, label %64, !dbg !45

; <label>:63:                                     ; preds = %54
  call void @__asan_report_store8(i64 %57), !dbg !45
  call void asm sideeffect "", ""(), !dbg !45
  unreachable, !dbg !45

; <label>:64:                                     ; preds = %54
  store %0* %56, %0** %5, align 8, !dbg !45
  %65 = load %0*, %0** %5, align 8, !dbg !54
  %66 = add i64 lshr (i64 ptrtoint (i8** @OBJC_SELECTOR_REFERENCES_.4 to i64), i64 3), %0, !dbg !55
  %67 = inttoptr i64 %66 to i8*, !dbg !55
  %68 = load i8, i8* %67, !dbg !55
  %69 = icmp ne i8 %68, 0, !dbg !55
  br i1 %69, label %70, label %71, !dbg !55

; <label>:70:                                     ; preds = %64
  call void @__asan_report_load8(i64 ptrtoint (i8** @OBJC_SELECTOR_REFERENCES_.4 to i64)), !dbg !55
  call void asm sideeffect "", ""(), !dbg !55
  unreachable, !dbg !55

; <label>:71:                                     ; preds = %64
  %72 = load i8*, i8** @OBJC_SELECTOR_REFERENCES_.4, align 8, !dbg !55, !invariant.load !2
  %73 = bitcast %0* %65 to i8*, !dbg !55
  %call2 = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* %73, i8* %72), !dbg !55
  call void asm sideeffect "mov\09fp, fp\09\09; marker for objc_retainAutoreleaseReturnValue", ""(), !dbg !55
  %74 = call i8* @objc_retainAutoreleasedReturnValue(i8* %call2) #3, !dbg !55
  %75 = bitcast %0** %5 to i8**, !dbg !56
  call void @objc_storeStrong(i8** %75, i8* null) #3, !dbg !56
  %76 = tail call i8* @objc_autoreleaseReturnValue(i8* %74) #3, !dbg !56
  store i64 1172321806, i64* %6, !dbg !56
  %77 = add i64 %12, 0, !dbg !56
  %78 = inttoptr i64 %77 to i64*, !dbg !56
  store i64 0, i64* %78, align 1, !dbg !56
  %79 = add i64 %12, 9, !dbg !56
  %80 = inttoptr i64 %79 to i16*, !dbg !56
  store i16 0, i16* %80, align 1, !dbg !56
  %81 = add i64 %12, 11, !dbg !56
  %82 = inttoptr i64 %81 to i8*, !dbg !56
  store i8 0, i8* %82, align 1, !dbg !56
  ret i8* %76, !dbg !56
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nonlazybind
declare i8* @objc_msgSend(i8*, i8*, ...) #2

declare i8* @objc_retainAutoreleasedReturnValue(i8* returned)

declare void @objc_storeStrong(i8**, i8*)

declare i8* @objc_autoreleaseReturnValue(i8* returned)

define internal void @asan.module_ctor() {
  call void @__asan_init()
  call void @__asan_version_mismatch_check_v8()
  ret void
}

declare void @__asan_init()

declare void @__asan_version_mismatch_check_v8()

declare void @__asan_report_load8(i64)

declare void @__asan_report_load16(i64)

declare void @__asan_report_store8(i64)

declare void @__asan_report_store16(i64)

attributes #0 = { noinline sanitize_address ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nonlazybind }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7, !8, !9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !1, producer: "clang version 5.0.0 (trunk 295779) (llvm/trunk 295777)", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "m.m", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyObject", scope: !1, file: !1, line: 15, flags: DIFlagObjcClassComplete, elements: !2, runtimeLang: DW_LANG_ObjC)
!5 = !{i32 1, !"Objective-C Version", i32 2}
!6 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!7 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!8 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
!9 = !{i32 1, !"Objective-C Class Properties", i32 64}
!10 = !{i32 2, !"Dwarf Version", i32 2}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"PIC Level", i32 2}
!13 = !{!"clang version 5.0.0 (trunk 295779) (llvm/trunk 295777)"}
!14 = distinct !DISubprogram(name: "+[MyObject doWithSize:]", scope: !1, file: !1, line: 16, type: !15, isLocal: true, isDefinition: true, scopeLine: 16, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{!17, !24, !26, !29}
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "id", file: !1, baseType: !18)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = !DICompositeType(tag: DW_TAG_structure_type, name: "objc_object", file: !1, elements: !20)
!20 = !{!21}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "isa", scope: !19, file: !1, baseType: !22, size: 64)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 64)
!23 = !DICompositeType(tag: DW_TAG_structure_type, name: "objc_class", file: !1, flags: DIFlagFwdDecl)
!24 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !25, flags: DIFlagArtificial | DIFlagObjectPointer)
!25 = !DIDerivedType(tag: DW_TAG_typedef, name: "Class", file: !1, baseType: !22)
!26 = !DIDerivedType(tag: DW_TAG_typedef, name: "SEL", file: !1, baseType: !27, flags: DIFlagArtificial)
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !28, size: 64)
!28 = !DICompositeType(tag: DW_TAG_structure_type, name: "objc_selector", file: !1, flags: DIFlagFwdDecl)
!29 = !DIDerivedType(tag: DW_TAG_typedef, name: "CGSize", file: !1, line: 10, baseType: !30)
!30 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CGSize", file: !1, line: 6, size: 128, elements: !31)
!31 = !{!32, !34}
!32 = !DIDerivedType(tag: DW_TAG_member, name: "width", scope: !30, file: !1, line: 7, baseType: !33, size: 64)
!33 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!34 = !DIDerivedType(tag: DW_TAG_member, name: "height", scope: !30, file: !1, line: 8, baseType: !33, size: 64, offset: 64)
!35 = !DILocation(line: 16, scope: !14)
!36 = !DILocalVariable(name: "imageSize", arg: 3, scope: !14, file: !1, line: 16, type: !29)
!37 = !DIExpression(DW_OP_deref)
!38 = !DILocation(line: 16, column: 26, scope: !14)
!39 = !DILocalVariable(name: "object", scope: !14, file: !1, line: 17, type: !40)
!40 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !41, size: 64)
!41 = !DICompositeType(tag: DW_TAG_structure_type, name: "Object", scope: !1, file: !1, line: 11, elements: !42, runtimeLang: DW_LANG_ObjC)
!42 = !{!43}
!43 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !41, baseType: !44)
!44 = !DICompositeType(tag: DW_TAG_structure_type, name: "NSObject", scope: !1, file: !1, line: 3, elements: !2, runtimeLang: DW_LANG_ObjC)
!45 = !DILocation(line: 17, column: 11, scope: !14)
!46 = !DILocalVariable(name: "self", arg: 1, scope: !14, type: !47, flags: DIFlagArtificial | DIFlagObjectPointer)
!47 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !25)
!48 = !DIExpression()
!49 = !DILocation(line: 0, scope: !14)
!50 = !DILocalVariable(name: "_cmd", arg: 2, scope: !14, type: !51, flags: DIFlagArtificial)
!51 = !DIDerivedType(tag: DW_TAG_typedef, name: "SEL", file: !1, baseType: !27)
!52 = !DILocation(line: 17, column: 21, scope: !14)
!53 = !DILocation(line: 17, column: 20, scope: !14)
!54 = !DILocation(line: 18, column: 11, scope: !14)
!55 = !DILocation(line: 18, column: 10, scope: !14)
!56 = !DILocation(line: 19, column: 1, scope: !14)

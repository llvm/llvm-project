; RUN: llc -mtriple x86_64-apple-darwin -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -v -debug-info %t.o | FileCheck %s

; RUN: llc --try-experimental-debuginfo-iterators -mtriple x86_64-apple-darwin -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -v -debug-info %t.o | FileCheck %s

; Generated from llvm/tools/clang/test/CodeGenObjC/debug-info-blocks.m
; rdar://problem/9279956
; test that the DW_AT_location of self is at ( fbreg +{{[0-9]+}}, deref, +{{[0-9]+}} )

; CHECK: [[A:.*]]:   DW_TAG_structure_type
; CHECK-NEXT: DW_AT_APPLE_objc_complete_type
; CHECK-NEXT: DW_AT_name{{.*}}"A"

; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_object_pointer
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}_block_invoke

; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_location
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}.block_descriptor

; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_location{{.*}}(DW_OP_fbreg -24, DW_OP_deref, DW_OP_plus_uconst 0x20)
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"self"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_type{{.*}}{[[APTR:.*]]}
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_artificial

; CHECK: [[APTR]]:   DW_TAG_pointer_type
; CHECK-NEXT: {[[A]]}


; ModuleID = 'llvm/tools/clang/test/CodeGenObjC/debug-info-blocks.m'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

%0 = type opaque
%1 = type opaque
%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }
%struct._objc_cache = type opaque
%struct._class_ro_t = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct.__method_list_t = type { i32, i32, [0 x %struct._objc_method] }
%struct._objc_method = type { ptr, ptr, ptr }
%struct._objc_protocol_list = type { i64, [0 x ptr] }
%struct._protocol_t = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, ptr }
%struct._prop_list_t = type { i32, i32, [0 x %struct._prop_t] }
%struct._prop_t = type { ptr, ptr }
%struct._ivar_list_t = type { i32, i32, [0 x %struct._ivar_t] }
%struct._ivar_t = type { ptr, ptr, ptr, i32, i32 }
%struct._message_ref_t = type { ptr, ptr }
%struct._objc_super = type { ptr, ptr }
%struct.__block_descriptor = type { i64, i64 }
%struct.__block_literal_generic = type { ptr, i32, i32, ptr, ptr }

@"OBJC_CLASS_$_A" = global %struct._class_t { ptr @"OBJC_METACLASS_$_A", ptr @"OBJC_CLASS_$_NSObject", ptr @_objc_empty_cache, ptr @_objc_empty_vtable, ptr @"\01l_OBJC_CLASS_RO_$_A" }, section "__DATA, __objc_data", align 8
@"\01L_OBJC_CLASSLIST_SUP_REFS_$_" = internal global ptr @"OBJC_CLASS_$_A", section "__DATA, __objc_superrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_METH_VAR_NAME_" = internal global [5 x i8] c"init\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_SELECTOR_REFERENCES_" = internal externally_initialized global ptr @"\01L_OBJC_METH_VAR_NAME_", section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"OBJC_CLASS_$_NSMutableDictionary" = external global %struct._class_t
@"\01L_OBJC_CLASSLIST_REFERENCES_$_" = internal global ptr @"OBJC_CLASS_$_NSMutableDictionary", section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_METH_VAR_NAME_1" = internal global [6 x i8] c"alloc\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01l_objc_msgSend_fixup_alloc" = weak hidden global { ptr, ptr } { ptr @objc_msgSend_fixup, ptr @"\01L_OBJC_METH_VAR_NAME_1" }, section "__DATA, __objc_msgrefs, coalesced", align 16
@"\01L_OBJC_METH_VAR_NAME_2" = internal global [6 x i8] c"count\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01l_objc_msgSend_fixup_count" = weak hidden global { ptr, ptr } { ptr @objc_msgSend_fixup, ptr @"\01L_OBJC_METH_VAR_NAME_2" }, section "__DATA, __objc_msgrefs, coalesced", align 16
@"OBJC_IVAR_$_A.ivar" = global i64 0, section "__DATA, __objc_ivar", align 8
@_NSConcreteStackBlock = external global ptr
@.str = private unnamed_addr constant [6 x i8] c"v8@?0\00", align 1
@__block_descriptor_tmp = internal constant { i64, i64, ptr, ptr, ptr, i64 } { i64 0, i64 40, ptr @__copy_helper_block_, ptr @__destroy_helper_block_, ptr @.str, i64 256 }
@_objc_empty_cache = external global %struct._objc_cache
@_objc_empty_vtable = external global ptr
@"OBJC_METACLASS_$_NSObject" = external global %struct._class_t
@"\01L_OBJC_CLASS_NAME_" = internal global [2 x i8] c"A\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@"\01l_OBJC_METACLASS_RO_$_A" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @"\01L_OBJC_CLASS_NAME_", ptr null, ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
@"OBJC_METACLASS_$_A" = global %struct._class_t { ptr @"OBJC_METACLASS_$_NSObject", ptr @"OBJC_METACLASS_$_NSObject", ptr @_objc_empty_cache, ptr @_objc_empty_vtable, ptr @"\01l_OBJC_METACLASS_RO_$_A" }, section "__DATA, __objc_data", align 8
@"OBJC_CLASS_$_NSObject" = external global %struct._class_t
@"\01L_OBJC_METH_VAR_TYPE_" = internal global [8 x i8] c"@16@0:8\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@"\01l_OBJC_$_INSTANCE_METHODS_A" = internal global { i32, i32, [1 x %struct._objc_method] } { i32 24, i32 1, [1 x %struct._objc_method] [%struct._objc_method { ptr @"\01L_OBJC_METH_VAR_NAME_", ptr @"\01L_OBJC_METH_VAR_TYPE_", ptr @"\01-[A init]" }] }, section "__DATA, __objc_const", align 8
@"\01L_OBJC_METH_VAR_NAME_3" = internal global [5 x i8] c"ivar\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_METH_VAR_TYPE_4" = internal global [2 x i8] c"i\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@"\01l_OBJC_$_INSTANCE_VARIABLES_A" = internal global { i32, i32, [1 x %struct._ivar_t] } { i32 32, i32 1, [1 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_A.ivar", ptr @"\01L_OBJC_METH_VAR_NAME_3", ptr @"\01L_OBJC_METH_VAR_TYPE_4", i32 2, i32 4 }] }, section "__DATA, __objc_const", align 8
@"\01l_OBJC_CLASS_RO_$_A" = internal global %struct._class_ro_t { i32 0, i32 0, i32 4, ptr null, ptr @"\01L_OBJC_CLASS_NAME_", ptr @"\01l_OBJC_$_INSTANCE_METHODS_A", ptr null, ptr @"\01l_OBJC_$_INSTANCE_VARIABLES_A", ptr null, ptr null }, section "__DATA, __objc_const", align 8
@"\01L_OBJC_CLASSLIST_REFERENCES_$_5" = internal global ptr @"OBJC_CLASS_$_A", section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_LABEL_CLASS_$" = internal global [1 x ptr] [ptr @"OBJC_CLASS_$_A"], section "__DATA, __objc_classlist, regular, no_dead_strip", align 8
@llvm.used = appending global [14 x ptr] [ptr @"\01L_OBJC_CLASSLIST_SUP_REFS_$_", ptr @"\01L_OBJC_METH_VAR_NAME_", ptr @"\01L_OBJC_SELECTOR_REFERENCES_", ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_", ptr @"\01L_OBJC_METH_VAR_NAME_1", ptr @"\01L_OBJC_METH_VAR_NAME_2", ptr @"\01L_OBJC_CLASS_NAME_", ptr @"\01L_OBJC_METH_VAR_TYPE_", ptr @"\01l_OBJC_$_INSTANCE_METHODS_A", ptr @"\01L_OBJC_METH_VAR_NAME_3", ptr @"\01L_OBJC_METH_VAR_TYPE_4", ptr @"\01l_OBJC_$_INSTANCE_VARIABLES_A", ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_5", ptr @"\01L_OBJC_LABEL_CLASS_$"], section "llvm.metadata"

define internal ptr @"\01-[A init]"(ptr %self, ptr %_cmd) #0 !dbg !13 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca %struct._objc_super
  %4 = alloca <{ ptr, i32, i32, ptr, ptr, ptr }>, align 8
  store ptr %self, ptr %1, align 8
  call void @llvm.dbg.declare(metadata ptr %1, metadata !60, metadata !DIExpression()), !dbg !62
  store ptr %_cmd, ptr %2, align 8
  call void @llvm.dbg.declare(metadata ptr %2, metadata !63, metadata !DIExpression()), !dbg !62
  %5 = load ptr, ptr %1, !dbg !65
  %6 = bitcast ptr %5 to ptr, !dbg !65
  %7 = getelementptr inbounds %struct._objc_super, ptr %3, i32 0, i32 0, !dbg !65
  store ptr %6, ptr %7, !dbg !65
  %8 = load ptr, ptr @"\01L_OBJC_CLASSLIST_SUP_REFS_$_", !dbg !65
  %9 = bitcast ptr %8 to ptr, !dbg !65
  %10 = getelementptr inbounds %struct._objc_super, ptr %3, i32 0, i32 1, !dbg !65
  store ptr %9, ptr %10, !dbg !65
  %11 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", !dbg !65, !invariant.load !67
  %12 = call ptr @objc_msgSendSuper2(ptr %3, ptr %11), !dbg !65
  %13 = bitcast ptr %12 to ptr, !dbg !65
  store ptr %13, ptr %1, align 8, !dbg !65
  %14 = icmp ne ptr %13, null, !dbg !65
  br i1 %14, label %15, label %24, !dbg !65

; <label>:15                                      ; preds = %0
  %16 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %4, i32 0, i32 0, !dbg !68
  store ptr @_NSConcreteStackBlock, ptr %16, !dbg !68
  %17 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %4, i32 0, i32 1, !dbg !68
  store i32 -1040187392, ptr %17, !dbg !68
  %18 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %4, i32 0, i32 2, !dbg !68
  store i32 0, ptr %18, !dbg !68
  %19 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %4, i32 0, i32 3, !dbg !68
  store ptr @"__9-[A init]_block_invoke", ptr %19, !dbg !68
  %20 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %4, i32 0, i32 4, !dbg !68
  store ptr @__block_descriptor_tmp, ptr %20, !dbg !68
  %21 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %4, i32 0, i32 5, !dbg !68
  %22 = load ptr, ptr %1, align 8, !dbg !68
  store ptr %22, ptr %21, align 8, !dbg !68
  %23 = bitcast ptr %4 to ptr, !dbg !68
  call void @run(ptr %23), !dbg !68
  br label %24, !dbg !70

; <label>:24                                      ; preds = %15, %0
  %25 = load ptr, ptr %1, align 8, !dbg !71
  %26 = bitcast ptr %25 to ptr, !dbg !71
  ret ptr %26, !dbg !71
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare ptr @objc_msgSendSuper2(ptr, ptr, ...)

define internal void @run(ptr %block) #0 !dbg !39 {
  %1 = alloca ptr, align 8
  store ptr %block, ptr %1, align 8
  call void @llvm.dbg.declare(metadata ptr %1, metadata !72, metadata !DIExpression()), !dbg !73
  %2 = load ptr, ptr %1, align 8, !dbg !74
  %3 = bitcast ptr %2 to ptr, !dbg !74
  %4 = getelementptr inbounds %struct.__block_literal_generic, ptr %3, i32 0, i32 3, !dbg !74
  %5 = bitcast ptr %3 to ptr, !dbg !74
  %6 = load ptr, ptr %4, !dbg !74
  %7 = bitcast ptr %6 to ptr, !dbg !74
  call void %7(ptr %5), !dbg !74
  ret void, !dbg !75
}

define internal void @"__9-[A init]_block_invoke"(ptr %.block_descriptor) #0 !dbg !27 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %d = alloca ptr, align 8
  store ptr %.block_descriptor, ptr %1, align 8
  %3 = load ptr, ptr %1
  call void @llvm.dbg.value(metadata ptr %3, metadata !76, metadata !DIExpression()), !dbg !88
  call void @llvm.dbg.declare(metadata ptr %.block_descriptor, metadata !76, metadata !DIExpression()), !dbg !88
  %4 = bitcast ptr %.block_descriptor to ptr, !dbg !88
  store ptr %4, ptr %2, align 8, !dbg !88
  %5 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %4, i32 0, i32 5, !dbg !88
  call void @llvm.dbg.declare(metadata ptr %2, metadata !89, metadata !111), !dbg !90
  call void @llvm.dbg.declare(metadata ptr %d, metadata !91, metadata !DIExpression()), !dbg !100
  %6 = load ptr, ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_", !dbg !100
  %7 = bitcast ptr %6 to ptr, !dbg !100
  %8 = load ptr, ptr @"\01l_objc_msgSend_fixup_alloc", !dbg !100
  %9 = bitcast ptr %8 to ptr, !dbg !100
  %10 = call ptr %9(ptr %7, ptr @"\01l_objc_msgSend_fixup_alloc"), !dbg !100
  %11 = bitcast ptr %10 to ptr, !dbg !100
  %12 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", !dbg !100, !invariant.load !67
  %13 = bitcast ptr %11 to ptr, !dbg !100
  %14 = call ptr @objc_msgSend(ptr %13, ptr %12), !dbg !100
  %15 = bitcast ptr %14 to ptr, !dbg !100
  store ptr %15, ptr %d, align 8, !dbg !100
  %16 = load ptr, ptr %d, align 8, !dbg !101
  %17 = bitcast ptr %16 to ptr, !dbg !101
  %18 = load ptr, ptr @"\01l_objc_msgSend_fixup_count", !dbg !101
  %19 = bitcast ptr %18 to ptr, !dbg !101
  %20 = call i32 %19(ptr %17, ptr @"\01l_objc_msgSend_fixup_count"), !dbg !101
  %21 = add nsw i32 42, %20, !dbg !101
  %22 = load ptr, ptr %5, align 8, !dbg !101
  %23 = load i64, ptr @"OBJC_IVAR_$_A.ivar", !dbg !101, !invariant.load !67
  %24 = bitcast ptr %22 to ptr, !dbg !101
  %25 = getelementptr inbounds i8, ptr %24, i64 %23, !dbg !101
  %26 = bitcast ptr %25 to ptr, !dbg !101
  store i32 %21, ptr %26, align 4, !dbg !101
  ret void, !dbg !90
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #1

declare ptr @objc_msgSend_fixup(ptr, ptr, ...)

declare ptr @objc_msgSend(ptr, ptr, ...) #2

define internal void @__copy_helper_block_(ptr, ptr) !dbg !31 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  call void @llvm.dbg.declare(metadata ptr %3, metadata !102, metadata !DIExpression()), !dbg !103
  store ptr %1, ptr %4, align 8
  call void @llvm.dbg.declare(metadata ptr %4, metadata !104, metadata !DIExpression()), !dbg !103
  %5 = load ptr, ptr %4, !dbg !103
  %6 = bitcast ptr %5 to ptr, !dbg !103
  %7 = load ptr, ptr %3, !dbg !103
  %8 = bitcast ptr %7 to ptr, !dbg !103
  %9 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %6, i32 0, i32 5, !dbg !103
  %10 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %8, i32 0, i32 5, !dbg !103
  %11 = load ptr, ptr %9, !dbg !103
  %12 = bitcast ptr %11 to ptr, !dbg !103
  %13 = bitcast ptr %10 to ptr, !dbg !103
  call void @_Block_object_assign(ptr %13, ptr %12, i32 3) #3, !dbg !103
  ret void, !dbg !103
}

declare void @_Block_object_assign(ptr, ptr, i32)

define internal void @__destroy_helper_block_(ptr) !dbg !35 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  call void @llvm.dbg.declare(metadata ptr %2, metadata !105, metadata !DIExpression()), !dbg !106
  %3 = load ptr, ptr %2, !dbg !106
  %4 = bitcast ptr %3 to ptr, !dbg !106
  %5 = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %4, i32 0, i32 5, !dbg !106
  %6 = load ptr, ptr %5, !dbg !106
  %7 = bitcast ptr %6 to ptr, !dbg !106
  call void @_Block_object_dispose(ptr %7, i32 3) #3, !dbg !106
  ret void, !dbg !106
}

declare void @_Block_object_dispose(ptr, i32)

define i32 @main() #0 !dbg !36 {
  %1 = alloca i32, align 4
  %a = alloca ptr, align 8
  store i32 0, ptr %1
  call void @llvm.dbg.declare(metadata ptr %a, metadata !107, metadata !DIExpression()), !dbg !108
  %2 = load ptr, ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_5", !dbg !108
  %3 = bitcast ptr %2 to ptr, !dbg !108
  %4 = load ptr, ptr @"\01l_objc_msgSend_fixup_alloc", !dbg !108
  %5 = bitcast ptr %4 to ptr, !dbg !108
  %6 = call ptr %5(ptr %3, ptr @"\01l_objc_msgSend_fixup_alloc"), !dbg !108
  %7 = bitcast ptr %6 to ptr, !dbg !108
  %8 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_", !dbg !108, !invariant.load !67
  %9 = bitcast ptr %7 to ptr, !dbg !108
  %10 = call ptr @objc_msgSend(ptr %9, ptr %8), !dbg !108
  %11 = bitcast ptr %10 to ptr, !dbg !108
  store ptr %11, ptr %a, align 8, !dbg !108
  ret i32 0, !dbg !109
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nonlazybind }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!56, !57, !58, !59, !110}

!0 = distinct !DICompileUnit(language: DW_LANG_ObjC, producer: "clang version 3.3 ", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports:  !2)
!1 = !DIFile(filename: "llvm/tools/clang/test/CodeGenObjC/<unknown>", directory: "llvm/_build.ninja.Debug")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "A", line: 33, size: 32, align: 32, flags: DIFlagObjcClassComplete, runtimeLang: DW_LANG_ObjC, file: !5, scope: !6, elements: !7)
!5 = !DIFile(filename: "llvm/tools/clang/test/CodeGenObjC/debug-info-blocks.m", directory: "llvm/_build.ninja.Debug")
!6 = !DIFile(filename: "llvm/tools/clang/test/CodeGenObjC/debug-info-blocks.m", directory: "llvm/_build.ninja.Debug")
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !4, baseType: !9)
!9 = !DICompositeType(tag: DW_TAG_structure_type, name: "NSObject", line: 21, align: 8, runtimeLang: DW_LANG_ObjC, file: !5, scope: !6, elements: !2)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "ivar", line: 35, size: 32, align: 32, file: !5, scope: !6, baseType: !11)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = distinct !DISubprogram(name: "-[A init]", line: 46, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 46, file: !5, scope: !6, type: !14, retainedNodes: !2)
!14 = !DISubroutineType(types: !15)
!15 = !{!16, !23, !24}
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "id", line: 46, file: !5, baseType: !17)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !18)
!18 = !DICompositeType(tag: DW_TAG_structure_type, name: "objc_object", file: !1, elements: !19)
!19 = !{!20}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "isa", size: 64, file: !1, scope: !18, baseType: !21)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, baseType: !22)
!22 = !DICompositeType(tag: DW_TAG_structure_type, name: "objc_class", flags: DIFlagFwdDecl, file: !1)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !4)
!24 = !DIDerivedType(tag: DW_TAG_typedef, name: "SEL", line: 46, flags: DIFlagArtificial, file: !5, baseType: !25)
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !26)
!26 = !DICompositeType(tag: DW_TAG_structure_type, name: "objc_selector", flags: DIFlagFwdDecl, file: !1)
!27 = distinct !DISubprogram(name: "__9-[A init]_block_invoke", line: 49, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 49, file: !5, scope: !6, type: !28, retainedNodes: !2)
!28 = !DISubroutineType(types: !29)
!29 = !{null, !30}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: null)
!31 = distinct !DISubprogram(name: "__copy_helper_block_", line: 52, isLocal: true, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0, scopeLine: 52, file: !1, scope: !32, type: !33, retainedNodes: !2)
!32 = !DIFile(filename: "llvm/tools/clang/test/CodeGenObjC/<unknown>", directory: "llvm/_build.ninja.Debug")
!33 = !DISubroutineType(types: !34)
!34 = !{null, !30, !30}
!35 = distinct !DISubprogram(name: "__destroy_helper_block_", line: 52, isLocal: true, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0, scopeLine: 52, file: !1, scope: !32, type: !28, retainedNodes: !2)
!36 = distinct !DISubprogram(name: "main", line: 59, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0, scopeLine: 60, file: !5, scope: !6, type: !37, retainedNodes: !2)
!37 = !DISubroutineType(types: !38)
!38 = !{!11}
!39 = distinct !DISubprogram(name: "run", line: 39, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 40, file: !5, scope: !6, type: !40, retainedNodes: !2)
!40 = !DISubroutineType(types: !41)
!41 = !{null, !42}
!42 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, baseType: !43)
!43 = !DICompositeType(tag: DW_TAG_structure_type, name: "__block_literal_generic", line: 40, size: 256, flags: DIFlagAppleBlock, file: !5, scope: !6, elements: !44)
!44 = !{!45, !46, !47, !48, !49}
!45 = !DIDerivedType(tag: DW_TAG_member, name: "__isa", size: 64, align: 64, file: !5, scope: !6, baseType: !30)
!46 = !DIDerivedType(tag: DW_TAG_member, name: "__flags", size: 32, align: 32, offset: 64, file: !5, scope: !6, baseType: !11)
!47 = !DIDerivedType(tag: DW_TAG_member, name: "__reserved", size: 32, align: 32, offset: 96, file: !5, scope: !6, baseType: !11)
!48 = !DIDerivedType(tag: DW_TAG_member, name: "__FuncPtr", size: 64, align: 64, offset: 128, file: !5, scope: !6, baseType: !30)
!49 = !DIDerivedType(tag: DW_TAG_member, name: "__descriptor", line: 40, size: 64, align: 64, offset: 192, file: !5, scope: !6, baseType: !50)
!50 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, baseType: !51)
!51 = !DICompositeType(tag: DW_TAG_structure_type, name: "__block_descriptor", line: 40, size: 128, flags: DIFlagAppleBlock, file: !5, scope: !6, elements: !52)
!52 = !{!53, !55}
!53 = !DIDerivedType(tag: DW_TAG_member, name: "reserved", size: 64, align: 64, file: !5, scope: !6, baseType: !54)
!54 = !DIBasicType(tag: DW_TAG_base_type, name: "long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!55 = !DIDerivedType(tag: DW_TAG_member, name: "Size", size: 64, align: 64, offset: 64, file: !5, scope: !6, baseType: !54)
!56 = !{i32 1, !"Objective-C Version", i32 2}
!57 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!58 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!59 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
!60 = !DILocalVariable(name: "self", line: 46, arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !13, file: !32, type: !61)
!61 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !4)
!62 = !DILocation(line: 46, scope: !13)
!63 = !DILocalVariable(name: "_cmd", line: 46, arg: 2, flags: DIFlagArtificial, scope: !13, file: !32, type: !64)
!64 = !DIDerivedType(tag: DW_TAG_typedef, name: "SEL", line: 46, file: !5, baseType: !25)
!65 = !DILocation(line: 48, scope: !66)
!66 = distinct !DILexicalBlock(line: 47, column: 0, file: !5, scope: !13)
!67 = !{}
!68 = !DILocation(line: 49, scope: !69)
!69 = distinct !DILexicalBlock(line: 48, column: 0, file: !5, scope: !66)
!70 = !DILocation(line: 53, scope: !69)
!71 = !DILocation(line: 54, scope: !66)
!72 = !DILocalVariable(name: "block", line: 39, arg: 1, scope: !39, file: !6, type: !42)
!73 = !DILocation(line: 39, scope: !39)
!74 = !DILocation(line: 41, scope: !39)
!75 = !DILocation(line: 42, scope: !39)
!76 = !DILocalVariable(name: ".block_descriptor", line: 49, arg: 1, flags: DIFlagArtificial, scope: !27, file: !6, type: !77)
!77 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, baseType: !78)
!78 = !DICompositeType(tag: DW_TAG_structure_type, name: "__block_literal_1", line: 49, size: 320, align: 64, file: !5, scope: !6, elements: !79)
!79 = !{!80, !81, !82, !83, !84, !87}
!80 = !DIDerivedType(tag: DW_TAG_member, name: "__isa", line: 49, size: 64, align: 64, file: !5, scope: !6, baseType: !30)
!81 = !DIDerivedType(tag: DW_TAG_member, name: "__flags", line: 49, size: 32, align: 32, offset: 64, file: !5, scope: !6, baseType: !11)
!82 = !DIDerivedType(tag: DW_TAG_member, name: "__reserved", line: 49, size: 32, align: 32, offset: 96, file: !5, scope: !6, baseType: !11)
!83 = !DIDerivedType(tag: DW_TAG_member, name: "__FuncPtr", line: 49, size: 64, align: 64, offset: 128, file: !5, scope: !6, baseType: !30)
!84 = !DIDerivedType(tag: DW_TAG_member, name: "__descriptor", line: 49, size: 64, align: 64, offset: 192, file: !5, scope: !6, baseType: !85)
!85 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !86)
!86 = !DICompositeType(tag: DW_TAG_structure_type, name: "__block_descriptor_withcopydispose", line: 49, flags: DIFlagFwdDecl, file: !1)
!87 = !DIDerivedType(tag: DW_TAG_member, name: "self", line: 49, size: 64, align: 64, offset: 256, file: !5, scope: !6, baseType: !61)
!88 = !DILocation(line: 49, scope: !27)
!89 = !DILocalVariable(name: "self", line: 52, scope: !27, file: !32, type: !23)
!90 = !DILocation(line: 52, scope: !27)
!91 = !DILocalVariable(name: "d", line: 50, scope: !92, file: !6, type: !93)
!92 = distinct !DILexicalBlock(line: 49, column: 0, file: !5, scope: !27)
!93 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !94)
!94 = !DICompositeType(tag: DW_TAG_structure_type, name: "NSMutableDictionary", line: 30, align: 8, runtimeLang: DW_LANG_ObjC, file: !5, scope: !6, elements: !95)
!95 = !{!96}
!96 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !94, baseType: !97)
!97 = !DICompositeType(tag: DW_TAG_structure_type, name: "NSDictionary", line: 26, align: 8, runtimeLang: DW_LANG_ObjC, file: !5, scope: !6, elements: !98)
!98 = !{!99}
!99 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !97, baseType: !9)
!100 = !DILocation(line: 50, scope: !92)
!101 = !DILocation(line: 51, scope: !92)
!102 = !DILocalVariable(name: "", line: 52, arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !31, file: !32, type: !30)
!103 = !DILocation(line: 52, scope: !31)
!104 = !DILocalVariable(name: "", line: 52, arg: 2, flags: DIFlagArtificial, scope: !31, file: !32, type: !30)
!105 = !DILocalVariable(name: "", line: 52, arg: 1, flags: DIFlagArtificial | DIFlagObjectPointer, scope: !35, file: !32, type: !30)
!106 = !DILocation(line: 52, scope: !35)
!107 = !DILocalVariable(name: "a", line: 61, scope: !36, file: !6, type: !61)
!108 = !DILocation(line: 61, scope: !36)
!109 = !DILocation(line: 62, scope: !36)
!110 = !{i32 1, !"Debug Info Version", i32 3}
!111 = !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 32)

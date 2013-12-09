; RUN: llc -mtriple x86_64-apple-darwin -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -debug-dump=info %t.o | FileCheck %s

; Generated from llvm/tools/clang/test/CodeGenObjC/debug-info-blocks.m
; rdar://problem/9279956
; test that the DW_AT_location of self is at ( fbreg +{{[0-9]+}}, deref, +{{[0-9]+}} )

; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_name{{.*}}_block_invoke

; CHECK-NOT: DW_TAG_subprogram
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_name{{.*}}.block_descriptor
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_location

; CHECK-NOT: DW_TAG_subprogram
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_name{{.*}}"self"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_type{{.*}}{[[APTR:.*]]}
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_artificial
; CHECK-NOT: DW_TAG
; 0x06 = DW_OP_deref
; 0x23 = DW_OP_uconst
; 0x91 = DW_OP_fbreg
; CHECK: DW_AT_location{{.*}}91 {{[0-9]+}} 06 23 {{[0-9]+}} )

; CHECK: [[A:.*]]:   DW_TAG_structure_type
; CHECK-NEXT: DW_AT_APPLE_objc_complete_type
; CHECK-NEXT: DW_AT_name{{.*}}"A"
; CHECK: [[APTR]]:   DW_TAG_pointer_type [5]
; CHECK-NEXT: {[[A]]}


; ModuleID = 'llvm/tools/clang/test/CodeGenObjC/debug-info-blocks.m'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

%0 = type opaque
%1 = type opaque
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
%struct._message_ref_t = type { i8*, i8* }
%struct._objc_super = type { i8*, i8* }
%struct.__block_descriptor = type { i64, i64 }
%struct.__block_literal_generic = type { i8*, i32, i32, i8*, %struct.__block_descriptor* }

@"OBJC_CLASS_$_A" = global %struct._class_t { %struct._class_t* @"OBJC_METACLASS_$_A", %struct._class_t* @"OBJC_CLASS_$_NSObject", %struct._objc_cache* @_objc_empty_cache, i8* (i8*, i8*)** @_objc_empty_vtable, %struct._class_ro_t* @"\01l_OBJC_CLASS_RO_$_A" }, section "__DATA, __objc_data", align 8
@"\01L_OBJC_CLASSLIST_SUP_REFS_$_" = internal global %struct._class_t* @"OBJC_CLASS_$_A", section "__DATA, __objc_superrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_METH_VAR_NAME_" = internal global [5 x i8] c"init\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_SELECTOR_REFERENCES_" = internal externally_initialized global i8* getelementptr inbounds ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"OBJC_CLASS_$_NSMutableDictionary" = external global %struct._class_t
@"\01L_OBJC_CLASSLIST_REFERENCES_$_" = internal global %struct._class_t* @"OBJC_CLASS_$_NSMutableDictionary", section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_METH_VAR_NAME_1" = internal global [6 x i8] c"alloc\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01l_objc_msgSend_fixup_alloc" = weak hidden global { i8* (i8*, %struct._message_ref_t*, ...)*, i8* } { i8* (i8*, %struct._message_ref_t*, ...)* @objc_msgSend_fixup, i8* getelementptr inbounds ([6 x i8]* @"\01L_OBJC_METH_VAR_NAME_1", i32 0, i32 0) }, section "__DATA, __objc_msgrefs, coalesced", align 16
@"\01L_OBJC_METH_VAR_NAME_2" = internal global [6 x i8] c"count\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01l_objc_msgSend_fixup_count" = weak hidden global { i8* (i8*, %struct._message_ref_t*, ...)*, i8* } { i8* (i8*, %struct._message_ref_t*, ...)* @objc_msgSend_fixup, i8* getelementptr inbounds ([6 x i8]* @"\01L_OBJC_METH_VAR_NAME_2", i32 0, i32 0) }, section "__DATA, __objc_msgrefs, coalesced", align 16
@"OBJC_IVAR_$_A.ivar" = global i64 0, section "__DATA, __objc_ivar", align 8
@_NSConcreteStackBlock = external global i8*
@.str = private unnamed_addr constant [6 x i8] c"v8@?0\00", align 1
@__block_descriptor_tmp = internal constant { i64, i64, i8*, i8*, i8*, i64 } { i64 0, i64 40, i8* bitcast (void (i8*, i8*)* @__copy_helper_block_ to i8*), i8* bitcast (void (i8*)* @__destroy_helper_block_ to i8*), i8* getelementptr inbounds ([6 x i8]* @.str, i32 0, i32 0), i64 256 }
@_objc_empty_cache = external global %struct._objc_cache
@_objc_empty_vtable = external global i8* (i8*, i8*)*
@"OBJC_METACLASS_$_NSObject" = external global %struct._class_t
@"\01L_OBJC_CLASS_NAME_" = internal global [2 x i8] c"A\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@"\01l_OBJC_METACLASS_RO_$_A" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, i8* null, i8* getelementptr inbounds ([2 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), %struct.__method_list_t* null, %struct._objc_protocol_list* null, %struct._ivar_list_t* null, i8* null, %struct._prop_list_t* null }, section "__DATA, __objc_const", align 8
@"OBJC_METACLASS_$_A" = global %struct._class_t { %struct._class_t* @"OBJC_METACLASS_$_NSObject", %struct._class_t* @"OBJC_METACLASS_$_NSObject", %struct._objc_cache* @_objc_empty_cache, i8* (i8*, i8*)** @_objc_empty_vtable, %struct._class_ro_t* @"\01l_OBJC_METACLASS_RO_$_A" }, section "__DATA, __objc_data", align 8
@"OBJC_CLASS_$_NSObject" = external global %struct._class_t
@"\01L_OBJC_METH_VAR_TYPE_" = internal global [8 x i8] c"@16@0:8\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@"\01l_OBJC_$_INSTANCE_METHODS_A" = internal global { i32, i32, [1 x %struct._objc_method] } { i32 24, i32 1, [1 x %struct._objc_method] [%struct._objc_method { i8* getelementptr inbounds ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @"\01L_OBJC_METH_VAR_TYPE_", i32 0, i32 0), i8* bitcast (i8* (%0*, i8*)* @"\01-[A init]" to i8*) }] }, section "__DATA, __objc_const", align 8
@"\01L_OBJC_METH_VAR_NAME_3" = internal global [5 x i8] c"ivar\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"\01L_OBJC_METH_VAR_TYPE_4" = internal global [2 x i8] c"i\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@"\01l_OBJC_$_INSTANCE_VARIABLES_A" = internal global { i32, i32, [1 x %struct._ivar_t] } { i32 32, i32 1, [1 x %struct._ivar_t] [%struct._ivar_t { i64* @"OBJC_IVAR_$_A.ivar", i8* getelementptr inbounds ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_3", i32 0, i32 0), i8* getelementptr inbounds ([2 x i8]* @"\01L_OBJC_METH_VAR_TYPE_4", i32 0, i32 0), i32 2, i32 4 }] }, section "__DATA, __objc_const", align 8
@"\01l_OBJC_CLASS_RO_$_A" = internal global %struct._class_ro_t { i32 0, i32 0, i32 4, i8* null, i8* getelementptr inbounds ([2 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), %struct.__method_list_t* bitcast ({ i32, i32, [1 x %struct._objc_method] }* @"\01l_OBJC_$_INSTANCE_METHODS_A" to %struct.__method_list_t*), %struct._objc_protocol_list* null, %struct._ivar_list_t* bitcast ({ i32, i32, [1 x %struct._ivar_t] }* @"\01l_OBJC_$_INSTANCE_VARIABLES_A" to %struct._ivar_list_t*), i8* null, %struct._prop_list_t* null }, section "__DATA, __objc_const", align 8
@"\01L_OBJC_CLASSLIST_REFERENCES_$_5" = internal global %struct._class_t* @"OBJC_CLASS_$_A", section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_LABEL_CLASS_$" = internal global [1 x i8*] [i8* bitcast (%struct._class_t* @"OBJC_CLASS_$_A" to i8*)], section "__DATA, __objc_classlist, regular, no_dead_strip", align 8
@llvm.used = appending global [14 x i8*] [i8* bitcast (%struct._class_t** @"\01L_OBJC_CLASSLIST_SUP_REFS_$_" to i8*), i8* getelementptr inbounds ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_", i32 0, i32 0), i8* bitcast (i8** @"\01L_OBJC_SELECTOR_REFERENCES_" to i8*), i8* bitcast (%struct._class_t** @"\01L_OBJC_CLASSLIST_REFERENCES_$_" to i8*), i8* getelementptr inbounds ([6 x i8]* @"\01L_OBJC_METH_VAR_NAME_1", i32 0, i32 0), i8* getelementptr inbounds ([6 x i8]* @"\01L_OBJC_METH_VAR_NAME_2", i32 0, i32 0), i8* getelementptr inbounds ([2 x i8]* @"\01L_OBJC_CLASS_NAME_", i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @"\01L_OBJC_METH_VAR_TYPE_", i32 0, i32 0), i8* bitcast ({ i32, i32, [1 x %struct._objc_method] }* @"\01l_OBJC_$_INSTANCE_METHODS_A" to i8*), i8* getelementptr inbounds ([5 x i8]* @"\01L_OBJC_METH_VAR_NAME_3", i32 0, i32 0), i8* getelementptr inbounds ([2 x i8]* @"\01L_OBJC_METH_VAR_TYPE_4", i32 0, i32 0), i8* bitcast ({ i32, i32, [1 x %struct._ivar_t] }* @"\01l_OBJC_$_INSTANCE_VARIABLES_A" to i8*), i8* bitcast (%struct._class_t** @"\01L_OBJC_CLASSLIST_REFERENCES_$_5" to i8*), i8* bitcast ([1 x i8*]* @"\01L_OBJC_LABEL_CLASS_$" to i8*)], section "llvm.metadata"

define internal i8* @"\01-[A init]"(%0* %self, i8* %_cmd) #0 {
  %1 = alloca %0*, align 8
  %2 = alloca i8*, align 8
  %3 = alloca %struct._objc_super
  %4 = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>, align 8
  store %0* %self, %0** %1, align 8
  call void @llvm.dbg.declare(metadata !{%0** %1}, metadata !60), !dbg !62
  store i8* %_cmd, i8** %2, align 8
  call void @llvm.dbg.declare(metadata !{i8** %2}, metadata !63), !dbg !62
  %5 = load %0** %1, !dbg !65
  %6 = bitcast %0* %5 to i8*, !dbg !65
  %7 = getelementptr inbounds %struct._objc_super* %3, i32 0, i32 0, !dbg !65
  store i8* %6, i8** %7, !dbg !65
  %8 = load %struct._class_t** @"\01L_OBJC_CLASSLIST_SUP_REFS_$_", !dbg !65
  %9 = bitcast %struct._class_t* %8 to i8*, !dbg !65
  %10 = getelementptr inbounds %struct._objc_super* %3, i32 0, i32 1, !dbg !65
  store i8* %9, i8** %10, !dbg !65
  %11 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", !dbg !65, !invariant.load !67
  %12 = call i8* bitcast (i8* (%struct._objc_super*, i8*, ...)* @objc_msgSendSuper2 to i8* (%struct._objc_super*, i8*)*)(%struct._objc_super* %3, i8* %11), !dbg !65
  %13 = bitcast i8* %12 to %0*, !dbg !65
  store %0* %13, %0** %1, align 8, !dbg !65
  %14 = icmp ne %0* %13, null, !dbg !65
  br i1 %14, label %15, label %24, !dbg !65

; <label>:15                                      ; preds = %0
  %16 = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %4, i32 0, i32 0, !dbg !68
  store i8* bitcast (i8** @_NSConcreteStackBlock to i8*), i8** %16, !dbg !68
  %17 = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %4, i32 0, i32 1, !dbg !68
  store i32 -1040187392, i32* %17, !dbg !68
  %18 = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %4, i32 0, i32 2, !dbg !68
  store i32 0, i32* %18, !dbg !68
  %19 = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %4, i32 0, i32 3, !dbg !68
  store i8* bitcast (void (i8*)* @"__9-[A init]_block_invoke" to i8*), i8** %19, !dbg !68
  %20 = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %4, i32 0, i32 4, !dbg !68
  store %struct.__block_descriptor* bitcast ({ i64, i64, i8*, i8*, i8*, i64 }* @__block_descriptor_tmp to %struct.__block_descriptor*), %struct.__block_descriptor** %20, !dbg !68
  %21 = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %4, i32 0, i32 5, !dbg !68
  %22 = load %0** %1, align 8, !dbg !68
  store %0* %22, %0** %21, align 8, !dbg !68
  %23 = bitcast <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %4 to void ()*, !dbg !68
  call void @run(void ()* %23), !dbg !68
  br label %24, !dbg !70

; <label>:24                                      ; preds = %15, %0
  %25 = load %0** %1, align 8, !dbg !71
  %26 = bitcast %0* %25 to i8*, !dbg !71
  ret i8* %26, !dbg !71
}

declare void @llvm.dbg.declare(metadata, metadata) #1

declare i8* @objc_msgSendSuper2(%struct._objc_super*, i8*, ...)

define internal void @run(void ()* %block) #0 {
  %1 = alloca void ()*, align 8
  store void ()* %block, void ()** %1, align 8
  call void @llvm.dbg.declare(metadata !{void ()** %1}, metadata !72), !dbg !73
  %2 = load void ()** %1, align 8, !dbg !74
  %3 = bitcast void ()* %2 to %struct.__block_literal_generic*, !dbg !74
  %4 = getelementptr inbounds %struct.__block_literal_generic* %3, i32 0, i32 3, !dbg !74
  %5 = bitcast %struct.__block_literal_generic* %3 to i8*, !dbg !74
  %6 = load i8** %4, !dbg !74
  %7 = bitcast i8* %6 to void (i8*)*, !dbg !74
  call void %7(i8* %5), !dbg !74
  ret void, !dbg !75
}

define internal void @"__9-[A init]_block_invoke"(i8* %.block_descriptor) #0 {
  %1 = alloca i8*, align 8
  %2 = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>*, align 8
  %d = alloca %1*, align 8
  store i8* %.block_descriptor, i8** %1, align 8
  %3 = load i8** %1
  call void @llvm.dbg.value(metadata !{i8* %3}, i64 0, metadata !76), !dbg !88
  call void @llvm.dbg.declare(metadata !{i8* %.block_descriptor}, metadata !76), !dbg !88
  %4 = bitcast i8* %.block_descriptor to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>*, !dbg !88
  store <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %4, <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>** %2, align 8, !dbg !88
  %5 = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %4, i32 0, i32 5, !dbg !88
  call void @llvm.dbg.declare(metadata !{<{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>** %2}, metadata !89), !dbg !90
  call void @llvm.dbg.declare(metadata !{%1** %d}, metadata !91), !dbg !100
  %6 = load %struct._class_t** @"\01L_OBJC_CLASSLIST_REFERENCES_$_", !dbg !100
  %7 = bitcast %struct._class_t* %6 to i8*, !dbg !100
  %8 = load i8** getelementptr inbounds (%struct._message_ref_t* bitcast ({ i8* (i8*, %struct._message_ref_t*, ...)*, i8* }* @"\01l_objc_msgSend_fixup_alloc" to %struct._message_ref_t*), i32 0, i32 0), !dbg !100
  %9 = bitcast i8* %8 to i8* (i8*, i8*)*, !dbg !100
  %10 = call i8* %9(i8* %7, i8* bitcast ({ i8* (i8*, %struct._message_ref_t*, ...)*, i8* }* @"\01l_objc_msgSend_fixup_alloc" to i8*)), !dbg !100
  %11 = bitcast i8* %10 to %1*, !dbg !100
  %12 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", !dbg !100, !invariant.load !67
  %13 = bitcast %1* %11 to i8*, !dbg !100
  %14 = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* %13, i8* %12), !dbg !100
  %15 = bitcast i8* %14 to %1*, !dbg !100
  store %1* %15, %1** %d, align 8, !dbg !100
  %16 = load %1** %d, align 8, !dbg !101
  %17 = bitcast %1* %16 to i8*, !dbg !101
  %18 = load i8** getelementptr inbounds (%struct._message_ref_t* bitcast ({ i8* (i8*, %struct._message_ref_t*, ...)*, i8* }* @"\01l_objc_msgSend_fixup_count" to %struct._message_ref_t*), i32 0, i32 0), !dbg !101
  %19 = bitcast i8* %18 to i32 (i8*, i8*)*, !dbg !101
  %20 = call i32 %19(i8* %17, i8* bitcast ({ i8* (i8*, %struct._message_ref_t*, ...)*, i8* }* @"\01l_objc_msgSend_fixup_count" to i8*)), !dbg !101
  %21 = add nsw i32 42, %20, !dbg !101
  %22 = load %0** %5, align 8, !dbg !101
  %23 = load i64* @"OBJC_IVAR_$_A.ivar", !dbg !101, !invariant.load !67
  %24 = bitcast %0* %22 to i8*, !dbg !101
  %25 = getelementptr inbounds i8* %24, i64 %23, !dbg !101
  %26 = bitcast i8* %25 to i32*, !dbg !101
  store i32 %21, i32* %26, align 4, !dbg !101
  ret void, !dbg !90
}

declare void @llvm.dbg.value(metadata, i64, metadata) #1

declare i8* @objc_msgSend_fixup(i8*, %struct._message_ref_t*, ...)

declare i8* @objc_msgSend(i8*, i8*, ...) #2

define internal void @__copy_helper_block_(i8*, i8*) {
  %3 = alloca i8*, align 8
  %4 = alloca i8*, align 8
  store i8* %0, i8** %3, align 8
  call void @llvm.dbg.declare(metadata !{i8** %3}, metadata !102), !dbg !103
  store i8* %1, i8** %4, align 8
  call void @llvm.dbg.declare(metadata !{i8** %4}, metadata !104), !dbg !103
  %5 = load i8** %4, !dbg !103
  %6 = bitcast i8* %5 to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>*, !dbg !103
  %7 = load i8** %3, !dbg !103
  %8 = bitcast i8* %7 to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>*, !dbg !103
  %9 = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %6, i32 0, i32 5, !dbg !103
  %10 = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %8, i32 0, i32 5, !dbg !103
  %11 = load %0** %9, !dbg !103
  %12 = bitcast %0* %11 to i8*, !dbg !103
  %13 = bitcast %0** %10 to i8*, !dbg !103
  call void @_Block_object_assign(i8* %13, i8* %12, i32 3) #3, !dbg !103
  ret void, !dbg !103
}

declare void @_Block_object_assign(i8*, i8*, i32)

define internal void @__destroy_helper_block_(i8*) {
  %2 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  call void @llvm.dbg.declare(metadata !{i8** %2}, metadata !105), !dbg !106
  %3 = load i8** %2, !dbg !106
  %4 = bitcast i8* %3 to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>*, !dbg !106
  %5 = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %4, i32 0, i32 5, !dbg !106
  %6 = load %0** %5, !dbg !106
  %7 = bitcast %0* %6 to i8*, !dbg !106
  call void @_Block_object_dispose(i8* %7, i32 3) #3, !dbg !106
  ret void, !dbg !106
}

declare void @_Block_object_dispose(i8*, i32)

define i32 @main() #0 {
  %1 = alloca i32, align 4
  %a = alloca %0*, align 8
  store i32 0, i32* %1
  call void @llvm.dbg.declare(metadata !{%0** %a}, metadata !107), !dbg !108
  %2 = load %struct._class_t** @"\01L_OBJC_CLASSLIST_REFERENCES_$_5", !dbg !108
  %3 = bitcast %struct._class_t* %2 to i8*, !dbg !108
  %4 = load i8** getelementptr inbounds (%struct._message_ref_t* bitcast ({ i8* (i8*, %struct._message_ref_t*, ...)*, i8* }* @"\01l_objc_msgSend_fixup_alloc" to %struct._message_ref_t*), i32 0, i32 0), !dbg !108
  %5 = bitcast i8* %4 to i8* (i8*, i8*)*, !dbg !108
  %6 = call i8* %5(i8* %3, i8* bitcast ({ i8* (i8*, %struct._message_ref_t*, ...)*, i8* }* @"\01l_objc_msgSend_fixup_alloc" to i8*)), !dbg !108
  %7 = bitcast i8* %6 to %0*, !dbg !108
  %8 = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", !dbg !108, !invariant.load !67
  %9 = bitcast %0* %7 to i8*, !dbg !108
  %10 = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* %9, i8* %8), !dbg !108
  %11 = bitcast i8* %10 to %0*, !dbg !108
  store %0* %11, %0** %a, align 8, !dbg !108
  ret i32 0, !dbg !109
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nonlazybind }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!56, !57, !58, !59, !110}

!0 = metadata !{i32 786449, metadata !1, i32 16, metadata !"clang version 3.3 ", i1 false, metadata !"", i32 2, metadata !2, metadata !3, metadata !12, metadata !2,  metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [llvm/tools/clang/test/CodeGenObjC/<unknown>] [DW_LANG_ObjC]
!1 = metadata !{metadata !"llvm/tools/clang/test/CodeGenObjC/<unknown>", metadata !"llvm/_build.ninja.Debug"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786451, metadata !5, metadata !6, metadata !"A", i32 33, i64 32, i64 32, i32 0, i32 512, null, metadata !7, i32 16, null, null, null} ; [ DW_TAG_structure_type ] [A] [line 33, size 32, align 32, offset 0] [def] [from ]
!5 = metadata !{metadata !"llvm/tools/clang/test/CodeGenObjC/debug-info-blocks.m", metadata !"llvm/_build.ninja.Debug"}
!6 = metadata !{i32 786473, metadata !5}          ; [ DW_TAG_file_type ] [llvm/tools/clang/test/CodeGenObjC/debug-info-blocks.m]
!7 = metadata !{metadata !8, metadata !10}
!8 = metadata !{i32 786460, null, metadata !4, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !9} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [from NSObject]
!9 = metadata !{i32 786451, metadata !5, metadata !6, metadata !"NSObject", i32 21, i64 0, i64 8, i32 0, i32 0, null, metadata !2, i32 16, null, null, null} ; [ DW_TAG_structure_type ] [NSObject] [line 21, size 0, align 8, offset 0] [def] [from ]
!10 = metadata !{i32 786445, metadata !5, metadata !6, metadata !"ivar", i32 35, i64 32, i64 32, i64 0, i32 0, metadata !11, null} ; [ DW_TAG_member ] [ivar] [line 35, size 32, align 32, offset 0] [from int]
!11 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = metadata !{metadata !13, metadata !27, metadata !31, metadata !35, metadata !36, metadata !39}
!13 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"-[A init]", metadata !"-[A init]", metadata !"", i32 46, metadata !14, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, i8* (%0*, i8*)* @"\01-[A init]", null, null, metadata !2, i32 46} ; [ DW_TAG_subprogram ] [line 46] [local] [def] [-[A init]]
!14 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !15, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = metadata !{metadata !16, metadata !23, metadata !24}
!16 = metadata !{i32 786454, metadata !5, null, metadata !"id", i32 46, i64 0, i64 0, i64 0, i32 0, metadata !17} ; [ DW_TAG_typedef ] [id] [line 46, size 0, align 0, offset 0] [from ]
!17 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !18} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from objc_object]
!18 = metadata !{i32 786451, metadata !1, null, metadata !"objc_object", i32 0, i64 0, i64 0, i32 0, i32 0, null, metadata !19, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [objc_object] [line 0, size 0, align 0, offset 0] [def] [from ]
!19 = metadata !{metadata !20}
!20 = metadata !{i32 786445, metadata !1, metadata !18, metadata !"isa", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !21} ; [ DW_TAG_member ] [isa] [line 0, size 64, align 0, offset 0] [from ]
!21 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !22} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from objc_class]
!22 = metadata !{i32 786451, metadata !1, null, metadata !"objc_class", i32 0, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [objc_class] [line 0, size 0, align 0, offset 0] [decl] [from ]
!23 = metadata !{i32 786447, i32 0, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !4} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from A]
!24 = metadata !{i32 786454, metadata !5, null, metadata !"SEL", i32 46, i64 0, i64 0, i64 0, i32 64, metadata !25} ; [ DW_TAG_typedef ] [SEL] [line 46, size 0, align 0, offset 0] [artificial] [from ]
!25 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !26} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from objc_selector]
!26 = metadata !{i32 786451, metadata !1, null, metadata !"objc_selector", i32 0, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [objc_selector] [line 0, size 0, align 0, offset 0] [decl] [from ]
!27 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"__9-[A init]_block_invoke", metadata !"__9-[A init]_block_invoke", metadata !"", i32 49, metadata !28, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8*)* @"__9-[A init]_block_invoke", null, null, metadata !2, i32 49} ; [ DW_TAG_subprogram ] [line 49] [local] [def] [__9-[A init]_block_invoke]
!28 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !29, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!29 = metadata !{null, metadata !30}
!30 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, null} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!31 = metadata !{i32 786478, metadata !1, metadata !32, metadata !"__copy_helper_block_", metadata !"__copy_helper_block_", metadata !"", i32 52, metadata !33, i1 true, i1 true, i32 0, i32 0, null, i32 0, i1 false, void (i8*, i8*)* @__copy_helper_block_, null, null, metadata !2, i32 52} ; [ DW_TAG_subprogram ] [line 52] [local] [def] [__copy_helper_block_]
!32 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [llvm/tools/clang/test/CodeGenObjC/<unknown>]
!33 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !34, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!34 = metadata !{null, metadata !30, metadata !30}
!35 = metadata !{i32 786478, metadata !1, metadata !32, metadata !"__destroy_helper_block_", metadata !"__destroy_helper_block_", metadata !"", i32 52, metadata !28, i1 true, i1 true, i32 0, i32 0, null, i32 0, i1 false, void (i8*)* @__destroy_helper_block_, null, null, metadata !2, i32 52} ; [ DW_TAG_subprogram ] [line 52] [local] [def] [__destroy_helper_block_]
!36 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"main", metadata !"main", metadata !"", i32 59, metadata !37, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, i32 ()* @main, null, null, metadata !2, i32 60} ; [ DW_TAG_subprogram ] [line 59] [def] [scope 60] [main]
!37 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !38, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!38 = metadata !{metadata !11}
!39 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"run", metadata !"run", metadata !"", i32 39, metadata !40, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (void ()*)* @run, null, null, metadata !2, i32 40} ; [ DW_TAG_subprogram ] [line 39] [local] [def] [scope 40] [run]
!40 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !41, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!41 = metadata !{null, metadata !42}
!42 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !43} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __block_literal_generic]
!43 = metadata !{i32 786451, metadata !5, metadata !6, metadata !"__block_literal_generic", i32 40, i64 256, i64 0, i32 0, i32 8, null, metadata !44, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [__block_literal_generic] [line 40, size 256, align 0, offset 0] [def] [from ]
!44 = metadata !{metadata !45, metadata !46, metadata !47, metadata !48, metadata !49}
!45 = metadata !{i32 786445, metadata !5, metadata !6, metadata !"__isa", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !30} ; [ DW_TAG_member ] [__isa] [line 0, size 64, align 64, offset 0] [from ]
!46 = metadata !{i32 786445, metadata !5, metadata !6, metadata !"__flags", i32 0, i64 32, i64 32, i64 64, i32 0, metadata !11} ; [ DW_TAG_member ] [__flags] [line 0, size 32, align 32, offset 64] [from int]
!47 = metadata !{i32 786445, metadata !5, metadata !6, metadata !"__reserved", i32 0, i64 32, i64 32, i64 96, i32 0, metadata !11} ; [ DW_TAG_member ] [__reserved] [line 0, size 32, align 32, offset 96] [from int]
!48 = metadata !{i32 786445, metadata !5, metadata !6, metadata !"__FuncPtr", i32 0, i64 64, i64 64, i64 128, i32 0, metadata !30} ; [ DW_TAG_member ] [__FuncPtr] [line 0, size 64, align 64, offset 128] [from ]
!49 = metadata !{i32 786445, metadata !5, metadata !6, metadata !"__descriptor", i32 40, i64 64, i64 64, i64 192, i32 0, metadata !50} ; [ DW_TAG_member ] [__descriptor] [line 40, size 64, align 64, offset 192] [from ]
!50 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !51} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __block_descriptor]
!51 = metadata !{i32 786451, metadata !5, metadata !6, metadata !"__block_descriptor", i32 40, i64 128, i64 0, i32 0, i32 8, null, metadata !52, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [__block_descriptor] [line 40, size 128, align 0, offset 0] [def] [from ]
!52 = metadata !{metadata !53, metadata !55}
!53 = metadata !{i32 786445, metadata !5, metadata !6, metadata !"reserved", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !54} ; [ DW_TAG_member ] [reserved] [line 0, size 64, align 64, offset 0] [from long unsigned int]
!54 = metadata !{i32 786468, null, null, metadata !"long unsigned int", i32 0, i64 64, i64 64, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ] [long unsigned int] [line 0, size 64, align 64, offset 0, enc DW_ATE_unsigned]
!55 = metadata !{i32 786445, metadata !5, metadata !6, metadata !"Size", i32 0, i64 64, i64 64, i64 64, i32 0, metadata !54} ; [ DW_TAG_member ] [Size] [line 0, size 64, align 64, offset 64] [from long unsigned int]
!56 = metadata !{i32 1, metadata !"Objective-C Version", i32 2}
!57 = metadata !{i32 1, metadata !"Objective-C Image Info Version", i32 0}
!58 = metadata !{i32 1, metadata !"Objective-C Image Info Section", metadata !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!59 = metadata !{i32 4, metadata !"Objective-C Garbage Collection", i32 0}
!60 = metadata !{i32 786689, metadata !13, metadata !"self", metadata !32, i32 16777262, metadata !61, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [self] [line 46]
!61 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !4} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!62 = metadata !{i32 46, i32 0, metadata !13, null}
!63 = metadata !{i32 786689, metadata !13, metadata !"_cmd", metadata !32, i32 33554478, metadata !64, i32 64, i32 0} ; [ DW_TAG_arg_variable ] [_cmd] [line 46]
!64 = metadata !{i32 786454, metadata !5, null, metadata !"SEL", i32 46, i64 0, i64 0, i64 0, i32 0, metadata !25} ; [ DW_TAG_typedef ] [SEL] [line 46, size 0, align 0, offset 0] [from ]
!65 = metadata !{i32 48, i32 0, metadata !66, null}
!66 = metadata !{i32 786443, metadata !5, metadata !13, i32 47, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [llvm/tools/clang/test/CodeGenObjC/debug-info-blocks.m]
!67 = metadata !{}
!68 = metadata !{i32 49, i32 0, metadata !69, null}
!69 = metadata !{i32 786443, metadata !5, metadata !66, i32 48, i32 0, i32 1} ; [ DW_TAG_lexical_block ] [llvm/tools/clang/test/CodeGenObjC/debug-info-blocks.m]
!70 = metadata !{i32 53, i32 0, metadata !69, null}
!71 = metadata !{i32 54, i32 0, metadata !66, null}
!72 = metadata !{i32 786689, metadata !39, metadata !"block", metadata !6, i32 16777255, metadata !42, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [block] [line 39]
!73 = metadata !{i32 39, i32 0, metadata !39, null}
!74 = metadata !{i32 41, i32 0, metadata !39, null}
!75 = metadata !{i32 42, i32 0, metadata !39, null}
!76 = metadata !{i32 786689, metadata !27, metadata !".block_descriptor", metadata !6, i32 16777265, metadata !77, i32 64, i32 0} ; [ DW_TAG_arg_variable ] [.block_descriptor] [line 49]
!77 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !78} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __block_literal_1]
!78 = metadata !{i32 786451, metadata !5, metadata !6, metadata !"__block_literal_1", i32 49, i64 320, i64 64, i32 0, i32 0, null, metadata !79, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [__block_literal_1] [line 49, size 320, align 64, offset 0] [def] [from ]
!79 = metadata !{metadata !80, metadata !81, metadata !82, metadata !83, metadata !84, metadata !87}
!80 = metadata !{i32 786445, metadata !5, metadata !6, metadata !"__isa", i32 49, i64 64, i64 64, i64 0, i32 0, metadata !30} ; [ DW_TAG_member ] [__isa] [line 49, size 64, align 64, offset 0] [from ]
!81 = metadata !{i32 786445, metadata !5, metadata !6, metadata !"__flags", i32 49, i64 32, i64 32, i64 64, i32 0, metadata !11} ; [ DW_TAG_member ] [__flags] [line 49, size 32, align 32, offset 64] [from int]
!82 = metadata !{i32 786445, metadata !5, metadata !6, metadata !"__reserved", i32 49, i64 32, i64 32, i64 96, i32 0, metadata !11} ; [ DW_TAG_member ] [__reserved] [line 49, size 32, align 32, offset 96] [from int]
!83 = metadata !{i32 786445, metadata !5, metadata !6, metadata !"__FuncPtr", i32 49, i64 64, i64 64, i64 128, i32 0, metadata !30} ; [ DW_TAG_member ] [__FuncPtr] [line 49, size 64, align 64, offset 128] [from ]
!84 = metadata !{i32 786445, metadata !5, metadata !6, metadata !"__descriptor", i32 49, i64 64, i64 64, i64 192, i32 0, metadata !85} ; [ DW_TAG_member ] [__descriptor] [line 49, size 64, align 64, offset 192] [from ]
!85 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !86} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from __block_descriptor_withcopydispose]
!86 = metadata !{i32 786451, metadata !1, null, metadata !"__block_descriptor_withcopydispose", i32 49, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [__block_descriptor_withcopydispose] [line 49, size 0, align 0, offset 0] [decl] [from ]
!87 = metadata !{i32 786445, metadata !5, metadata !6, metadata !"self", i32 49, i64 64, i64 64, i64 256, i32 0, metadata !61} ; [ DW_TAG_member ] [self] [line 49, size 64, align 64, offset 256] [from ]
!88 = metadata !{i32 49, i32 0, metadata !27, null}
!89 = metadata !{i32 786688, metadata !27, metadata !"self", metadata !32, i32 52, metadata !23, i32 0, i32 0, i64 2, i64 1, i64 32} ; [ DW_TAG_auto_variable ] [self] [line 52]
!90 = metadata !{i32 52, i32 0, metadata !27, null}
!91 = metadata !{i32 786688, metadata !92, metadata !"d", metadata !6, i32 50, metadata !93, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [d] [line 50]
!92 = metadata !{i32 786443, metadata !5, metadata !27, i32 49, i32 0, i32 2} ; [ DW_TAG_lexical_block ] [llvm/tools/clang/test/CodeGenObjC/debug-info-blocks.m]
!93 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !94} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from NSMutableDictionary]
!94 = metadata !{i32 786451, metadata !5, metadata !6, metadata !"NSMutableDictionary", i32 30, i64 0, i64 8, i32 0, i32 0, null, metadata !95, i32 16, null, null, null} ; [ DW_TAG_structure_type ] [NSMutableDictionary] [line 30, size 0, align 8, offset 0] [def] [from ]
!95 = metadata !{metadata !96}
!96 = metadata !{i32 786460, null, metadata !94, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !97} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [from NSDictionary]
!97 = metadata !{i32 786451, metadata !5, metadata !6, metadata !"NSDictionary", i32 26, i64 0, i64 8, i32 0, i32 0, null, metadata !98, i32 16, null, null, null} ; [ DW_TAG_structure_type ] [NSDictionary] [line 26, size 0, align 8, offset 0] [def] [from ]
!98 = metadata !{metadata !99}
!99 = metadata !{i32 786460, null, metadata !97, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !9} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [from NSObject]
!100 = metadata !{i32 50, i32 0, metadata !92, null}
!101 = metadata !{i32 51, i32 0, metadata !92, null}
!102 = metadata !{i32 786689, metadata !31, metadata !"", metadata !32, i32 16777268, metadata !30, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [line 52]
!103 = metadata !{i32 52, i32 0, metadata !31, null}
!104 = metadata !{i32 786689, metadata !31, metadata !"", metadata !32, i32 33554484, metadata !30, i32 64, i32 0} ; [ DW_TAG_arg_variable ] [line 52]
!105 = metadata !{i32 786689, metadata !35, metadata !"", metadata !32, i32 16777268, metadata !30, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [line 52]
!106 = metadata !{i32 52, i32 0, metadata !35, null}
!107 = metadata !{i32 786688, metadata !36, metadata !"a", metadata !6, i32 61, metadata !61, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [a] [line 61]
!108 = metadata !{i32 61, i32 0, metadata !36, null}
!109 = metadata !{i32 62, i32 0, metadata !36, null}
!110 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}

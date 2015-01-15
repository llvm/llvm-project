; RUN: llc -mtriple x86_64-apple-darwin -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -debug-dump=info %t.o | FileCheck %s

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
; 0x06 = DW_OP_deref
; 0x23 = DW_OP_uconst
; 0x91 = DW_OP_fbreg
; CHECK: DW_AT_location{{.*}}91 {{[0-9]+}} 06 23 {{[0-9]+}} )
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
  call void @llvm.dbg.declare(metadata %0** %1, metadata !60, metadata !{!"0x102"}), !dbg !62
  store i8* %_cmd, i8** %2, align 8
  call void @llvm.dbg.declare(metadata i8** %2, metadata !63, metadata !{!"0x102"}), !dbg !62
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

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i8* @objc_msgSendSuper2(%struct._objc_super*, i8*, ...)

define internal void @run(void ()* %block) #0 {
  %1 = alloca void ()*, align 8
  store void ()* %block, void ()** %1, align 8
  call void @llvm.dbg.declare(metadata void ()** %1, metadata !72, metadata !{!"0x102"}), !dbg !73
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
  call void @llvm.dbg.value(metadata i8* %3, i64 0, metadata !76, metadata !{!"0x102"}), !dbg !88
  call void @llvm.dbg.declare(metadata i8* %.block_descriptor, metadata !76, metadata !{!"0x102"}), !dbg !88
  %4 = bitcast i8* %.block_descriptor to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>*, !dbg !88
  store <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %4, <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>** %2, align 8, !dbg !88
  %5 = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>* %4, i32 0, i32 5, !dbg !88
  call void @llvm.dbg.declare(metadata <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, %0* }>** %2, metadata !89, metadata !111), !dbg !90
  call void @llvm.dbg.declare(metadata %1** %d, metadata !91, metadata !{!"0x102"}), !dbg !100
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

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

declare i8* @objc_msgSend_fixup(i8*, %struct._message_ref_t*, ...)

declare i8* @objc_msgSend(i8*, i8*, ...) #2

define internal void @__copy_helper_block_(i8*, i8*) {
  %3 = alloca i8*, align 8
  %4 = alloca i8*, align 8
  store i8* %0, i8** %3, align 8
  call void @llvm.dbg.declare(metadata i8** %3, metadata !102, metadata !{!"0x102"}), !dbg !103
  store i8* %1, i8** %4, align 8
  call void @llvm.dbg.declare(metadata i8** %4, metadata !104, metadata !{!"0x102"}), !dbg !103
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
  call void @llvm.dbg.declare(metadata i8** %2, metadata !105, metadata !{!"0x102"}), !dbg !106
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
  call void @llvm.dbg.declare(metadata %0** %a, metadata !107, metadata !{!"0x102"}), !dbg !108
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

!0 = !{!"0x11\0016\00clang version 3.3 \000\00\002\00\001", !1, !2, !3, !12, !2,  !2} ; [ DW_TAG_compile_unit ] [llvm/tools/clang/test/CodeGenObjC/<unknown>] [DW_LANG_ObjC]
!1 = !{!"llvm/tools/clang/test/CodeGenObjC/<unknown>", !"llvm/_build.ninja.Debug"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x13\00A\0033\0032\0032\000\00512\0016", !5, !6, null, !7, null, null, null} ; [ DW_TAG_structure_type ] [A] [line 33, size 32, align 32, offset 0] [def] [from ]
!5 = !{!"llvm/tools/clang/test/CodeGenObjC/debug-info-blocks.m", !"llvm/_build.ninja.Debug"}
!6 = !{!"0x29", !5}          ; [ DW_TAG_file_type ] [llvm/tools/clang/test/CodeGenObjC/debug-info-blocks.m]
!7 = !{!8, !10}
!8 = !{!"0x1c\00\000\000\000\000\000", null, !4, !9} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [from NSObject]
!9 = !{!"0x13\00NSObject\0021\000\008\000\000\0016", !5, !6, null, !2, null, null, null} ; [ DW_TAG_structure_type ] [NSObject] [line 21, size 0, align 8, offset 0] [def] [from ]
!10 = !{!"0xd\00ivar\0035\0032\0032\000\000", !5, !6, !11, null} ; [ DW_TAG_member ] [ivar] [line 35, size 32, align 32, offset 0] [from int]
!11 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = !{!13, !27, !31, !35, !36, !39}
!13 = !{!"0x2e\00-[A init]\00-[A init]\00\0046\001\001\000\006\00256\000\0046", !5, !6, !14, null, i8* (%0*, i8*)* @"\01-[A init]", null, null, !2} ; [ DW_TAG_subprogram ] [line 46] [local] [def] [-[A init]]
!14 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !15, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!15 = !{!16, !23, !24}
!16 = !{!"0x16\00id\0046\000\000\000\000", !5, null, !17} ; [ DW_TAG_typedef ] [id] [line 46, size 0, align 0, offset 0] [from ]
!17 = !{!"0xf\00\000\0064\0064\000\000", null, null, !18} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from objc_object]
!18 = !{!"0x13\00objc_object\000\000\000\000\000\000", !1, null, null, !19, null, null, null} ; [ DW_TAG_structure_type ] [objc_object] [line 0, size 0, align 0, offset 0] [def] [from ]
!19 = !{!20}
!20 = !{!"0xd\00isa\000\0064\000\000\000", !1, !18, !21} ; [ DW_TAG_member ] [isa] [line 0, size 64, align 0, offset 0] [from ]
!21 = !{!"0xf\00\000\0064\000\000\000", null, null, !22} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from objc_class]
!22 = !{!"0x13\00objc_class\000\000\000\000\004\000", !1, null, null, null, null, null, null} ; [ DW_TAG_structure_type ] [objc_class] [line 0, size 0, align 0, offset 0] [decl] [from ]
!23 = !{!"0xf\00\000\0064\0064\000\001088", i32 0, null, !4} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from A]
!24 = !{!"0x16\00SEL\0046\000\000\000\0064", !5, null, !25} ; [ DW_TAG_typedef ] [SEL] [line 46, size 0, align 0, offset 0] [artificial] [from ]
!25 = !{!"0xf\00\000\0064\0064\000\000", null, null, !26} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from objc_selector]
!26 = !{!"0x13\00objc_selector\000\000\000\000\004\000", !1, null, null, null, null, null, null} ; [ DW_TAG_structure_type ] [objc_selector] [line 0, size 0, align 0, offset 0] [decl] [from ]
!27 = !{!"0x2e\00__9-[A init]_block_invoke\00__9-[A init]_block_invoke\00\0049\001\001\000\006\00256\000\0049", !5, !6, !28, null, void (i8*)* @"__9-[A init]_block_invoke", null, null, !2} ; [ DW_TAG_subprogram ] [line 49] [local] [def] [__9-[A init]_block_invoke]
!28 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !29, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!29 = !{null, !30}
!30 = !{!"0xf\00\000\0064\0064\000\000", null, null, null} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!31 = !{!"0x2e\00__copy_helper_block_\00__copy_helper_block_\00\0052\001\001\000\006\000\000\0052", !1, !32, !33, null, void (i8*, i8*)* @__copy_helper_block_, null, null, !2} ; [ DW_TAG_subprogram ] [line 52] [local] [def] [__copy_helper_block_]
!32 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [llvm/tools/clang/test/CodeGenObjC/<unknown>]
!33 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !34, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!34 = !{null, !30, !30}
!35 = !{!"0x2e\00__destroy_helper_block_\00__destroy_helper_block_\00\0052\001\001\000\006\000\000\0052", !1, !32, !28, null, void (i8*)* @__destroy_helper_block_, null, null, !2} ; [ DW_TAG_subprogram ] [line 52] [local] [def] [__destroy_helper_block_]
!36 = !{!"0x2e\00main\00main\00\0059\000\001\000\006\000\000\0060", !5, !6, !37, null, i32 ()* @main, null, null, !2} ; [ DW_TAG_subprogram ] [line 59] [def] [scope 60] [main]
!37 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !38, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!38 = !{!11}
!39 = !{!"0x2e\00run\00run\00\0039\001\001\000\006\00256\000\0040", !5, !6, !40, null, void (void ()*)* @run, null, null, !2} ; [ DW_TAG_subprogram ] [line 39] [local] [def] [scope 40] [run]
!40 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !41, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!41 = !{null, !42}
!42 = !{!"0xf\00\000\0064\000\000\000", null, null, !43} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __block_literal_generic]
!43 = !{!"0x13\00__block_literal_generic\0040\00256\000\000\008\000", !5, !6, null, !44, null, null, null} ; [ DW_TAG_structure_type ] [__block_literal_generic] [line 40, size 256, align 0, offset 0] [def] [from ]
!44 = !{!45, !46, !47, !48, !49}
!45 = !{!"0xd\00__isa\000\0064\0064\000\000", !5, !6, !30} ; [ DW_TAG_member ] [__isa] [line 0, size 64, align 64, offset 0] [from ]
!46 = !{!"0xd\00__flags\000\0032\0032\0064\000", !5, !6, !11} ; [ DW_TAG_member ] [__flags] [line 0, size 32, align 32, offset 64] [from int]
!47 = !{!"0xd\00__reserved\000\0032\0032\0096\000", !5, !6, !11} ; [ DW_TAG_member ] [__reserved] [line 0, size 32, align 32, offset 96] [from int]
!48 = !{!"0xd\00__FuncPtr\000\0064\0064\00128\000", !5, !6, !30} ; [ DW_TAG_member ] [__FuncPtr] [line 0, size 64, align 64, offset 128] [from ]
!49 = !{!"0xd\00__descriptor\0040\0064\0064\00192\000", !5, !6, !50} ; [ DW_TAG_member ] [__descriptor] [line 40, size 64, align 64, offset 192] [from ]
!50 = !{!"0xf\00\000\0064\000\000\000", null, null, !51} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __block_descriptor]
!51 = !{!"0x13\00__block_descriptor\0040\00128\000\000\008\000", !5, !6, null, !52, null, null, null} ; [ DW_TAG_structure_type ] [__block_descriptor] [line 40, size 128, align 0, offset 0] [def] [from ]
!52 = !{!53, !55}
!53 = !{!"0xd\00reserved\000\0064\0064\000\000", !5, !6, !54} ; [ DW_TAG_member ] [reserved] [line 0, size 64, align 64, offset 0] [from long unsigned int]
!54 = !{!"0x24\00long unsigned int\000\0064\0064\000\000\007", null, null} ; [ DW_TAG_base_type ] [long unsigned int] [line 0, size 64, align 64, offset 0, enc DW_ATE_unsigned]
!55 = !{!"0xd\00Size\000\0064\0064\0064\000", !5, !6, !54} ; [ DW_TAG_member ] [Size] [line 0, size 64, align 64, offset 64] [from long unsigned int]
!56 = !{i32 1, !"Objective-C Version", i32 2}
!57 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!58 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!59 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
!60 = !{!"0x101\00self\0016777262\001088", !13, !32, !61} ; [ DW_TAG_arg_variable ] [self] [line 46]
!61 = !{!"0xf\00\000\0064\0064\000\000", null, null, !4} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from A]
!62 = !MDLocation(line: 46, scope: !13)
!63 = !{!"0x101\00_cmd\0033554478\0064", !13, !32, !64} ; [ DW_TAG_arg_variable ] [_cmd] [line 46]
!64 = !{!"0x16\00SEL\0046\000\000\000\000", !5, null, !25} ; [ DW_TAG_typedef ] [SEL] [line 46, size 0, align 0, offset 0] [from ]
!65 = !MDLocation(line: 48, scope: !66)
!66 = !{!"0xb\0047\000\000", !5, !13} ; [ DW_TAG_lexical_block ] [llvm/tools/clang/test/CodeGenObjC/debug-info-blocks.m]
!67 = !{}
!68 = !MDLocation(line: 49, scope: !69)
!69 = !{!"0xb\0048\000\001", !5, !66} ; [ DW_TAG_lexical_block ] [llvm/tools/clang/test/CodeGenObjC/debug-info-blocks.m]
!70 = !MDLocation(line: 53, scope: !69)
!71 = !MDLocation(line: 54, scope: !66)
!72 = !{!"0x101\00block\0016777255\000", !39, !6, !42} ; [ DW_TAG_arg_variable ] [block] [line 39]
!73 = !MDLocation(line: 39, scope: !39)
!74 = !MDLocation(line: 41, scope: !39)
!75 = !MDLocation(line: 42, scope: !39)
!76 = !{!"0x101\00.block_descriptor\0016777265\0064", !27, !6, !77} ; [ DW_TAG_arg_variable ] [.block_descriptor] [line 49]
!77 = !{!"0xf\00\000\0064\000\000\000", null, null, !78} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __block_literal_1]
!78 = !{!"0x13\00__block_literal_1\0049\00320\0064\000\000\000", !5, !6, null, !79, null, null, null} ; [ DW_TAG_structure_type ] [__block_literal_1] [line 49, size 320, align 64, offset 0] [def] [from ]
!79 = !{!80, !81, !82, !83, !84, !87}
!80 = !{!"0xd\00__isa\0049\0064\0064\000\000", !5, !6, !30} ; [ DW_TAG_member ] [__isa] [line 49, size 64, align 64, offset 0] [from ]
!81 = !{!"0xd\00__flags\0049\0032\0032\0064\000", !5, !6, !11} ; [ DW_TAG_member ] [__flags] [line 49, size 32, align 32, offset 64] [from int]
!82 = !{!"0xd\00__reserved\0049\0032\0032\0096\000", !5, !6, !11} ; [ DW_TAG_member ] [__reserved] [line 49, size 32, align 32, offset 96] [from int]
!83 = !{!"0xd\00__FuncPtr\0049\0064\0064\00128\000", !5, !6, !30} ; [ DW_TAG_member ] [__FuncPtr] [line 49, size 64, align 64, offset 128] [from ]
!84 = !{!"0xd\00__descriptor\0049\0064\0064\00192\000", !5, !6, !85} ; [ DW_TAG_member ] [__descriptor] [line 49, size 64, align 64, offset 192] [from ]
!85 = !{!"0xf\00\000\0064\0064\000\000", null, null, !86} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from __block_descriptor_withcopydispose]
!86 = !{!"0x13\00__block_descriptor_withcopydispose\0049\000\000\000\004\000", !1, null, null, null, null, null, null} ; [ DW_TAG_structure_type ] [__block_descriptor_withcopydispose] [line 49, size 0, align 0, offset 0] [decl] [from ]
!87 = !{!"0xd\00self\0049\0064\0064\00256\000", !5, !6, !61} ; [ DW_TAG_member ] [self] [line 49, size 64, align 64, offset 256] [from ]
!88 = !MDLocation(line: 49, scope: !27)
!89 = !{!"0x100\00self\0052\000", !27, !32, !23} ; [ DW_TAG_auto_variable ] [self] [line 52]
!90 = !MDLocation(line: 52, scope: !27)
!91 = !{!"0x100\00d\0050\000", !92, !6, !93} ; [ DW_TAG_auto_variable ] [d] [line 50]
!92 = !{!"0xb\0049\000\002", !5, !27} ; [ DW_TAG_lexical_block ] [llvm/tools/clang/test/CodeGenObjC/debug-info-blocks.m]
!93 = !{!"0xf\00\000\0064\0064\000\000", null, null, !94} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from NSMutableDictionary]
!94 = !{!"0x13\00NSMutableDictionary\0030\000\008\000\000\0016", !5, !6, null, !95, null, null, null} ; [ DW_TAG_structure_type ] [NSMutableDictionary] [line 30, size 0, align 8, offset 0] [def] [from ]
!95 = !{!96}
!96 = !{!"0x1c\00\000\000\000\000\000", null, !94, !97} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [from NSDictionary]
!97 = !{!"0x13\00NSDictionary\0026\000\008\000\000\0016", !5, !6, null, !98, null, null, null} ; [ DW_TAG_structure_type ] [NSDictionary] [line 26, size 0, align 8, offset 0] [def] [from ]
!98 = !{!99}
!99 = !{!"0x1c\00\000\000\000\000\000", null, !97, !9} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [from NSObject]
!100 = !MDLocation(line: 50, scope: !92)
!101 = !MDLocation(line: 51, scope: !92)
!102 = !{!"0x101\00\0016777268\001088", !31, !32, !30} ; [ DW_TAG_arg_variable ] [line 52]
!103 = !MDLocation(line: 52, scope: !31)
!104 = !{!"0x101\00\0033554484\0064", !31, !32, !30} ; [ DW_TAG_arg_variable ] [line 52]
!105 = !{!"0x101\00\0016777268\001088", !35, !32, !30} ; [ DW_TAG_arg_variable ] [line 52]
!106 = !MDLocation(line: 52, scope: !35)
!107 = !{!"0x100\00a\0061\000", !36, !6, !61} ; [ DW_TAG_auto_variable ] [a] [line 61]
!108 = !MDLocation(line: 61, scope: !36)
!109 = !MDLocation(line: 62, scope: !36)
!110 = !{i32 1, !"Debug Info Version", i32 2}
!111 = !{!"0x102\006\0034\0032"} ; [ DW_TAG_expression ] [DW_OP_deref DW_OP_plus 32]

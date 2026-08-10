; RUN: llvm-as %s -o - | llvm-dis - | FileCheck %s

%struct._class_t = type { %struct._class_t*, %struct._class_t*, %struct._objc_cache*, i8* (i8*, i8*)**, %struct._class_ro_t* }
%struct._objc_cache = type opaque
%struct._class_ro_t = type { i32, i32, i32, i8*, i8*, %struct.__method_list_t*, %struct._objc_protocol_list*, %struct._ivar_list_t*, i8*, %struct._prop_list_t* }
%struct.__method_list_t = type { i32, i32, [0 x %struct._objc_method] }
%struct._objc_method = type { i8*, i8*, i8* }
%struct._objc_protocol_list = type { i64, [0 x %struct._protocol_t*] }
%struct._protocol_t = type { i8*, i8*, %struct._objc_protocol_list*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct._prop_list_t*, i32, i32, i8**, i8*, %struct._prop_list_t* }
%struct._ivar_list_t = type { i32, i32, [0 x %struct._ivar_t] }
%struct._ivar_t = type { i64*, i8*, i8*, i32, i32 }
%struct._prop_list_t = type { i32, i32, [0 x %struct._prop_t] }
%struct._prop_t = type { i8*, i8* }
%struct._category_t = type { i8*, %struct._class_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct._objc_protocol_list*, %struct._prop_list_t*, %struct._prop_list_t*, i32 }

@OBJC_CLASS_NAME_ = private unnamed_addr constant [6 x i8] c"Robot\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@"OBJC_CLASS_$_I" = external global %struct._class_t
@"\01l_OBJC_$_CATEGORY_I_$_Robot" = private global %struct._category_t { i8* getelementptr inbounds ([6 x i8], [6 x i8]* @OBJC_CLASS_NAME_, i32 0, i32 0), %struct._class_t* @"OBJC_CLASS_$_I", %struct.__method_list_t* null, %struct.__method_list_t* null, %struct._objc_protocol_list* null, %struct._prop_list_t* null, %struct._prop_list_t* null, i32 64 }, section "__DATA, __objc_const", align 8
@"OBJC_LABEL_CATEGORY_$" = private global [1 x i8*] [i8* bitcast (%struct._category_t* @"\01l_OBJC_$_CATEGORY_I_$_Robot" to i8*)], section "__DATA, __objc_catlist, regular, no_dead_strip", align 8
@llvm.compiler.used = appending global [3 x i8*] [i8* bitcast (%struct._category_t* @"\01l_OBJC_$_CATEGORY_I_$_Robot" to i8*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @OBJC_CLASS_NAME_, i32 0, i32 0), i8* bitcast ([1 x i8*]* @"OBJC_LABEL_CATEGORY_$" to i8*)], section "llvm.metadata"

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5}

!0 = !{i32 1, !"Objective-C Version", i32 2}
!1 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!2 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!3 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
!4 = !{i32 1, !"Objective-C Class Properties", i32 64}
!5 = !{i32 1, !"PIC Level", i32 2}

; CHECK: @"OBJC_LABEL_CATEGORY_$" = {{.*}}, section "__DATA,__objc_catlist,regular,no_dead_strip"

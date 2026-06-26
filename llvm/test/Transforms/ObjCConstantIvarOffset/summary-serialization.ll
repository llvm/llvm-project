; Test that ObjC class hierarchy info is correctly serialized into
; per-module summary bitcode (FS_OBJC_CLASS_INFO records).
;
; Hierarchy:
;   NSObject
;     `- Super
;          int x;                        // x at offset 8
;          int hidden;                   // hidden at offset 12 (extension)
;
; RUN: rm -rf %t && split-file %s %t
; RUN: opt -module-summary %t/body.ll -o %t/out.bc
; RUN: llvm-bcanalyzer --dump %t/out.bc | FileCheck %s
;
; Check that the class info record contains correct values:
; op2=instanceStart(8), op3=instanceSize(16), op4=maxIvarAlignment(4)
; CHECK: <OBJC_CLASS_INFO op0={{[0-9]+}} op1={{[0-9]+}} op2=8 op3=16 op4=4/>

; Segments:
;   source.m - ingredients (requires Darwin SDK to regenerate)
;   gen      - refresher
;   body.ll  - single module used by both FullLTO and ThinLTO paths

;--- source.m
#import <objc/NSObject.h>
@interface Super : NSObject { int x; } @end
@interface Super () { int hidden; } @end
@implementation Super @end

;--- gen
clang -target x86_64-apple-macosx10.15 -fobjc-runtime=macosx-10.15 -S -emit-llvm -isysroot $(xcrun --show-sdk-path) source.m -o -
# body.ll is auto-generated from source.m
;--- body.ll
; ModuleID = 'source.m'
source_filename = "source.m"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

%struct._objc_cache = type opaque
%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }
%struct._class_ro_t = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct._ivar_t = type { ptr, ptr, ptr, i32, i32 }

@_objc_empty_cache = external global %struct._objc_cache
@"OBJC_METACLASS_$_NSObject" = external global %struct._class_t
@OBJC_CLASS_NAME_ = private unnamed_addr constant [6 x i8] c"Super\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@"_OBJC_METACLASS_RO_$_Super" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @OBJC_CLASS_NAME_, ptr null, ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
@"OBJC_METACLASS_$_Super" = global %struct._class_t { ptr @"OBJC_METACLASS_$_NSObject", ptr @"OBJC_METACLASS_$_NSObject", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_Super" }, section "__DATA, __objc_data", align 8
@"OBJC_CLASS_$_NSObject" = external global %struct._class_t
@"OBJC_IVAR_$_Super.x" = constant i64 8, section "__DATA, __objc_ivar", align 8
@OBJC_METH_VAR_NAME_ = private unnamed_addr constant [2 x i8] c"x\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_METH_VAR_TYPE_ = private unnamed_addr constant [2 x i8] c"i\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@"OBJC_IVAR_$_Super.hidden" = hidden constant i64 12, section "__DATA, __objc_ivar", align 8
@OBJC_METH_VAR_NAME_.1 = private unnamed_addr constant [7 x i8] c"hidden\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"_OBJC_$_INSTANCE_VARIABLES_Super" = internal global { i32, i32, [2 x %struct._ivar_t] } { i32 32, i32 2, [2 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_Super.x", ptr @OBJC_METH_VAR_NAME_, ptr @OBJC_METH_VAR_TYPE_, i32 2, i32 4 }, %struct._ivar_t { ptr @"OBJC_IVAR_$_Super.hidden", ptr @OBJC_METH_VAR_NAME_.1, ptr @OBJC_METH_VAR_TYPE_, i32 2, i32 4 }] }, section "__DATA, __objc_const", align 8
@"_OBJC_CLASS_RO_$_Super" = internal global %struct._class_ro_t { i32 0, i32 8, i32 16, ptr null, ptr @OBJC_CLASS_NAME_, ptr null, ptr null, ptr @"_OBJC_$_INSTANCE_VARIABLES_Super", ptr null, ptr null }, section "__DATA, __objc_const", align 8
@"OBJC_CLASS_$_Super" = global %struct._class_t { ptr @"OBJC_METACLASS_$_Super", ptr @"OBJC_CLASS_$_NSObject", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_Super" }, section "__DATA, __objc_data", align 8
@"OBJC_LABEL_CLASS_$" = private global [1 x ptr] [ptr @"OBJC_CLASS_$_Super"], section "__DATA,__objc_classlist,regular,no_dead_strip", align 8
@llvm.compiler.used = appending global [6 x ptr] [ptr @OBJC_CLASS_NAME_, ptr @OBJC_METH_VAR_NAME_, ptr @OBJC_METH_VAR_TYPE_, ptr @OBJC_METH_VAR_NAME_.1, ptr @"_OBJC_$_INSTANCE_VARIABLES_Super", ptr @"OBJC_LABEL_CLASS_$"], section "llvm.metadata"

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 2]}
!1 = !{i32 1, !"Objective-C Version", i32 2}
!2 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!3 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!4 = !{i32 1, !"Objective-C Garbage Collection", i8 0}
!5 = !{i32 1, !"Objective-C Class Properties", i32 64}
!6 = !{i32 1, !"Objective-C Enforce ClassRO Pointer Signing", i8 0}
!7 = !{i32 8, !"PIC Level", i32 2}
!8 = !{i32 7, !"uwtable", i32 2}
!9 = !{i32 7, !"frame-pointer", i32 2}

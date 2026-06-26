; Test that all ivars in a complete single-module hierarchy are
; promoted to constant without sliding.
;
; Hierarchy:
;   NSObject
;     `- SuperClass
;          @property int x;          // _x at offset 8
;          `- SubClass
;               @property int y;     // _y at offset 12
;
;
; --- FullLTO path ---
; RUN: rm -rf %t && split-file %s %t
; RUN: opt -passes=objc-constant-ivar-offset -S %t/body.ll | FileCheck %s --check-prefixes=SUPER,SUB
;
; SUPER: @"OBJC_IVAR_$_SuperClass._x" = {{.*}}constant i64 8
; SUB: @"OBJC_IVAR_$_SubClass._y" = {{.*}}constant i64 12
;
; --- ThinLTO path ---
; RUN: opt -module-summary %t/body.ll -o %t/body.bc
; RUN: llvm-lto2 run %t/body.bc -save-temps -o %t/out \
; RUN:   -r=%t/body.bc,_OBJC_CLASS_$_SuperClass,plx \
; RUN:   -r=%t/body.bc,_OBJC_CLASS_$_SubClass,plx \
; RUN:   -r=%t/body.bc,_OBJC_CLASS_$_NSObject, \
; RUN:   -r=%t/body.bc,_OBJC_METACLASS_$_SuperClass,plx \
; RUN:   -r=%t/body.bc,_OBJC_METACLASS_$_SubClass,plx \
; RUN:   -r=%t/body.bc,_OBJC_METACLASS_$_NSObject, \
; RUN:   -r=%t/body.bc,_OBJC_IVAR_$_SuperClass._x,plx \
; RUN:   -r=%t/body.bc,_OBJC_IVAR_$_SubClass._y,plx \
; RUN:   -r=%t/body.bc,__objc_empty_cache,
; RUN: llvm-dis %t/out.1.4.opt.bc -o - | FileCheck %s --check-prefixes=SUPER,SUB

; Segments:
;   source.m - ingredients (requires Darwin SDK to regenerate)
;   gen      - refresher
;   body.ll  - single module used by both FullLTO and ThinLTO paths

;--- source.m
#import <objc/NSObject.h>
@interface SuperClass : NSObject
@property int x;
@end
@interface SubClass : SuperClass
@property int y;
@end
@implementation SuperClass @end
@implementation SubClass @end

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
%struct._objc_method = type { ptr, ptr, ptr }
%struct._ivar_t = type { ptr, ptr, ptr, i32, i32 }
%struct._prop_t = type { ptr, ptr }

@_objc_empty_cache = external global %struct._objc_cache
@"OBJC_METACLASS_$_NSObject" = external global %struct._class_t
@OBJC_CLASS_NAME_ = private unnamed_addr constant [11 x i8] c"SuperClass\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@"_OBJC_METACLASS_RO_$_SuperClass" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @OBJC_CLASS_NAME_, ptr null, ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
@"OBJC_METACLASS_$_SuperClass" = global %struct._class_t { ptr @"OBJC_METACLASS_$_NSObject", ptr @"OBJC_METACLASS_$_NSObject", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_SuperClass" }, section "__DATA, __objc_data", align 8
@"OBJC_CLASS_$_NSObject" = external global %struct._class_t
@OBJC_METH_VAR_NAME_ = private unnamed_addr constant [2 x i8] c"x\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_METH_VAR_TYPE_ = private unnamed_addr constant [8 x i8] c"i16@0:8\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@OBJC_METH_VAR_NAME_.1 = private unnamed_addr constant [6 x i8] c"setX:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_METH_VAR_TYPE_.2 = private unnamed_addr constant [11 x i8] c"v20@0:8i16\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@"_OBJC_$_INSTANCE_METHODS_SuperClass" = internal global { i32, i32, [2 x %struct._objc_method] } { i32 24, i32 2, [2 x %struct._objc_method] [%struct._objc_method { ptr @OBJC_METH_VAR_NAME_, ptr @OBJC_METH_VAR_TYPE_, ptr @"\01-[SuperClass x]" }, %struct._objc_method { ptr @OBJC_METH_VAR_NAME_.1, ptr @OBJC_METH_VAR_TYPE_.2, ptr @"\01-[SuperClass setX:]" }] }, section "__DATA, __objc_const", align 8
@"OBJC_IVAR_$_SuperClass._x" = hidden constant i64 8, section "__DATA, __objc_ivar", align 8
@OBJC_METH_VAR_NAME_.3 = private unnamed_addr constant [3 x i8] c"_x\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_METH_VAR_TYPE_.4 = private unnamed_addr constant [2 x i8] c"i\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@"_OBJC_$_INSTANCE_VARIABLES_SuperClass" = internal global { i32, i32, [1 x %struct._ivar_t] } { i32 32, i32 1, [1 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_SuperClass._x", ptr @OBJC_METH_VAR_NAME_.3, ptr @OBJC_METH_VAR_TYPE_.4, i32 2, i32 4 }] }, section "__DATA, __objc_const", align 8
@OBJC_PROP_NAME_ATTR_ = private unnamed_addr constant [2 x i8] c"x\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_PROP_NAME_ATTR_.5 = private unnamed_addr constant [7 x i8] c"Ti,V_x\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"_OBJC_$_PROP_LIST_SuperClass" = internal global { i32, i32, [1 x %struct._prop_t] } { i32 16, i32 1, [1 x %struct._prop_t] [%struct._prop_t { ptr @OBJC_PROP_NAME_ATTR_, ptr @OBJC_PROP_NAME_ATTR_.5 }] }, section "__DATA, __objc_const", align 8
@"_OBJC_CLASS_RO_$_SuperClass" = internal global %struct._class_ro_t { i32 0, i32 8, i32 12, ptr null, ptr @OBJC_CLASS_NAME_, ptr @"_OBJC_$_INSTANCE_METHODS_SuperClass", ptr null, ptr @"_OBJC_$_INSTANCE_VARIABLES_SuperClass", ptr null, ptr @"_OBJC_$_PROP_LIST_SuperClass" }, section "__DATA, __objc_const", align 8
@"OBJC_CLASS_$_SuperClass" = global %struct._class_t { ptr @"OBJC_METACLASS_$_SuperClass", ptr @"OBJC_CLASS_$_NSObject", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_SuperClass" }, section "__DATA, __objc_data", align 8
@OBJC_CLASS_NAME_.6 = private unnamed_addr constant [9 x i8] c"SubClass\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@"_OBJC_METACLASS_RO_$_SubClass" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @OBJC_CLASS_NAME_.6, ptr null, ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
@"OBJC_METACLASS_$_SubClass" = global %struct._class_t { ptr @"OBJC_METACLASS_$_NSObject", ptr @"OBJC_METACLASS_$_SuperClass", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_SubClass" }, section "__DATA, __objc_data", align 8
@OBJC_METH_VAR_NAME_.7 = private unnamed_addr constant [2 x i8] c"y\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_METH_VAR_NAME_.8 = private unnamed_addr constant [6 x i8] c"setY:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"_OBJC_$_INSTANCE_METHODS_SubClass" = internal global { i32, i32, [2 x %struct._objc_method] } { i32 24, i32 2, [2 x %struct._objc_method] [%struct._objc_method { ptr @OBJC_METH_VAR_NAME_.7, ptr @OBJC_METH_VAR_TYPE_, ptr @"\01-[SubClass y]" }, %struct._objc_method { ptr @OBJC_METH_VAR_NAME_.8, ptr @OBJC_METH_VAR_TYPE_.2, ptr @"\01-[SubClass setY:]" }] }, section "__DATA, __objc_const", align 8
@"OBJC_IVAR_$_SubClass._y" = hidden constant i64 12, section "__DATA, __objc_ivar", align 8
@OBJC_METH_VAR_NAME_.9 = private unnamed_addr constant [3 x i8] c"_y\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"_OBJC_$_INSTANCE_VARIABLES_SubClass" = internal global { i32, i32, [1 x %struct._ivar_t] } { i32 32, i32 1, [1 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_SubClass._y", ptr @OBJC_METH_VAR_NAME_.9, ptr @OBJC_METH_VAR_TYPE_.4, i32 2, i32 4 }] }, section "__DATA, __objc_const", align 8
@OBJC_PROP_NAME_ATTR_.10 = private unnamed_addr constant [2 x i8] c"y\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_PROP_NAME_ATTR_.11 = private unnamed_addr constant [7 x i8] c"Ti,V_y\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"_OBJC_$_PROP_LIST_SubClass" = internal global { i32, i32, [1 x %struct._prop_t] } { i32 16, i32 1, [1 x %struct._prop_t] [%struct._prop_t { ptr @OBJC_PROP_NAME_ATTR_.10, ptr @OBJC_PROP_NAME_ATTR_.11 }] }, section "__DATA, __objc_const", align 8
@"_OBJC_CLASS_RO_$_SubClass" = internal global %struct._class_ro_t { i32 0, i32 12, i32 16, ptr null, ptr @OBJC_CLASS_NAME_.6, ptr @"_OBJC_$_INSTANCE_METHODS_SubClass", ptr null, ptr @"_OBJC_$_INSTANCE_VARIABLES_SubClass", ptr null, ptr @"_OBJC_$_PROP_LIST_SubClass" }, section "__DATA, __objc_const", align 8
@"OBJC_CLASS_$_SubClass" = global %struct._class_t { ptr @"OBJC_METACLASS_$_SubClass", ptr @"OBJC_CLASS_$_SuperClass", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_SubClass" }, section "__DATA, __objc_data", align 8
@"OBJC_LABEL_CLASS_$" = private global [2 x ptr] [ptr @"OBJC_CLASS_$_SuperClass", ptr @"OBJC_CLASS_$_SubClass"], section "__DATA,__objc_classlist,regular,no_dead_strip", align 8
@llvm.compiler.used = appending global [22 x ptr] [ptr @OBJC_CLASS_NAME_, ptr @OBJC_METH_VAR_NAME_, ptr @OBJC_METH_VAR_TYPE_, ptr @OBJC_METH_VAR_NAME_.1, ptr @OBJC_METH_VAR_TYPE_.2, ptr @"_OBJC_$_INSTANCE_METHODS_SuperClass", ptr @OBJC_METH_VAR_NAME_.3, ptr @OBJC_METH_VAR_TYPE_.4, ptr @"_OBJC_$_INSTANCE_VARIABLES_SuperClass", ptr @OBJC_PROP_NAME_ATTR_, ptr @OBJC_PROP_NAME_ATTR_.5, ptr @"_OBJC_$_PROP_LIST_SuperClass", ptr @OBJC_CLASS_NAME_.6, ptr @OBJC_METH_VAR_NAME_.7, ptr @OBJC_METH_VAR_NAME_.8, ptr @"_OBJC_$_INSTANCE_METHODS_SubClass", ptr @OBJC_METH_VAR_NAME_.9, ptr @"_OBJC_$_INSTANCE_VARIABLES_SubClass", ptr @OBJC_PROP_NAME_ATTR_.10, ptr @OBJC_PROP_NAME_ATTR_.11, ptr @"_OBJC_$_PROP_LIST_SubClass", ptr @"OBJC_LABEL_CLASS_$"], section "llvm.metadata"

; Function Attrs: noinline optnone ssp uwtable
define internal i32 @"\01-[SuperClass x]"(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = getelementptr inbounds i8, ptr %5, i64 8
  %7 = load atomic i32, ptr %6 unordered, align 4
  ret i32 %7
}

; Function Attrs: noinline optnone ssp uwtable
define internal void @"\01-[SuperClass setX:]"(ptr noundef %0, ptr noundef %1, i32 noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store i32 %2, ptr %6, align 4
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr inbounds i8, ptr %7, i64 8
  %9 = load i32, ptr %6, align 4
  store atomic i32 %9, ptr %8 unordered, align 4
  ret void
}

; Function Attrs: noinline optnone ssp uwtable
define internal i32 @"\01-[SubClass y]"(ptr noundef %0, ptr noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = getelementptr inbounds i8, ptr %5, i64 12
  %7 = load atomic i32, ptr %6 unordered, align 4
  ret i32 %7
}

; Function Attrs: noinline optnone ssp uwtable
define internal void @"\01-[SubClass setY:]"(ptr noundef %0, ptr noundef %1, i32 noundef %2) #0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store i32 %2, ptr %6, align 4
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr inbounds i8, ptr %7, i64 12
  %9 = load i32, ptr %6, align 4
  store atomic i32 %9, ptr %8 unordered, align 4
  ret void
}

attributes #0 = { noinline optnone ssp uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cmov,+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }

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

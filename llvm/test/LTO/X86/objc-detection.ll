; RUN: llvm-as < %s -o %t
; RUN: llvm-lto -check-for-objc %t | FileCheck %s

; CHECK: contains ObjC

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%0 = type opaque
%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }
%struct._objc_cache = type opaque
%struct._class_ro_t = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct.__method_list_t = type { i32, i32, [0 x %struct._objc_method] }
%struct._objc_method = type { ptr, ptr, ptr }
%struct._objc_protocol_list = type { i64, [0 x ptr] }
%struct._protocol_t = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr }
%struct._ivar_list_t = type { i32, i32, [0 x %struct._ivar_t] }
%struct._ivar_t = type { ptr, ptr, ptr, i32, i32 }
%struct._prop_list_t = type { i32, i32, [0 x %struct._prop_t] }
%struct._prop_t = type { ptr, ptr }
%struct._category_t = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32 }

@OBJC_CLASS_NAME_ = private global [4 x i8] c"foo\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@"OBJC_CLASS_$_A" = external global %struct._class_t
@OBJC_METH_VAR_NAME_ = private global [12 x i8] c"foo_myStuff\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_METH_VAR_TYPE_ = private global [8 x i8] c"v16@0:8\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@"\01l_OBJC_$_CATEGORY_INSTANCE_METHODS_A_$_foo" = private global { i32, i32, [1 x %struct._objc_method] } { i32 24, i32 1, [1 x %struct._objc_method] [%struct._objc_method { ptr @OBJC_METH_VAR_NAME_, ptr @OBJC_METH_VAR_TYPE_, ptr @"\01-[A(foo) foo_myStuff]" }] }, section "__DATA, __objc_const", align 8
@"\01l_OBJC_$_CATEGORY_A_$_foo" = private global %struct._category_t { ptr @OBJC_CLASS_NAME_, ptr @"OBJC_CLASS_$_A", ptr @"\01l_OBJC_$_CATEGORY_INSTANCE_METHODS_A_$_foo", ptr null, ptr null, ptr null, ptr null, i32 64 }, section "__DATA, __objc_const", align 8
@"OBJC_LABEL_CATEGORY_$" = private global [1 x ptr] [ptr @"\01l_OBJC_$_CATEGORY_A_$_foo"], section "__DATA, __objc_catlist, regular, no_dead_strip", align 8
@llvm.compiler.used = appending global [6 x ptr] [ptr @OBJC_CLASS_NAME_, ptr @OBJC_METH_VAR_NAME_, ptr @OBJC_METH_VAR_TYPE_, ptr @"\01l_OBJC_$_CATEGORY_INSTANCE_METHODS_A_$_foo", ptr @"\01l_OBJC_$_CATEGORY_A_$_foo", ptr @"OBJC_LABEL_CATEGORY_$"], section "llvm.metadata"

; Function Attrs: ssp uwtable
define internal void @"\01-[A(foo) foo_myStuff]"(ptr, ptr) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  ret void
}

attributes #0 = { ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = !{i32 1, !"Objective-C Version", i32 2}
!1 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!2 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!3 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
!4 = !{i32 1, !"Objective-C Class Properties", i32 64}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"Apple LLVM version 8.0.0 (clang-800.0.24.1)"}

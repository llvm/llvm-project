; RUN: llvm-as < %s -o %t
; RUN: llvm-lto -check-for-objc %t | FileCheck %s

; CHECK: contains ObjC


target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.12.0"

module asm "\09.lazy_reference .objc_class_name_A"
module asm "\09.objc_category_name_A_foo=0"
module asm "\09.globl .objc_category_name_A_foo"

%0 = type opaque
%struct._objc_method = type { i8*, i8*, i8* }
%struct._objc_category = type { i8*, i8*, %struct._objc_method_list*, %struct._objc_method_list*, %struct._objc_protocol_list*, i32, %struct._prop_list_t*, %struct._prop_list_t* }
%struct._objc_method_list = type opaque
%struct._objc_protocol_list = type { %struct._objc_protocol_list*, i32, [0 x %struct._objc_protocol] }
%struct._objc_protocol = type { %struct._objc_protocol_extension*, i8*, %struct._objc_protocol_list*, %struct._objc_method_description_list*, %struct._objc_method_description_list* }
%struct._objc_protocol_extension = type { i32, %struct._objc_method_description_list*, %struct._objc_method_description_list*, %struct._prop_list_t*, i8**, %struct._prop_list_t* }
%struct._objc_method_description_list = type { i32, [0 x %struct._objc_method_description] }
%struct._objc_method_description = type { i8*, i8* }
%struct._prop_list_t = type { i32, i32, [0 x %struct._prop_t] }
%struct._prop_t = type { i8*, i8* }
%struct._objc_module = type { i32, i32, i8*, %struct._objc_symtab* }
%struct._objc_symtab = type { i32, i8*, i16, i16, [0 x i8*] }

@OBJC_METH_VAR_NAME_ = private global [12 x i8] c"foo_myStuff\00", section "__TEXT,__cstring,cstring_literals", align 1
@OBJC_METH_VAR_TYPE_ = private global [7 x i8] c"v8@0:4\00", section "__TEXT,__cstring,cstring_literals", align 1
@OBJC_CLASS_NAME_ = private global [4 x i8] c"foo\00", section "__TEXT,__cstring,cstring_literals", align 1
@OBJC_CLASS_NAME_.1 = private global [2 x i8] c"A\00", section "__TEXT,__cstring,cstring_literals", align 1
@OBJC_CATEGORY_INSTANCE_METHODS_A_foo = private global { i8*, i32, [1 x %struct._objc_method] } { i8* null, i32 1, [1 x %struct._objc_method] [%struct._objc_method { i8* getelementptr inbounds ([12 x i8], [12 x i8]* @OBJC_METH_VAR_NAME_, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @OBJC_METH_VAR_TYPE_, i32 0, i32 0), i8* bitcast (void (%0*, i8*)* @"\01-[A(foo) foo_myStuff]" to i8*) }] }, section "__OBJC,__cat_inst_meth,regular,no_dead_strip", align 4
@OBJC_CATEGORY_A_foo = private global %struct._objc_category { i8* getelementptr inbounds ([4 x i8], [4 x i8]* @OBJC_CLASS_NAME_, i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @OBJC_CLASS_NAME_.1, i32 0, i32 0), %struct._objc_method_list* bitcast ({ i8*, i32, [1 x %struct._objc_method] }* @OBJC_CATEGORY_INSTANCE_METHODS_A_foo to %struct._objc_method_list*), %struct._objc_method_list* null, %struct._objc_protocol_list* null, i32 32, %struct._prop_list_t* null, %struct._prop_list_t* null }, section "__OBJC,__category,regular,no_dead_strip", align 4
@OBJC_CLASS_NAME_.2 = private global [1 x i8] zeroinitializer, section "__TEXT,__cstring,cstring_literals", align 1
@OBJC_SYMBOLS = private global { i32, i8*, i16, i16, [1 x i8*] } { i32 0, i8* null, i16 0, i16 1, [1 x i8*] [i8* bitcast (%struct._objc_category* @OBJC_CATEGORY_A_foo to i8*)] }, section "__OBJC,__symbols,regular,no_dead_strip", align 4
@OBJC_MODULES = private global %struct._objc_module { i32 7, i32 16, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @OBJC_CLASS_NAME_.2, i32 0, i32 0), %struct._objc_symtab* bitcast ({ i32, i8*, i16, i16, [1 x i8*] }* @OBJC_SYMBOLS to %struct._objc_symtab*) }, section "__OBJC,__module_info,regular,no_dead_strip", align 4
@llvm.compiler.used = appending global [9 x i8*] [i8* getelementptr inbounds ([12 x i8], [12 x i8]* @OBJC_METH_VAR_NAME_, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @OBJC_METH_VAR_TYPE_, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @OBJC_CLASS_NAME_, i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @OBJC_CLASS_NAME_.1, i32 0, i32 0), i8* bitcast ({ i8*, i32, [1 x %struct._objc_method] }* @OBJC_CATEGORY_INSTANCE_METHODS_A_foo to i8*), i8* bitcast (%struct._objc_category* @OBJC_CATEGORY_A_foo to i8*), i8* getelementptr inbounds ([1 x i8], [1 x i8]* @OBJC_CLASS_NAME_.2, i32 0, i32 0), i8* bitcast ({ i32, i8*, i16, i16, [1 x i8*] }* @OBJC_SYMBOLS to i8*), i8* bitcast (%struct._objc_module* @OBJC_MODULES to i8*)], section "llvm.metadata"

; Function Attrs: nounwind ssp
define internal void @"\01-[A(foo) foo_myStuff]"(%0*, i8*) #0 {
  %3 = alloca %0*, align 4
  %4 = alloca i8*, align 4
  store %0* %0, %0** %3, align 4
  store i8* %1, i8** %4, align 4
  ret void
}

attributes #0 = { nounwind ssp "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = !{i32 1, !"Objective-C Version", i32 1}
!1 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!2 = !{i32 1, !"Objective-C Image Info Section", !"__OBJC, __image_info,regular"}
!3 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
!4 = !{i32 1, !"Objective-C Class Properties", i32 64}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"Apple LLVM version 8.0.0 (clang-800.0.24.1)"}

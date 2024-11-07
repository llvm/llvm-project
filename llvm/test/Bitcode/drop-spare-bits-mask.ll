; RUN: llvm-dis < %s.bc -o - | FileCheck  %s

; check that the spare_bits_mask field was dropped.
; CHECK-NOT: spare_bits_mask

; ModuleID = 't.ll'
source_filename = "t.ll"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

%T1t19EitherWithSpareBitsO = type <{ [8 x i8] }>
%objc_class = type { ptr, ptr, ptr, ptr, i64 }
%swift.opaque = type opaque
%swift.method_descriptor = type { i32, i32 }
%swift.enum_vwtable = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64, i32, i32, ptr, ptr, ptr }
%swift.type_descriptor = type opaque
%swift.type_metadata_record = type { i32 }
%swift.type = type { i64 }
%swift.metadata_response = type { ptr, i64 }

@"$s1t5RightAA19EitherWithSpareBitsOvp" = hidden global %T1t19EitherWithSpareBitsO zeroinitializer, align 8, !dbg !0
@"\01l_entry_point" = private constant { i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint (ptr @main to i64), i64 ptrtoint (ptr @"\01l_entry_point" to i64)) to i32), i32 0 }, section "__TEXT, __swift5_entry, regular, no_dead_strip", align 4
@"$sBoWV" = external global ptr, align 8
@"$s1t1CCMm" = hidden global %objc_class { ptr @"OBJC_METACLASS_$__TtCs12_SwiftObject", ptr @"OBJC_METACLASS_$__TtCs12_SwiftObject", ptr @_objc_empty_cache, ptr null, i64 ptrtoint (ptr @_METACLASS_DATA__TtC1t1C to i64) }, align 8
@"OBJC_CLASS_$__TtCs12_SwiftObject" = external global %objc_class, align 8
@_objc_empty_cache = external global %swift.opaque
@"OBJC_METACLASS_$__TtCs12_SwiftObject" = external global %objc_class, align 8
@.str.8._TtC1t1C = private unnamed_addr constant [9 x i8] c"_TtC1t1C\00"
@_METACLASS_DATA__TtC1t1C = internal constant { i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { i32 129, i32 40, i32 40, i32 0, ptr null, ptr @.str.8._TtC1t1C, ptr null, ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
@_DATA__TtC1t1C = internal constant { i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { i32 128, i32 16, i32 16, i32 0, ptr null, ptr @.str.8._TtC1t1C, ptr null, ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
@.str.1.t = private constant [2 x i8] c"t\00"
@"$s1tMXM" = linkonce_odr hidden constant <{ i32, i32, i32 }> <{ i32 0, i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @.str.1.t to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32 }>, ptr @"$s1tMXM", i32 0, i32 2) to i64)) to i32) }>, section "__TEXT,__constg_swiftt", align 4
@.str.1.C = private constant [2 x i8] c"C\00"
@"$s1t1CCMn" = hidden constant <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor }> <{ i32 -2147483568, i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s1tMXM" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor }>, ptr @"$s1t1CCMn", i32 0, i32 1) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @.str.1.C to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor }>, ptr @"$s1t1CCMn", i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s1t1CCMa" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor }>, ptr @"$s1t1CCMn", i32 0, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s1t1CCMF" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor }>, ptr @"$s1t1CCMn", i32 0, i32 4) to i64)) to i32), i32 0, i32 3, i32 11, i32 1, i32 0, i32 10, i32 10, i32 1, %swift.method_descriptor { i32 1, i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s1t1CCACycfC" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor }>, ptr @"$s1t1CCMn", i32 0, i32 13, i32 1) to i64)) to i32) } }>, section "__TEXT,__constg_swiftt", align 4
@"$s1t1CCMf" = internal global <{ ptr, ptr, ptr, i64, ptr, ptr, ptr, i64, i32, i32, i32, i16, i16, i32, i32, ptr, ptr, ptr }> <{ ptr null, ptr @"$s1t1CCfD", ptr @"$sBoWV", i64 ptrtoint (ptr @"$s1t1CCMm" to i64), ptr @"OBJC_CLASS_$__TtCs12_SwiftObject", ptr @_objc_empty_cache, ptr null, i64 add (i64 ptrtoint (ptr @_DATA__TtC1t1C to i64), i64 2), i32 2, i32 0, i32 16, i16 7, i16 0, i32 112, i32 24, ptr @"$s1t1CCMn", ptr null, ptr @"$s1t1CCACycfC" }>, align 8
@"symbolic _____ 1t1CC" = linkonce_odr hidden constant <{ i8, i32, i8 }> <{ i8 1, i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s1t1CCMn" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i8, i32, i8 }>, ptr @"symbolic _____ 1t1CC", i32 0, i32 1) to i64)) to i32), i8 0 }>, section "__TEXT,__swift5_typeref, regular", no_sanitize_address, align 2
@"$s1t1CCMF" = internal constant { i32, i32, i16, i16, i32 } { i32 trunc (i64 sub (i64 ptrtoint (ptr @"symbolic _____ 1t1CC" to i64), i64 ptrtoint (ptr @"$s1t1CCMF" to i64)) to i32), i32 0, i16 1, i16 12, i32 0 }, section "__TEXT,__swift5_fieldmd, regular", no_sanitize_address, align 4
@"$s1t19EitherWithSpareBitsOWV" = internal constant %swift.enum_vwtable { ptr @"$s1t19EitherWithSpareBitsOwCP", ptr @"$s1t19EitherWithSpareBitsOwxx", ptr @"$s1t19EitherWithSpareBitsOwcp", ptr @"$s1t19EitherWithSpareBitsOwca", ptr @__swift_memcpy8_8, ptr @"$s1t19EitherWithSpareBitsOwta", ptr @"$s1t19EitherWithSpareBitsOwet", ptr @"$s1t19EitherWithSpareBitsOwst", i64 8, i64 8, i32 2162695, i32 14, ptr @"$s1t19EitherWithSpareBitsOwug", ptr @"$s1t19EitherWithSpareBitsOwup", ptr @"$s1t19EitherWithSpareBitsOwui" }, align 8
@.str.19.EitherWithSpareBits = private constant [20 x i8] c"EitherWithSpareBits\00"
@"$s1t19EitherWithSpareBitsOMn" = hidden constant <{ i32, i32, i32, i32, i32, i32, i32 }> <{ i32 82, i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s1tMXM" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32 }>, ptr @"$s1t19EitherWithSpareBitsOMn", i32 0, i32 1) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @.str.19.EitherWithSpareBits to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32 }>, ptr @"$s1t19EitherWithSpareBitsOMn", i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s1t19EitherWithSpareBitsOMa" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32 }>, ptr @"$s1t19EitherWithSpareBitsOMn", i32 0, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s1t19EitherWithSpareBitsOMF" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32 }>, ptr @"$s1t19EitherWithSpareBitsOMn", i32 0, i32 4) to i64)) to i32), i32 2, i32 0 }>, section "__TEXT,__constg_swiftt", align 4
@"$s1t19EitherWithSpareBitsOMf" = internal constant <{ ptr, ptr, i64, ptr }> <{ ptr null, ptr @"$s1t19EitherWithSpareBitsOWV", i64 513, ptr @"$s1t19EitherWithSpareBitsOMn" }>, align 8
@"symbolic _____ 1t19EitherWithSpareBitsO" = linkonce_odr hidden constant <{ i8, i32, i8 }> <{ i8 1, i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s1t19EitherWithSpareBitsOMn" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i8, i32, i8 }>, ptr @"symbolic _____ 1t19EitherWithSpareBitsO", i32 0, i32 1) to i64)) to i32), i8 0 }>, section "__TEXT,__swift5_typeref, regular", no_sanitize_address, align 2
@"$s1t19EitherWithSpareBitsOMB" = internal constant { i32, i32, i32, i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint (ptr @"symbolic _____ 1t19EitherWithSpareBitsO" to i64), i64 ptrtoint (ptr @"$s1t19EitherWithSpareBitsOMB" to i64)) to i32), i32 8, i32 65544, i32 8, i32 14 }, section "__TEXT,__swift5_builtin, regular", no_sanitize_address, align 4
@"\01l__swift5_reflection_descriptor" = private constant { i32, i32, i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint (ptr @"symbolic _____ 1t19EitherWithSpareBitsO" to i64), i64 ptrtoint (ptr @"\01l__swift5_reflection_descriptor" to i64)) to i32), i32 196609, i32 458753, i32 240 }, section "__TEXT,__swift5_mpenum, regular", no_sanitize_address, align 4
@0 = private constant [5 x i8] c"Left\00", section "__TEXT,__swift5_reflstr, regular", no_sanitize_address
@"$ss5Int32VMn" = external global %swift.type_descriptor, align 4
@"got.$ss5Int32VMn" = private unnamed_addr constant ptr @"$ss5Int32VMn"
@"symbolic _____ s5Int32V" = linkonce_odr hidden constant <{ i8, i32, i8 }> <{ i8 2, i32 trunc (i64 sub (i64 ptrtoint (ptr @"got.$ss5Int32VMn" to i64), i64 ptrtoint (ptr getelementptr inbounds (<{ i8, i32, i8 }>, ptr @"symbolic _____ s5Int32V", i32 0, i32 1) to i64)) to i32), i8 0 }>, section "__TEXT,__swift5_typeref, regular", no_sanitize_address, align 2
@1 = private constant [6 x i8] c"Right\00", section "__TEXT,__swift5_reflstr, regular", no_sanitize_address
@"$s1t19EitherWithSpareBitsOMF" = internal constant { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint (ptr @"symbolic _____ 1t19EitherWithSpareBitsO" to i64), i64 ptrtoint (ptr @"$s1t19EitherWithSpareBitsOMF" to i64)) to i32), i32 0, i16 3, i16 12, i32 2, i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @"symbolic _____ 1t1CC" to i64), i64 ptrtoint (ptr getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32 }, ptr @"$s1t19EitherWithSpareBitsOMF", i32 0, i32 6) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @0 to i64), i64 ptrtoint (ptr getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32 }, ptr @"$s1t19EitherWithSpareBitsOMF", i32 0, i32 7) to i64)) to i32), i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @"symbolic _____ s5Int32V" to i64), i64 ptrtoint (ptr getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32 }, ptr @"$s1t19EitherWithSpareBitsOMF", i32 0, i32 9) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @1 to i64), i64 ptrtoint (ptr getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32 }, ptr @"$s1t19EitherWithSpareBitsOMF", i32 0, i32 10) to i64)) to i32) }, section "__TEXT,__swift5_fieldmd, regular", no_sanitize_address, align 4
@"$s1t1CCHn" = private constant %swift.type_metadata_record { i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s1t1CCMn" to i64), i64 ptrtoint (ptr @"$s1t1CCHn" to i64)) to i32) }, section "__TEXT, __swift5_types, regular", no_sanitize_address, align 4
@"$s1t19EitherWithSpareBitsOHn" = private constant %swift.type_metadata_record { i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s1t19EitherWithSpareBitsOMn" to i64), i64 ptrtoint (ptr @"$s1t19EitherWithSpareBitsOHn" to i64)) to i32) }, section "__TEXT, __swift5_types, regular", no_sanitize_address, align 4
@__swift_reflection_version = linkonce_odr hidden constant i16 3
@"objc_classes_$s1t1CCN" = internal global ptr @"$s1t1CCN", section "__DATA,__objc_classlist,regular,no_dead_strip", no_sanitize_address, align 8
@llvm.used = appending global [11 x ptr] [ptr @"$s1t5RightAA19EitherWithSpareBitsOvp", ptr @main, ptr @"\01l_entry_point", ptr @"$s1t1CCMF", ptr @"$s1t19EitherWithSpareBitsOMB", ptr @"\01l__swift5_reflection_descriptor", ptr @"$s1t19EitherWithSpareBitsOMF", ptr @"$s1t1CCHn", ptr @"$s1t19EitherWithSpareBitsOHn", ptr @__swift_reflection_version, ptr @"objc_classes_$s1t1CCN"], section "llvm.metadata"
@llvm.compiler.used = appending global [5 x ptr] [ptr @"$s1t1CCACycfCTq", ptr @"$s1t1CCMf", ptr @"$s1t1CCN", ptr @"$s1t19EitherWithSpareBitsOMf", ptr @"$s1t19EitherWithSpareBitsON"], section "llvm.metadata"

@"$s1t1CCACycfCTq" = hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor }>, ptr @"$s1t1CCMn", i32 0, i32 13)
@"$s1t1CCN" = hidden alias %swift.type, getelementptr inbounds (<{ ptr, ptr, ptr, i64, ptr, ptr, ptr, i64, i32, i32, i32, i16, i16, i32, i32, ptr, ptr, ptr }>, ptr @"$s1t1CCMf", i32 0, i32 3)
@"$s1t19EitherWithSpareBitsON" = hidden alias %swift.type, getelementptr inbounds (<{ ptr, ptr, i64, ptr }>, ptr @"$s1t19EitherWithSpareBitsOMf", i32 0, i32 2)

define i32 @main(i32 %0, ptr %1) #0 !dbg !72 {
entry:
  store i64 -9223372036854775776, ptr @"$s1t5RightAA19EitherWithSpareBitsOvp", align 8, !dbg !76
  ret i32 0, !dbg !76
}

define hidden swiftcc ptr @"$s1t1CCfd"(ptr swiftself %0) #0 !dbg !79 {
entry:
  %self.debug = alloca ptr, align 8
    #dbg_declare(ptr %self.debug, !84, !DIExpression(), !86)
  call void @llvm.memset.p0.i64(ptr align 8 %self.debug, i8 0, i64 8, i1 false)
  store ptr %0, ptr %self.debug, align 8, !dbg !87
  ret ptr %0, !dbg !87
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #1

define hidden swiftcc void @"$s1t1CCfD"(ptr swiftself %0) #0 !dbg !88 {
entry:
  %self.debug = alloca ptr, align 8
    #dbg_declare(ptr %self.debug, !93, !DIExpression(), !94)
  call void @llvm.memset.p0.i64(ptr align 8 %self.debug, i8 0, i64 8, i1 false)
  store ptr %0, ptr %self.debug, align 8, !dbg !95
  %1 = call swiftcc ptr @"$s1t1CCfd"(ptr swiftself %0), !dbg !95
  call void @swift_deallocClassInstance(ptr %1, i64 16, i64 7) #3, !dbg !95
  ret void, !dbg !95
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #2

; Function Attrs: nounwind
declare void @swift_deallocClassInstance(ptr, i64, i64) #3

define hidden swiftcc ptr @"$s1t1CCACycfC"(ptr swiftself %0) #0 !dbg !96 {
entry:
  %1 = call noalias ptr @swift_allocObject(ptr %0, i64 16, i64 7) #3, !dbg !101
  %2 = call swiftcc ptr @"$s1t1CCACycfc"(ptr swiftself %1), !dbg !101
  ret ptr %2, !dbg !101
}

; Function Attrs: nounwind
declare ptr @swift_allocObject(ptr, i64, i64) #3

define hidden swiftcc ptr @"$s1t1CCACycfc"(ptr swiftself %0) #0 !dbg !102 {
entry:
  %self.debug = alloca ptr, align 8
    #dbg_declare(ptr %self.debug, !107, !DIExpression(), !108)
  call void @llvm.memset.p0.i64(ptr align 8 %self.debug, i8 0, i64 8, i1 false)
  store ptr %0, ptr %self.debug, align 8, !dbg !109
  ret ptr %0, !dbg !109
}

; Function Attrs: noinline nounwind memory(none)
define hidden swiftcc %swift.metadata_response @"$s1t1CCMa"(i64 %0) #4 !dbg !110 {
entry:
  %1 = call ptr @objc_opt_self(ptr getelementptr inbounds (<{ ptr, ptr, ptr, i64, ptr, ptr, ptr, i64, i32, i32, i32, i16, i16, i32, i32, ptr, ptr, ptr }>, ptr @"$s1t1CCMf", i32 0, i32 3)) #3, !dbg !113
  %2 = insertvalue %swift.metadata_response undef, ptr %1, 0, !dbg !113
  %3 = insertvalue %swift.metadata_response %2, i64 0, 1, !dbg !113
  ret %swift.metadata_response %3, !dbg !113
}

; Function Attrs: nounwind
declare ptr @objc_opt_self(ptr) #3

; Function Attrs: nounwind
define internal ptr @"$s1t19EitherWithSpareBitsOwCP"(ptr noalias %dest, ptr noalias %src, ptr %EitherWithSpareBits) #5 !dbg !114 {
entry:
  %0 = load i64, ptr %src, align 8, !dbg !115
  %1 = lshr i64 %0, 63, !dbg !115
  %2 = trunc i64 %1 to i1, !dbg !115
  call void @"$s1t19EitherWithSpareBitsOWOy"(i64 %0), !dbg !115
  store i64 %0, ptr %dest, align 8, !dbg !115
  ret ptr %dest, !dbg !115
}

; Function Attrs: noinline nounwind
define linkonce_odr hidden void @"$s1t19EitherWithSpareBitsOWOy"(i64 %0) #6 !dbg !116 {
entry:
  %1 = lshr i64 %0, 63, !dbg !117
  %2 = trunc i64 %1 to i1, !dbg !117
  br i1 %2, label %6, label %3

3:                                                ; preds = %entry
  %4 = inttoptr i64 %0 to ptr, !dbg !117
  %5 = call ptr @swift_retain(ptr returned %4) #7, !dbg !117
  br label %6, !dbg !117

6:                                                ; preds = %3, %entry
  ret void, !dbg !117
}

; Function Attrs: nounwind willreturn
declare ptr @swift_retain(ptr returned) #7

; Function Attrs: nounwind
define internal void @"$s1t19EitherWithSpareBitsOwxx"(ptr noalias %object, ptr %EitherWithSpareBits) #5 !dbg !118 {
entry:
  %0 = load i64, ptr %object, align 8, !dbg !119
  %1 = lshr i64 %0, 63, !dbg !119
  %2 = trunc i64 %1 to i1, !dbg !119
  call void @"$s1t19EitherWithSpareBitsOWOe"(i64 %0), !dbg !119
  ret void, !dbg !119
}

; Function Attrs: noinline nounwind
define linkonce_odr hidden void @"$s1t19EitherWithSpareBitsOWOe"(i64 %0) #6 !dbg !120 {
entry:
  %1 = lshr i64 %0, 63, !dbg !121
  %2 = trunc i64 %1 to i1, !dbg !121
  br i1 %2, label %5, label %3

3:                                                ; preds = %entry
  %4 = inttoptr i64 %0 to ptr, !dbg !121
  call void @swift_release(ptr %4) #3, !dbg !121
  br label %5, !dbg !121

5:                                                ; preds = %3, %entry
  ret void, !dbg !121
}

; Function Attrs: nounwind
declare void @swift_release(ptr) #3

; Function Attrs: nounwind
define internal ptr @"$s1t19EitherWithSpareBitsOwcp"(ptr noalias %dest, ptr noalias %src, ptr %EitherWithSpareBits) #5 !dbg !122 {
entry:
  %0 = load i64, ptr %src, align 8, !dbg !123
  %1 = lshr i64 %0, 63, !dbg !123
  %2 = trunc i64 %1 to i1, !dbg !123
  call void @"$s1t19EitherWithSpareBitsOWOy"(i64 %0), !dbg !123
  store i64 %0, ptr %dest, align 8, !dbg !123
  ret ptr %dest, !dbg !123
}

; Function Attrs: nounwind
define internal ptr @"$s1t19EitherWithSpareBitsOwca"(ptr %dest, ptr %src, ptr %EitherWithSpareBits) #5 !dbg !124 {
entry:
  %0 = load i64, ptr %src, align 8, !dbg !125
  %1 = lshr i64 %0, 63, !dbg !125
  %2 = trunc i64 %1 to i1, !dbg !125
  call void @"$s1t19EitherWithSpareBitsOWOy"(i64 %0), !dbg !125
  %3 = load i64, ptr %dest, align 8, !dbg !125
  store i64 %0, ptr %dest, align 8, !dbg !125
  %4 = lshr i64 %3, 63, !dbg !125
  %5 = trunc i64 %4 to i1, !dbg !125
  call void @"$s1t19EitherWithSpareBitsOWOe"(i64 %3), !dbg !125
  ret ptr %dest, !dbg !125
}

; Function Attrs: nounwind
define linkonce_odr hidden ptr @__swift_memcpy8_8(ptr %0, ptr %1, ptr %2) #5 !dbg !126 {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %0, ptr align 8 %1, i64 8, i1 false), !dbg !127
  ret ptr %0, !dbg !127
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #8

; Function Attrs: nounwind
define internal ptr @"$s1t19EitherWithSpareBitsOwta"(ptr noalias %dest, ptr noalias %src, ptr %EitherWithSpareBits) #5 !dbg !128 {
entry:
  %0 = load i64, ptr %src, align 8, !dbg !129
  %1 = load i64, ptr %dest, align 8, !dbg !129
  store i64 %0, ptr %dest, align 8, !dbg !129
  %2 = lshr i64 %1, 63, !dbg !129
  %3 = trunc i64 %2 to i1, !dbg !129
  call void @"$s1t19EitherWithSpareBitsOWOe"(i64 %1), !dbg !129
  ret ptr %dest, !dbg !129
}

; Function Attrs: nounwind memory(read)
define internal i32 @"$s1t19EitherWithSpareBitsOwet"(ptr noalias %value, i32 %numEmptyCases, ptr %EitherWithSpareBits) #9 !dbg !130 {
entry:
  %0 = icmp eq i32 0, %numEmptyCases, !dbg !131
  br i1 %0, label %42, label %1, !dbg !131

1:                                                ; preds = %entry
  %2 = icmp ugt i32 %numEmptyCases, 14, !dbg !131
  br i1 %2, label %3, label %30, !dbg !131

3:                                                ; preds = %1
  %4 = sub i32 %numEmptyCases, 14, !dbg !131
  %5 = getelementptr inbounds i8, ptr %value, i32 8, !dbg !131
  br i1 false, label %6, label %7, !dbg !131

6:                                                ; preds = %3
  br label %19, !dbg !131

7:                                                ; preds = %3
  br i1 true, label %8, label %11, !dbg !131

8:                                                ; preds = %7
  %9 = load i8, ptr %5, align 1, !dbg !131
  %10 = zext i8 %9 to i32, !dbg !131
  br label %19, !dbg !131

11:                                               ; preds = %7
  br i1 false, label %12, label %15, !dbg !131

12:                                               ; preds = %11
  %13 = load i16, ptr %5, align 1, !dbg !131
  %14 = zext i16 %13 to i32, !dbg !131
  br label %19, !dbg !131

15:                                               ; preds = %11
  br i1 false, label %16, label %18, !dbg !131

16:                                               ; preds = %15
  %17 = load i32, ptr %5, align 1, !dbg !131
  br label %19, !dbg !131

18:                                               ; preds = %15
  unreachable, !dbg !131

19:                                               ; preds = %16, %12, %8, %6
  %20 = phi i32 [ 0, %6 ], [ %10, %8 ], [ %14, %12 ], [ %17, %16 ], !dbg !131
  %21 = icmp eq i32 %20, 0, !dbg !131
  br i1 %21, label %30, label %22, !dbg !131

22:                                               ; preds = %19
  %23 = sub i32 %20, 1, !dbg !131
  %24 = shl i32 %23, 64, !dbg !131
  %25 = select i1 true, i32 0, i32 %24, !dbg !131
  %26 = load i64, ptr %value, align 1, !dbg !131
  %27 = trunc i64 %26 to i32, !dbg !131
  %28 = or i32 %27, %25, !dbg !131
  %29 = add i32 14, %28, !dbg !131
  br label %43, !dbg !131

30:                                               ; preds = %19, %1
  %31 = load i64, ptr %value, align 8, !dbg !131
  %32 = lshr i64 %31, 60, !dbg !131
  %33 = trunc i64 %32 to i32, !dbg !131
  %34 = and i32 %33, 15, !dbg !131
  %35 = lshr i32 %34, 3, !dbg !131
  %36 = shl i32 %34, 1, !dbg !131
  %37 = or i32 %35, %36, !dbg !131
  %38 = and i32 %37, 15, !dbg !131
  %39 = sub i32 15, %38, !dbg !131
  %40 = icmp ult i32 %39, 14, !dbg !131
  %41 = select i1 %40, i32 %39, i32 -1, !dbg !131
  br label %43, !dbg !131

42:                                               ; preds = %entry
  br label %43, !dbg !131

43:                                               ; preds = %42, %30, %22
  %44 = phi i32 [ %41, %30 ], [ %29, %22 ], [ -1, %42 ], !dbg !131
  %45 = add i32 %44, 1, !dbg !131
  ret i32 %45, !dbg !131
}

; Function Attrs: nounwind
define internal void @"$s1t19EitherWithSpareBitsOwst"(ptr noalias %value, i32 %whichCase, i32 %numEmptyCases, ptr %EitherWithSpareBits) #5 !dbg !132 {
entry:
  %0 = getelementptr inbounds i8, ptr %value, i32 8, !dbg !133
  %1 = icmp ugt i32 %numEmptyCases, 14, !dbg !133
  br i1 %1, label %2, label %4, !dbg !133

2:                                                ; preds = %entry
  %3 = sub i32 %numEmptyCases, 14, !dbg !133
  br label %4, !dbg !133

4:                                                ; preds = %2, %entry
  %5 = phi i32 [ 0, %entry ], [ 1, %2 ], !dbg !133
  %6 = icmp ule i32 %whichCase, 14, !dbg !133
  br i1 %6, label %7, label %32, !dbg !133

7:                                                ; preds = %4
  %8 = icmp eq i32 %5, 0, !dbg !133
  br i1 %8, label %9, label %10, !dbg !133

9:                                                ; preds = %7
  br label %20, !dbg !133

10:                                               ; preds = %7
  %11 = icmp eq i32 %5, 1, !dbg !133
  br i1 %11, label %12, label %13, !dbg !133

12:                                               ; preds = %10
  store i8 0, ptr %0, align 8, !dbg !133
  br label %20, !dbg !133

13:                                               ; preds = %10
  %14 = icmp eq i32 %5, 2, !dbg !133
  br i1 %14, label %15, label %16, !dbg !133

15:                                               ; preds = %13
  store i16 0, ptr %0, align 8, !dbg !133
  br label %20, !dbg !133

16:                                               ; preds = %13
  %17 = icmp eq i32 %5, 4, !dbg !133
  br i1 %17, label %18, label %19, !dbg !133

18:                                               ; preds = %16
  store i32 0, ptr %0, align 8, !dbg !133
  br label %20, !dbg !133

19:                                               ; preds = %16
  unreachable, !dbg !133

20:                                               ; preds = %18, %15, %12, %9
  %21 = icmp eq i32 %whichCase, 0, !dbg !133
  br i1 %21, label %58, label %22, !dbg !133

22:                                               ; preds = %20
  %23 = sub i32 %whichCase, 1, !dbg !133
  %24 = xor i32 %23, -1, !dbg !133
  %25 = and i32 %24, 15, !dbg !133
  %26 = shl i32 %25, 3, !dbg !133
  %27 = lshr i32 %25, 1, !dbg !133
  %28 = or i32 %26, %27, !dbg !133
  %29 = zext i32 %28 to i64, !dbg !133
  %30 = shl i64 %29, 60, !dbg !133
  %31 = and i64 %30, -1152921504606846976, !dbg !133
  store i64 %31, ptr %value, align 8, !dbg !133
  br label %58, !dbg !133

32:                                               ; preds = %4
  %33 = sub i32 %whichCase, 1, !dbg !133
  %34 = sub i32 %33, 14, !dbg !133
  br i1 true, label %39, label %35, !dbg !133

35:                                               ; preds = %32
  %36 = lshr i32 %34, 64, !dbg !133
  %37 = add i32 1, %36, !dbg !133
  %38 = and i32 poison, %34, !dbg !133
  br label %39, !dbg !133

39:                                               ; preds = %35, %32
  %40 = phi i32 [ 1, %32 ], [ %37, %35 ], !dbg !133
  %41 = phi i32 [ %34, %32 ], [ %38, %35 ], !dbg !133
  %42 = zext i32 %41 to i64, !dbg !133
  store i64 %42, ptr %value, align 8, !dbg !133
  %43 = icmp eq i32 %5, 0, !dbg !133
  br i1 %43, label %44, label %45, !dbg !133

44:                                               ; preds = %39
  br label %57, !dbg !133

45:                                               ; preds = %39
  %46 = icmp eq i32 %5, 1, !dbg !133
  br i1 %46, label %47, label %49, !dbg !133

47:                                               ; preds = %45
  %48 = trunc i32 %40 to i8, !dbg !133
  store i8 %48, ptr %0, align 8, !dbg !133
  br label %57, !dbg !133

49:                                               ; preds = %45
  %50 = icmp eq i32 %5, 2, !dbg !133
  br i1 %50, label %51, label %53, !dbg !133

51:                                               ; preds = %49
  %52 = trunc i32 %40 to i16, !dbg !133
  store i16 %52, ptr %0, align 8, !dbg !133
  br label %57, !dbg !133

53:                                               ; preds = %49
  %54 = icmp eq i32 %5, 4, !dbg !133
  br i1 %54, label %55, label %56, !dbg !133

55:                                               ; preds = %53
  store i32 %40, ptr %0, align 8, !dbg !133
  br label %57, !dbg !133

56:                                               ; preds = %53
  unreachable, !dbg !133

57:                                               ; preds = %55, %51, %47, %44
  br label %58, !dbg !133

58:                                               ; preds = %57, %22, %20
  ret void, !dbg !133
}

; Function Attrs: nounwind
define internal i32 @"$s1t19EitherWithSpareBitsOwug"(ptr noalias %value, ptr %EitherWithSpareBits) #5 !dbg !134 {
entry:
  %0 = load i64, ptr %value, align 8, !dbg !135
  %1 = lshr i64 %0, 63, !dbg !135
  %2 = trunc i64 %1 to i1, !dbg !135
  %3 = zext i1 %2 to i32, !dbg !135
  ret i32 %3, !dbg !135
}

; Function Attrs: nounwind
define internal void @"$s1t19EitherWithSpareBitsOwup"(ptr noalias %value, ptr %EitherWithSpareBits) #5 !dbg !136 {
entry:
  %0 = load i64, ptr %value, align 8, !dbg !137
  %1 = and i64 %0, 9223372036854775807, !dbg !137
  store i64 %1, ptr %value, align 8, !dbg !137
  ret void, !dbg !137
}

; Function Attrs: nounwind
define internal void @"$s1t19EitherWithSpareBitsOwui"(ptr noalias %value, i32 %tag, ptr %EitherWithSpareBits) #5 !dbg !138 {
entry:
  %0 = and i32 %tag, 1, !dbg !139
  %1 = load i64, ptr %value, align 8, !dbg !139
  %2 = and i64 %1, 1152921504606846975, !dbg !139
  %3 = zext i32 %0 to i64, !dbg !139
  %4 = shl i64 %3, 63, !dbg !139
  %5 = and i64 %4, -9223372036854775808, !dbg !139
  %6 = or i64 %5, %2, !dbg !139
  store i64 %6, ptr %value, align 8, !dbg !139
  ret void, !dbg !139
}

; Function Attrs: noinline nounwind memory(none)
define hidden swiftcc %swift.metadata_response @"$s1t19EitherWithSpareBitsOMa"(i64 %0) #4 !dbg !140 {
entry:
  ret %swift.metadata_response { ptr getelementptr inbounds (<{ ptr, ptr, i64, ptr }>, ptr @"$s1t19EitherWithSpareBitsOMf", i32 0, i32 2), i64 0 }, !dbg !141
}

attributes #0 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-a12" "target-features"="+aes,+ccidx,+complxnum,+crc,+fp-armv8,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+ras,+rcpc,+rdm,+sha2,+v8.1a,+v8.2a,+v8.3a,+v8a,+zcm,+zcz" }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #3 = { nounwind }
attributes #4 = { noinline nounwind memory(none) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-a12" "target-features"="+aes,+ccidx,+complxnum,+crc,+fp-armv8,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+ras,+rcpc,+rdm,+sha2,+v8.1a,+v8.2a,+v8.3a,+v8a,+zcm,+zcz" }
attributes #5 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-a12" "target-features"="+aes,+ccidx,+complxnum,+crc,+fp-armv8,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+ras,+rcpc,+rdm,+sha2,+v8.1a,+v8.2a,+v8.3a,+v8a,+zcm,+zcz" }
attributes #6 = { noinline nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-a12" "target-features"="+aes,+ccidx,+complxnum,+crc,+fp-armv8,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+ras,+rcpc,+rdm,+sha2,+v8.1a,+v8.2a,+v8.3a,+v8a,+zcm,+zcz" }
attributes #7 = { nounwind willreturn }
attributes #8 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #9 = { nounwind memory(read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-a12" "target-features"="+aes,+ccidx,+complxnum,+crc,+fp-armv8,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+ras,+rcpc,+rdm,+sha2,+v8.1a,+v8.2a,+v8.3a,+v8a,+zcm,+zcz" }

!llvm.module.flags = !{!22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34}
!llvm.dbg.cu = !{!35, !62, !64}
!swift.module.flags = !{!66}
!llvm.linker.options = !{!67, !68, !69, !70, !71}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "Right", linkageName: "$s1t5RightAA19EitherWithSpareBitsOvp", scope: !2, file: !3, line: 15, type: !4, isLocal: false, isDefinition: true)
!2 = !DIModule(scope: null, name: "t")
!3 = !DIFile(filename: "t.swift", directory: "swift-macosx-arm64")
!4 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !5)
!5 = !DICompositeType(tag: DW_TAG_structure_type, name: "EitherWithSpareBits", scope: !2, file: !3, line: 2, size: 64, num_extra_inhabitants: 14, elements: !6, runtimeLang: DW_LANG_Swift, identifier: "$s1t19EitherWithSpareBitsOD")
!6 = !{!7}
!7 = !DICompositeType(tag: DW_TAG_variant_part, scope: !2, file: !3, line: 2, size: 64, offset: 56, spare_bits_mask: 240, elements: !8)
!8 = !{!9, !12}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "Left", scope: !2, file: !3, baseType: !10, size: 64)
!10 = !DICompositeType(tag: DW_TAG_structure_type, name: "C", scope: !2, file: !3, line: 1, size: 64, elements: !11, runtimeLang: DW_LANG_Swift, identifier: "$s1t1CCD")
!11 = !{}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "Right", scope: !2, file: !3, baseType: !13, size: 32)
!13 = !DICompositeType(tag: DW_TAG_structure_type, name: "Int32", scope: !15, file: !14, size: 32, elements: !16, runtimeLang: DW_LANG_Swift, identifier: "$ss5Int32VD")
!14 = !DIFile(filename: "lib/swift/macosx/Swift.swiftmodule/arm64-apple-macos.swiftmodule", directory: "swift-macosx-arm64")
!15 = !DIModule(scope: null, name: "Swift", includePath: "swift-macosx-arm64/lib/swift/macosx/Swift.swiftmodule/arm64-apple-macos.swiftmodule")
!16 = !{!17}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "_value", scope: !15, file: !14, baseType: !18, size: 32)
!18 = !DIDerivedType(tag: DW_TAG_typedef, name: "$sBi32_D", scope: !19, baseType: !20)
!19 = !DIModule(scope: null, name: "Builtin")
!20 = !DIDerivedType(tag: DW_TAG_typedef, name: "$sBi32_D", scope: !19, baseType: !21)
!21 = !DIBasicType(name: "$sBi32_D", size: 32, encoding: DW_ATE_unsigned)
!22 = !{i32 2, !"SDK Version", [2 x i32] [i32 14, i32 5]}
!23 = !{i32 1, !"Objective-C Version", i32 2}
!24 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!25 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!26 = !{i32 4, !"Objective-C Garbage Collection", i32 100730624}
!27 = !{i32 1, !"Objective-C Class Properties", i32 64}
!28 = !{i32 7, !"Dwarf Version", i32 5}
!29 = !{i32 2, !"Debug Info Version", i32 3}
!30 = !{i32 1, !"wchar_size", i32 4}
!31 = !{i32 8, !"PIC Level", i32 2}
!32 = !{i32 7, !"uwtable", i32 1}
!33 = !{i32 7, !"frame-pointer", i32 1}
!34 = !{i32 1, !"Swift Version", i32 7}
!35 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !3, producer: "Swift version 6.1-dev effective-5.10 (LLVM 240db8d531ea284, Swift af036feb5059859)", isOptimized: false, runtimeVersion: 6, emissionKind: FullDebug, retainedTypes: !36, globals: !50, imports: !51, sysroot: "", sdk: "MacOSX14.5.Internal.sdk")
!36 = !{!37, !38, !39, !40, !41, !42, !49}
!37 = !DICompositeType(tag: DW_TAG_structure_type, name: "$sBoD", scope: !19, file: !3, size: 64, num_extra_inhabitants: 2147483647, flags: DIFlagArtificial, runtimeLang: DW_LANG_Swift)
!38 = !DICompositeType(tag: DW_TAG_structure_type, name: "$syXlD", size: 64, elements: !11, runtimeLang: DW_LANG_Swift, identifier: "$syXlD")
!39 = !DICompositeType(tag: DW_TAG_structure_type, name: "$sBbD", scope: !19, file: !3, size: 64, num_extra_inhabitants: 2147483647, flags: DIFlagArtificial, runtimeLang: DW_LANG_Swift)
!40 = !DICompositeType(tag: DW_TAG_structure_type, name: "$sBpD", scope: !19, file: !3, size: 64, num_extra_inhabitants: 1, flags: DIFlagArtificial, runtimeLang: DW_LANG_Swift)
!41 = !DIBasicType(name: "<unknown>", size: 192)
!42 = !DICompositeType(tag: DW_TAG_structure_type, name: "$syyXfD", file: !3, size: 64, elements: !43, runtimeLang: DW_LANG_Swift, identifier: "$syyXfD")
!43 = !{!44}
!44 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", file: !3, baseType: !45, size: 64)
!45 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !46, size: 64)
!46 = !DISubroutineType(types: !47)
!47 = !{!48}
!48 = !DICompositeType(tag: DW_TAG_structure_type, name: "$sytD", flags: DIFlagFwdDecl, runtimeLang: DW_LANG_Swift)
!49 = !DICompositeType(tag: DW_TAG_structure_type, name: "$sypXpD", size: 64, flags: DIFlagArtificial, runtimeLang: DW_LANG_Swift, identifier: "$sypXpD")
!50 = !{!0}
!51 = !{!52, !53, !54, !56, !58, !60}
!52 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !3, entity: !2, file: !3)
!53 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !3, entity: !15, file: !3)
!54 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !3, entity: !55, file: !3)
!55 = !DIModule(scope: null, name: "_StringProcessing", includePath: "swift-macosx-arm64/lib/swift/macosx/_StringProcessing.swiftmodule/arm64-apple-macos.swiftmodule")
!56 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !3, entity: !57, file: !3)
!57 = !DIModule(scope: null, name: "_SwiftConcurrencyShims", includePath: "swift-macosx-arm64/lib/swift/shims")
!58 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !3, entity: !59, file: !3)
!59 = !DIModule(scope: null, name: "_Concurrency", includePath: "swift-macosx-arm64/lib/swift/macosx/_Concurrency.swiftmodule/arm64-apple-macos.swiftmodule")
!60 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !3, entity: !61, file: !3)
!61 = !DIModule(scope: null, name: "SwiftOnoneSupport", includePath: "swift-macosx-arm64/lib/swift/macosx/SwiftOnoneSupport.swiftmodule/arm64-apple-macos.swiftmodule")
!62 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !63, producer: "clang version 17.0.0 (git@github.com:swiftlang/llvm-project.git 240db8d531ea28417de83f19fdfae940c94b9891)", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "", sdk: "MacOSX14.5.Internal.sdk")
!63 = !DIFile(filename: "<swift-imported-modules>", directory: "swift-macosx-arm64", checksumkind: CSK_MD5, checksum: "99914b48a0432431a8f86e65380c1775")
!64 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !65, producer: "Swift version 6.1-dev effective-5.10 (LLVM 240db8d531ea284, Swift af036feb5059859)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, dwoId: 8421522778062590892)
!65 = !DIFile(filename: "_SwiftConcurrencyShims", directory: "swift-macosx-arm64/lib/swift/shims")
!66 = !{!"standard-library", i1 false}
!67 = !{!"-lswiftSwiftOnoneSupport"}
!68 = !{!"-lswiftCore"}
!69 = !{!"-lswift_Concurrency"}
!70 = !{!"-lswift_StringProcessing"}
!71 = !{!"-lobjc"}
!72 = distinct !DISubprogram(name: "main", linkageName: "main", scope: !2, file: !3, line: 1, type: !73, spFlags: DISPFlagDefinition, unit: !35)
!73 = !DISubroutineType(types: !74)
!74 = !{!13, !13, !75}
!75 = !DICompositeType(tag: DW_TAG_structure_type, name: "$sSpySpys4Int8VGSgGD", scope: !15, flags: DIFlagFwdDecl, runtimeLang: DW_LANG_Swift)
!76 = !DILocation(line: 15, column: 35, scope: !77)
!77 = distinct !DILexicalBlock(scope: !78, file: !3, line: 15, column: 34)
!78 = distinct !DILexicalBlock(scope: !72, file: !3, line: 15, column: 1)
!79 = distinct !DISubprogram(name: "deinit", linkageName: "$s1t1CCfd", scope: !10, file: !3, line: 1, type: !80, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !35, declaration: !82, retainedNodes: !83)
!80 = !DISubroutineType(types: !81)
!81 = !{!37, !10}
!82 = !DISubprogram(name: "deinit", linkageName: "$s1t1CCfd", scope: !10, file: !3, line: 1, type: !80, scopeLine: 1, spFlags: 0)
!83 = !{!84}
!84 = !DILocalVariable(name: "self", arg: 1, scope: !79, file: !3, line: 1, type: !85, flags: DIFlagArtificial)
!85 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !10)
!86 = !DILocation(line: 1, column: 7, scope: !79)
!87 = !DILocation(line: 0, scope: !79)
!88 = distinct !DISubprogram(name: "deinit", linkageName: "$s1t1CCfD", scope: !10, file: !3, line: 1, type: !89, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !35, declaration: !91, retainedNodes: !92)
!89 = !DISubroutineType(types: !90)
!90 = !{!48, !10}
!91 = !DISubprogram(name: "deinit", linkageName: "$s1t1CCfD", scope: !10, file: !3, line: 1, type: !89, scopeLine: 1, spFlags: 0)
!92 = !{!93}
!93 = !DILocalVariable(name: "self", arg: 1, scope: !88, file: !3, line: 1, type: !85, flags: DIFlagArtificial)
!94 = !DILocation(line: 1, column: 7, scope: !88)
!95 = !DILocation(line: 0, scope: !88)
!96 = distinct !DISubprogram(name: "init", linkageName: "$s1t1CCACycfC", scope: !10, file: !3, line: 1, type: !97, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !35, declaration: !100)
!97 = !DISubroutineType(types: !98)
!98 = !{!10, !99}
!99 = !DICompositeType(tag: DW_TAG_structure_type, name: "$s1t1CCXMTD", flags: DIFlagFwdDecl, runtimeLang: DW_LANG_Swift)
!100 = !DISubprogram(name: "init", linkageName: "$s1t1CCACycfC", scope: !10, file: !3, line: 1, type: !97, scopeLine: 1, spFlags: 0)
!101 = !DILocation(line: 0, scope: !96)
!102 = distinct !DISubprogram(name: "init", linkageName: "$s1t1CCACycfc", scope: !10, file: !3, line: 1, type: !103, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !35, declaration: !105, retainedNodes: !106)
!103 = !DISubroutineType(types: !104)
!104 = !{!10, !10}
!105 = !DISubprogram(name: "init", linkageName: "$s1t1CCACycfc", scope: !10, file: !3, line: 1, type: !103, scopeLine: 1, spFlags: 0)
!106 = !{!107}
!107 = !DILocalVariable(name: "self", arg: 1, scope: !102, file: !3, line: 1, type: !85, flags: DIFlagArtificial)
!108 = !DILocation(line: 1, column: 7, scope: !102)
!109 = !DILocation(line: 0, scope: !102)
!110 = distinct !DISubprogram(linkageName: "$s1t1CCMa", scope: !2, file: !111, type: !112, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !35)
!111 = !DIFile(filename: "<compiler-generated>", directory: "/")
!112 = !DISubroutineType(types: null)
!113 = !DILocation(line: 0, scope: !110)
!114 = distinct !DISubprogram(linkageName: "$s1t19EitherWithSpareBitsOwCP", scope: !2, file: !111, type: !112, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !35)
!115 = !DILocation(line: 0, scope: !114)
!116 = distinct !DISubprogram(linkageName: "$s1t19EitherWithSpareBitsOWOy", scope: !2, file: !111, type: !112, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !35)
!117 = !DILocation(line: 0, scope: !116)
!118 = distinct !DISubprogram(linkageName: "$s1t19EitherWithSpareBitsOwxx", scope: !2, file: !111, type: !112, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !35)
!119 = !DILocation(line: 0, scope: !118)
!120 = distinct !DISubprogram(linkageName: "$s1t19EitherWithSpareBitsOWOe", scope: !2, file: !111, type: !112, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !35)
!121 = !DILocation(line: 0, scope: !120)
!122 = distinct !DISubprogram(linkageName: "$s1t19EitherWithSpareBitsOwcp", scope: !2, file: !111, type: !112, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !35)
!123 = !DILocation(line: 0, scope: !122)
!124 = distinct !DISubprogram(linkageName: "$s1t19EitherWithSpareBitsOwca", scope: !2, file: !111, type: !112, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !35)
!125 = !DILocation(line: 0, scope: !124)
!126 = distinct !DISubprogram(linkageName: "__swift_memcpy8_8", scope: !2, file: !111, type: !112, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !35)
!127 = !DILocation(line: 0, scope: !126)
!128 = distinct !DISubprogram(linkageName: "$s1t19EitherWithSpareBitsOwta", scope: !2, file: !111, type: !112, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !35)
!129 = !DILocation(line: 0, scope: !128)
!130 = distinct !DISubprogram(linkageName: "$s1t19EitherWithSpareBitsOwet", scope: !2, file: !111, type: !112, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !35)
!131 = !DILocation(line: 0, scope: !130)
!132 = distinct !DISubprogram(linkageName: "$s1t19EitherWithSpareBitsOwst", scope: !2, file: !111, type: !112, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !35)
!133 = !DILocation(line: 0, scope: !132)
!134 = distinct !DISubprogram(linkageName: "$s1t19EitherWithSpareBitsOwug", scope: !2, file: !111, type: !112, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !35)
!135 = !DILocation(line: 0, scope: !134)
!136 = distinct !DISubprogram(linkageName: "$s1t19EitherWithSpareBitsOwup", scope: !2, file: !111, type: !112, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !35)
!137 = !DILocation(line: 0, scope: !136)
!138 = distinct !DISubprogram(linkageName: "$s1t19EitherWithSpareBitsOwui", scope: !2, file: !111, type: !112, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !35)
!139 = !DILocation(line: 0, scope: !138)
!140 = distinct !DISubprogram(linkageName: "$s1t19EitherWithSpareBitsOMa", scope: !2, file: !111, type: !112, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !35)
!141 = !DILocation(line: 0, scope: !140)


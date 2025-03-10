; RUN: llc  %s -filetype=obj -o - | llvm-dwarfdump -v - | FileCheck %s
; RUN: llc  --try-experimental-debuginfo-iterators %s -filetype=obj -o - | llvm-dwarfdump -v - | FileCheck %s

; C++ source to regenerate:

;template <typename T = char, int i = 3, float f = 1.0f, double d = 2.0>
;class foo {
;};
;
;int main() {
; foo<int, 6, 1.9f, 1.9> f1;
; foo<> f2;
; return 0;
;}

; $ clang++ -O0 -gdwarf-5 -S -gdwarf-5 test.cpp 

; CHECK: .debug_abbrev contents:
; CHECK: DW_AT_default_value     DW_FORM_flag_present

; CHECK: debug_info contents:

; CHECK: DW_AT_name {{.*}} "foo<int, 6, 1.900000e+00, 1.900000e+00>"
; CHECK: DW_AT_type {{.*}} "int"
; CHECK-NEXT: DW_AT_name {{.*}} "T"
; CHECK-NOT: DW_AT_default_value
; CHECK: DW_AT_type {{.*}} "int"
; CHECK-NEXT: DW_AT_name {{.*}} "i"
; CHECK-NEXT: DW_AT_const_value [DW_FORM_sdata] (6)
; CHECK-NOT: DW_AT_default_value
; CHECK: DW_AT_type {{.*}} "float"
; CHECK-NEXT: DW_AT_name {{.*}} "f"
; CHECK-NEXT: DW_AT_const_value [DW_FORM_udata] (1072902963)
; CHECK-NOT: DW_AT_default_value
; CHECK: DW_AT_type {{.*}} "double"
; CHECK-NEXT: DW_AT_name {{.*}} "d"
; CHECK-NEXT: DW_AT_const_value [DW_FORM_udata] (4611235658464650854)
; CHECK-NOT: DW_AT_default_value

; CHECK: DW_AT_name {{.*}} "foo<char, 3, 1.000000e+00, 2.000000e+00>"
; CHECK: DW_AT_type {{.*}} "char"
; CHECK-NEXT: DW_AT_name {{.*}} "T"
; CHECK-NEXT: DW_AT_default_value {{.*}} (true)
; CHECK: DW_AT_type {{.*}} "int"
; CHECK-NEXT: DW_AT_name {{.*}} "i"
; CHECK-NEXT: DW_AT_default_value {{.*}} (true)
; CHECK: DW_AT_type {{.*}} "float"
; CHECK-NEXT: DW_AT_name {{.*}} "f"
; CHECK-NEXT: DW_AT_default_value {{.*}} (true)
; CHECK: DW_AT_type {{.*}} "double"
; CHECK-NEXT: DW_AT_name {{.*}} "d"
; CHECK-NEXT: DW_AT_default_value {{.*}} (true)

; ModuleID = '/dir/test.cpp'
source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.foo = type { i8 }
%class.foo.0 = type { i8 }
; Function Attrs: noinline norecurse nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !7 {
entry:
  %retval = alloca i32, align 4
  %f1 = alloca %class.foo, align 1
  %f2 = alloca %class.foo.0, align 1
  store i32 0, ptr %retval, align 4
  call void @llvm.dbg.declare(metadata ptr %f1, metadata !11, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata ptr %f2, metadata !17, metadata !DIExpression()), !dbg !23
  ret i32 0, !dbg !24
}
; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline norecurse nounwind optnone uwtable }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 11.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/dir/", checksumkind: CSK_MD5, checksum: "863d08522c2300490dea873efc4b2369")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 11.0.0"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 29, type: !8, scopeLine: 29, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "f1", scope: !7, file: !1, line: 30, type: !12)
!12 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "foo<int, 6, 1.900000e+00, 1.900000e+00>", file: !1, line: 26, size: 8, flags: DIFlagTypePassByValue, elements: !2, templateParams: !13, identifier: "_ZTS3fooIiLi6ELf3ff33333ELd3ffe666666666666EE")
!13 = !{!14, !15, !25, !27}
!14 = !DITemplateTypeParameter(name: "T", type: !10)
!15 = !DITemplateValueParameter(name: "i", type: !10, value: i32 6)
!16 = !DILocation(line: 30, column: 14, scope: !7)
!17 = !DILocalVariable(name: "f2", scope: !7, file: !1, line: 31, type: !18)
!18 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "foo<char, 3, 1.000000e+00, 2.000000e+00>", file: !1, line: 26, size: 8, flags: DIFlagTypePassByValue, elements: !2, templateParams: !19, identifier: "_ZTS3fooIcLi3ELf3f800000ELd4000000000000000EE")
!19 = !{!20, !22, !29, !30}
!20 = !DITemplateTypeParameter(name: "T", type: !21, defaulted: true)
!21 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!22 = !DITemplateValueParameter(name: "i", type: !10, defaulted: true, value: i32 3)
!23 = !DILocation(line: 31, column: 9, scope: !7)
!24 = !DILocation(line: 32, column: 3, scope: !7)
!25 = !DITemplateValueParameter(name: "f", type: !26, value: float 0x3FFE666660000000)
!26 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!27 = !DITemplateValueParameter(name: "d", type: !28, value: double 1.900000e+00)
!28 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!29 = !DITemplateValueParameter(name: "f", type: !26, defaulted: true, value: float 1.000000e+00)
!30 = !DITemplateValueParameter(name: "d", type: !28, defaulted: true, value: double 2.000000e+00)

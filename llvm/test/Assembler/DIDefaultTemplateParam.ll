; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

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

;; Test that the templateParams for the DICompositeTypes are retained:
; CHECK:      ![[COMPTAG:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "foo<int, 6>",
; CHECK-SAME: templateParams: ![[PARAMLIST:[0-9]+]]
; CHECK:      ![[PARAMLIST]] = !{![[PARAM1:[0-9]+]], ![[PARAM2:[0-9]+]]}
; CHECK:      ![[PARAM1]] = !DITemplateTypeParameter(name: "T", type: !{{[0-9]*}})
; CHECK:      ![[PARAM2]] = !DITemplateValueParameter(name: "i", type: !{{[0-9]*}}, value: i32 6)

; CHECK:      ![[COMPTAG2:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "foo<char, 3>",
; CHECK-SAME: templateParams: ![[PARAMLIST2:[0-9]+]]
; CHECK:      ![[PARAMLIST2]] = !{![[PARAM3:[0-9]+]], ![[PARAM4:[0-9]+]]}
; CHECK:      ![[PARAM3]] = !DITemplateTypeParameter({{.*}}, defaulted: true
; CHECK:      ![[PARAM4]] = !DITemplateValueParameter({{.*}}, defaulted: true

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
!12 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "foo<int, 6>", file: !1, line: 26, size: 8, flags: DIFlagTypePassByValue, elements: !2, templateParams: !13, identifier: "_ZTS3fooIiLi6EE")
!13 = !{!14, !15}
!14 = !DITemplateTypeParameter(name: "T", type: !10)
!15 = !DITemplateValueParameter(name: "i", type: !10, value: i32 6)
!16 = !DILocation(line: 30, column: 14, scope: !7)
!17 = !DILocalVariable(name: "f2", scope: !7, file: !1, line: 31, type: !18)
!18 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "foo<char, 3>", file: !1, line: 26, size: 8, flags: DIFlagTypePassByValue, elements: !2, templateParams: !19, identifier: "_ZTS3fooIcLi3EE")
!19 = !{!20, !22}
!20 = !DITemplateTypeParameter(name: "T", type: !21, defaulted: true)
!21 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!22 = !DITemplateValueParameter(name: "i", type: !10, defaulted: true, value: i32 3)
!23 = !DILocation(line: 31, column: 9, scope: !7)
!24 = !DILocation(line: 32, column: 3, scope: !7)

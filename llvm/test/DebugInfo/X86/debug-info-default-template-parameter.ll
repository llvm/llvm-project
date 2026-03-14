; Tests that DW_AT_default_value is correctly emitted for C++ template
; parameters that are defaulted.
;
; ModuleID = 'debug-info-template-parameter.cpp'
;
; template <typename T = char, int i = 3, bool b = true, int x = sizeof(T)>
; class foo {
; };
; 
; int main() {
;   foo<int, 6, false, 3> f1;
;   foo<> f2;
;   return 0;
; }
;
; RUN: llc -filetype=obj -dwarf-version=4 %s -o - | llvm-dwarfdump - --debug-info | FileCheck %s --check-prefixes=DWARF-DUMP,DWARFv4
; RUN: llc -filetype=obj -dwarf-version=4 -strict-dwarf=true %s -o - | llvm-dwarfdump - --debug-info | FileCheck %s --check-prefixes=DWARF-DUMP,STRICT

; DWARF-DUMP:       DW_TAG_class_type
; DWARF-DUMP-LABEL:   DW_AT_name      ("foo<char, 3, true, 1>")
; DWARF-DUMP:         DW_TAG_template_type_parameter
; DWARF-DUMP-DAG:       DW_AT_type    ({{.*}} "char")
; DWARF-DUMP-DAG:       DW_AT_name    ("T")
; DWARFv4-DAG:          DW_AT_default_value   (true)
; STRICT-NOT:           DW_AT_default_value
; DWARF-DUMP:         DW_TAG_template_value_parameter
; DWARF-DUMP-DAG:       DW_AT_type    ({{.*}} "int")
; DWARF-DUMP-DAG:       DW_AT_name    ("i")
; DWARFv4-DAG:          DW_AT_default_value   (true)
; STRICT-NOT:           DW_AT_default_value   (true)
; DWARF-DUMP:         DW_TAG_template_value_parameter
; DWARF-DUMP-DAG:       DW_AT_type    ({{.*}} "bool")
; DWARF-DUMP-DAG:       DW_AT_name    ("b")
; DWARFv4-DAG:          DW_AT_default_value   (true)
; STRICT-NOT:           DW_AT_default_value
; DWARF-DUMP:           {{DW_TAG|NULL}}

source_filename = "/tmp/debug-info-template-parameter.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64"

%class.foo = type { i8 }
%class.foo.0 = type { i8 }

; Function Attrs: mustprogress noinline norecurse nounwind optnone
define dso_local noundef i32 @main() #0 !dbg !6 {
entry:
  %retval = alloca i32, align 4
  %f1 = alloca %class.foo, align 1
  %f2 = alloca %class.foo.0, align 1
  store i32 0, ptr %retval, align 4
  call void @llvm.dbg.declare(metadata ptr %f1, metadata !12, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.declare(metadata ptr %f2, metadata !21, metadata !DIExpression()), !dbg !29
  ret i32 0, !dbg !30
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { mustprogress noinline norecurse nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.linker.options = !{}
!llvm.module.flags = !{!2, !3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 16.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/debug-info-template-parameter.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 16.0.0"}
!6 = distinct !DISubprogram(name: "main", scope: !7, file: !7, line: 49, type: !8, scopeLine: 49, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !11)
!7 = !DIFile(filename: "/tmp/debug-info-template-parameter.cpp", directory: "/")
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{}
!12 = !DILocalVariable(name: "f1", scope: !6, file: !7, line: 50, type: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "foo<int, 6, false, 3>", file: !7, line: 46, size: 8, flags: DIFlagTypePassByValue, elements: !11, templateParams: !14, identifier: "_ZTS3fooIiLi6ELb0ELi3EE")
!14 = !{!15, !16, !17, !19}
!15 = !DITemplateTypeParameter(name: "T", type: !10)
!16 = !DITemplateValueParameter(name: "i", type: !10, value: i32 6)
!17 = !DITemplateValueParameter(name: "b", type: !18, value: i1 false)
!18 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!19 = !DITemplateValueParameter(name: "x", type: !10, value: i32 3)
!20 = !DILocation(line: 50, column: 25, scope: !6)
!21 = !DILocalVariable(name: "f2", scope: !6, file: !7, line: 51, type: !22)
!22 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "foo<char, 3, true, 1>", file: !7, line: 46, size: 8, flags: DIFlagTypePassByValue, elements: !11, templateParams: !23, identifier: "_ZTS3fooIcLi3ELb1ELi1EE")
!23 = !{!24, !26, !27, !28}
!24 = !DITemplateTypeParameter(name: "T", type: !25, defaulted: true)
!25 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!26 = !DITemplateValueParameter(name: "i", type: !10, defaulted: true, value: i32 3)
!27 = !DITemplateValueParameter(name: "b", type: !18, defaulted: true, value: i1 true)
!28 = !DITemplateValueParameter(name: "x", type: !10, value: i32 1)
!29 = !DILocation(line: 51, column: 9, scope: !6)
!30 = !DILocation(line: 52, column: 3, scope: !6)

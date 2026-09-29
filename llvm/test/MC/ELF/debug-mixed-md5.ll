; RUN: %llc_dwarf -filetype=asm -dwarf-version=5 %s -o - | FileCheck %s -check-prefix=ASM
; RUN: %llc_dwarf -filetype=obj -dwarf-version=5 %s -o - | llvm-dwarfdump -debug-line - | FileCheck %s -check-prefix=OBJ
; ASM: .file 0 "{{.+}}" md5
; ASM: .file 1 "t1.cpp"
; ASM-NOT: md5
; OBJ: file_names[ 0]:
; OBJ-NOT: md5
;
; Generated from this source (see PR37623):
;
; #define a(...) template __VA_ARGS__;
; template <class> class b {};
; a(class b<int>)
; # 1 ""
; int c;

; ModuleID = 't1.cpp'
source_filename = "t1.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@c = global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !3, line: 1, type: !10, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 7.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !11)
!3 = !DIFile(filename: "<stdin>", directory: "/home/probinson/projects/scratch", checksumkind: CSK_MD5, checksum: "9252ff18ee25a08c2b4216b21b5d66d4")
!4 = !{}
!5 = !{!6}
!6 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "b<int>", file: !7, line: 3, size: 8, flags: DIFlagTypePassByValue, elements: !4, templateParams: !8, identifier: "_ZTS1bIiE")
!7 = !DIFile(filename: "t1.cpp", directory: "/home/probinson/projects/scratch")
!8 = !{!9}
!9 = !DITemplateTypeParameter(type: !10)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!0}
!12 = !{i32 2, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{!"clang version 7.0.0 "}

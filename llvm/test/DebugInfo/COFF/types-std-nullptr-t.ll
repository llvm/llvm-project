; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s

; C++ source to regenerate:
; $ cat foo.cpp
; decltype(nullptr) NullPtr = nullptr;
; $ clang hello.cpp -S -emit-llvm -g -gcodeview -o t.ll

; CHECK: CodeViewDebugInfo [
; CHECK:   Subsection [
; CHECK:     SubSectionType: Symbols (0xF1)
; CHECK:     GlobalData {
; CHECK:       Kind: S_GDATA32 (0x110D)
; CHECK:       DataOffset: ?NullPtr@@3$$TA+0x0
; CHECK:       Type: std::nullptr_t (0x103)
; CHECK:       DisplayName: NullPtr
; CHECK:       LinkageName: ?NullPtr@@3$$TA
; CHECK:     }


; ModuleID = 'foo.cpp'
source_filename = "foo.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.15.26730"

@"?NullPtr@@3$$TA" = dso_local global ptr null, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "NullPtr", linkageName: "?NullPtr@@3$$TA", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 8.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "foo.cpp", directory: "D:\5Csrc\5Cllvmbuild\5Ccl\5CDebug\5Cx64", checksumkind: CSK_MD5, checksum: "0d5c7c9860a17e584808c03a24a135e6")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "decltype(nullptr)")
!7 = !{i32 2, !"CodeView", i32 1}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 2}
!10 = !{i32 7, !"PIC Level", i32 2}
!11 = !{!"clang version 8.0.0 "}

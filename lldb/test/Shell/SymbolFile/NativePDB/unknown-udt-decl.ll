; Test that the declaration for UDTs won't be "<unknown>" or "\<unknown>".
; Rustc sets the location of some builtin types to this string.

; REQUIRES: system-windows
; RUN: %build --compiler=clang-cl --nodefaultlib -o %t.exe -- %s
; RUN: lldb-test symbols %t.exe | FileCheck %s

; there shouldn't be a declaration (would be between size and compiler_type)
; CHECK: Type{{.*}} , name = "Foo", size = 1, compiler_type = {{.*}} struct Foo {

; This is edited output from clang  simulates rustc behavior (see !17)
; Source:
; struct Foo {};
;
; int main() { Foo f; }


; ModuleID = 'main.cpp'
source_filename = "main.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.44.35207"

%struct.Foo = type { i8 }

; Function Attrs: mustprogress noinline norecurse nounwind optnone uwtable
define dso_local noundef i32 @main() #0 !dbg !9 {
  %1 = alloca %struct.Foo, align 1
    #dbg_declare(ptr %1, !14, !DIExpression(), !16)
  ret i32 0, !dbg !16
}

attributes #0 = { mustprogress noinline norecurse nounwind optnone uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 20.1.6", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "main.cpp", directory: "F:\\Dev\\rust-dbg-test", checksumkind: CSK_MD5, checksum: "b8942260dadf9ec35328889f05afb954")
!2 = !{i32 2, !"CodeView", i32 1}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 2}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 2}
!7 = !{i32 1, !"MaxTLSAlign", i32 65536}
!8 = !{!"clang version 20.1.6"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{}
!14 = !DILocalVariable(name: "f", scope: !9, file: !1, line: 3, type: !15)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !17, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !13, identifier: ".?AUFoo@@")
!16 = !DILocation(line: 3, scope: !9)
; This is how rustc emits some types
!17 = !DIFile(filename: "<unknown>", directory: "")

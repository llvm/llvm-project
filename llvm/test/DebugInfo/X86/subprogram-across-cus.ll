; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -verify %t
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s
; RUN: rm %t
;
;$ cat -n 1.cpp
;     1  struct HHH;
;     2  HHH *zzz;
;
;$ cat -n 2.cpp
;     1  void __attribute__((optnone)) __attribute__((nodebug)) f1()  { }
;     2
;     3  struct HHH {
;     4    template <typename bbb>
;     5    static int __attribute__((always_inline)) ccc() {
;     6      f1();
;     7    }
;     8  };
;     9
;    10  int main() {
;    11    struct local { };
;    12    HHH::ccc<local>();
;    13  }
;
; $ clang -flto -O2 -g 1.cpp 2.cpp -o a.out
;
; Given this input, LLVM attempts to create a DIE for subprogram "main" in the
; wrong context. The definition of struct "HHH" is placed in the CU for 1.cpp.
; While creating the template instance in "HHH", function "ccc" is referenced
; via the struct local type, and "main" is created in the CU for 1.cpp, which
; is incorrect.
;
; See PR48790 for more discussion and original compile commands.
;
; Check that there are no verifier failures, and that the SP for "main" appears
; in the correct CU.
; CHECK-LABEL:      DW_TAG_compile_unit
; CHECK:              DW_AT_name ("1.cpp")
; CHECK-NOT:          DW_AT_name ("main")
; CHECK-LABEL:      DW_TAG_compile_unit
; CHECK:              DW_AT_name ("2.cpp")
; CHECK:            DW_TAG_subprogram
; CHECK:              DW_AT_name ("main")

source_filename = "ld-temp.o"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline norecurse nounwind optnone uwtable mustprogress
define internal fastcc void @_Z2f1v() unnamed_addr {
entry:
  ret void
}

; Function Attrs: norecurse noreturn nounwind uwtable mustprogress
define dso_local i32 @main() local_unnamed_addr !dbg !17 {
entry:
  tail call fastcc void @_Z2f1v(), !dbg !21
  unreachable, !dbg !21
}

!llvm.dbg.cu = !{!0, !9}
!llvm.ident = !{!10, !10}
!llvm.module.flags = !{!11, !12, !13, !14, !15}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 12.0.0 (git@github.com:llvm/llvm-project bc9ab9a5cd6bafc5e1293f3d5d51638f8f5cd26c)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "1.cpp", directory: "/tmp/bees")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = distinct !DIGlobalVariable(name: "zzz", scope: !0, file: !1, line: 2, type: !6, isLocal: false, isDefinition: true)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "HHH", file: !8, line: 3, size: 8, flags: DIFlagTypePassByValue, elements: !2, identifier: "_ZTS3HHH")
!8 = !DIFile(filename: "2.cpp", directory: "/tmp/bees")
!9 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !8, producer: "clang version 12.0.0 (git@github.com:llvm/llvm-project bc9ab9a5cd6bafc5e1293f3d5d51638f8f5cd26c)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!10 = !{!"clang version 12.0.0 (git@github.com:llvm/llvm-project bc9ab9a5cd6bafc5e1293f3d5d51638f8f5cd26c)"}
!11 = !{i32 7, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 1, !"ThinLTO", i32 0}
!15 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!17 = distinct !DISubprogram(name: "main", scope: !8, file: !8, line: 10, type: !18, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !9, retainedNodes: !2)
!18 = !DISubroutineType(types: !19)
!19 = !{!20}
!20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!21 = !DILocation(line: 6, column: 5, scope: !22, inlinedAt: !27)
!22 = distinct !DISubprogram(name: "ccc<local>", linkageName: "_ZN3HHH3cccIZ4mainE5localEEiv", scope: !7, file: !8, line: 5, type: !18, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !9, templateParams: !24, declaration: !23, retainedNodes: !2)
!23 = !DISubprogram(name: "ccc<local>", linkageName: "_ZN3HHH3cccIZ4mainE5localEEiv", scope: !7, file: !8, line: 5, type: !18, scopeLine: 5, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: DISPFlagOptimized, templateParams: !24)
!24 = !{!25}
!25 = !DITemplateTypeParameter(name: "bbb", type: !26)
!26 = !DICompositeType(tag: DW_TAG_structure_type, name: "local", scope: !17, file: !8, line: 11, size: 8, flags: DIFlagFwdDecl)
!27 = distinct !DILocation(line: 12, column: 3, scope: !17)

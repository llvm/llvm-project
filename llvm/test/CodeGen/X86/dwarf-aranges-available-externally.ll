; Ensures that the compiler doesn't assert when DWARF aranges are enabled in the presence of
; `available_externally` constants.

; Generated from the following C++ source:
;
;   template <typename T>
;   struct Baz {
;       static constexpr int boo = int(-1);
;   };
;
;   extern template struct Baz<char>;
;
;   void bar(const int& a);
;
;   void foo() {
;       bar(Baz<char>::boo);
;   }
;
; Compiled with:
;
;     $ clang -cc1 -triple=x86_64-unknown-linux-gnu -debug-info-kind=standalone \
;       -mllvm -generate-arange-section -std=c++17 foo.cpp

; RUN: llc --generate-arange-section < %s

; ModuleID = 'reduced2.cpp'
source_filename = "reduced2.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZN3BazIcE3booE = available_externally constant i32 -1, align 4, !dbg !0

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z3foov() #0 !dbg !17 {
entry:
  call void @_Z3barRKi(ptr noundef nonnull align 4 dereferenceable(4) @_ZN3BazIcE3booE), !dbg !21
  ret void, !dbg !22
}

declare void @_Z3barRKi(ptr noundef nonnull align 4 dereferenceable(4)) #1

attributes #0 = { mustprogress noinline nounwind optnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "boo", linkageName: "_ZN3BazIcE3booE", scope: !2, file: !5, line: 3, type: !6, isLocal: false, isDefinition: true, declaration: !8)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 2f52a868225755ebfa5242992d3a650ac6aadce7)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "<stdin>", directory: "/Users/pcwalton/Downloads")
!4 = !{!0}
!5 = !DIFile(filename: "reduced2.cpp", directory: "/Users/pcwalton/Downloads")
!6 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !7)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIDerivedType(tag: DW_TAG_member, name: "boo", scope: !9, file: !5, line: 3, baseType: !6, flags: DIFlagStaticMember, extraData: i32 -1)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Baz<char>", file: !5, line: 6, size: 8, flags: DIFlagTypePassByValue, elements: !10, templateParams: !11, identifier: "_ZTS3BazIcE")
!10 = !{!8}
!11 = !{!12}
!12 = !DITemplateTypeParameter(name: "T", type: !13)
!13 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git 2f52a868225755ebfa5242992d3a650ac6aadce7)"}
!17 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !5, file: !5, line: 10, type: !18, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
!18 = !DISubroutineType(types: !19)
!19 = !{null}
!20 = !{}
!21 = !DILocation(line: 11, column: 3, scope: !17)
!22 = !DILocation(line: 12, column: 1, scope: !17)

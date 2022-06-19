; RUN: opt < %s -instrprof -debug-info-correlate -S > %t.ll
; RUN: FileCheck < %t.ll --implicit-check-not "{{__llvm_prf_data|__llvm_prf_names}}" %s
; RUN: %llc_dwarf -O0 -filetype=obj < %t.ll | llvm-dwarfdump - | FileCheck --implicit-check-not "{{DW_TAG|NULL}}" %s --check-prefix CHECK-DWARF

; REQUIRES: system-linux, object-emission

@__profn_foo = private constant [3 x i8] c"foo"
; CHECK:      @__profc_foo =
; CHECK-SAME: !dbg ![[EXPR:[0-9]+]]

; CHECK:      ![[EXPR]] = !DIGlobalVariableExpression(var: ![[GLOBAL:[0-9]+]]
; CHECK:      ![[GLOBAL]] = {{.*}} !DIGlobalVariable(name: "__profc_foo"
; CHECK-SAME: scope: ![[SCOPE:[0-9]+]]
; CHECK-SAME: annotations: ![[ANNOTATIONS:[0-9]+]]
; CHECK:      ![[SCOPE]] = {{.*}} !DISubprogram(name: "foo"
; CHECK:      ![[ANNOTATIONS]] = !{![[NAME:[0-9]+]], ![[HASH:[0-9]+]], ![[COUNTERS:[0-9]+]]}
; CHECK:      ![[NAME]] = !{!"Function Name", !"foo"}
; CHECK:      ![[HASH]] = !{!"CFG Hash", i64 12345678}
; CHECK:      ![[COUNTERS]] = !{!"Num Counters", i32 2}

define void @_Z3foov() !dbg !12 {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 12345678, i32 2, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-info-correlate.cpp", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"branch-target-enforcement", i32 0}
!6 = !{i32 8, !"sign-return-address", i32 0}
!7 = !{i32 8, !"sign-return-address-all", i32 0}
!8 = !{i32 8, !"sign-return-address-with-bkey", i32 0}
!9 = !{i32 7, !"uwtable", i32 1}
!10 = !{i32 7, !"frame-pointer", i32 1}
!11 = !{!"clang version 14.0.0"}
!12 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !13, file: !13, line: 1, type: !14, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
!13 = !DIFile(filename: "debug-info-correlate.cpp", directory: "")
!14 = !DISubroutineType(types: !15)
!15 = !{null}
!16 = !{}

; CHECK-DWARF: DW_TAG_compile_unit
; CHECK-DWARF:   DW_TAG_subprogram
; CHECK-DWARF:     DW_AT_name	("foo")
; CHECK-DWARF:     DW_TAG_variable
; CHECK-DWARF:       DW_AT_name	("__profc_foo")
; CHECK-DWARF:       DW_AT_type	({{.*}} "Profile Data Type")
; CHECK-DWARF:       DW_TAG_LLVM_annotation
; CHECK-DWARF:         DW_AT_name	("Function Name")
; CHECK-DWARF:         DW_AT_const_value	("foo")
; CHECK-DWARF:       DW_TAG_LLVM_annotation
; CHECK-DWARF:         DW_AT_name	("CFG Hash")
; CHECK-DWARF:         DW_AT_const_value	(12345678)
; CHECK-DWARF:       DW_TAG_LLVM_annotation
; CHECK-DWARF:         DW_AT_name	("Num Counters")
; CHECK-DWARF:         DW_AT_const_value	(2)
; CHECK-DWARF:       NULL
; CHECK-DWARF:     NULL
; CHECK-DWARF:   DW_TAG_unspecified_type
; CHECK-DWARF:   NULL

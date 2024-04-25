;; Roundtrip tests.

;; Load RemoveDIs mode in llvm-dis but write out debug intrinsics.
; RUN: llvm-as --write-experimental-debuginfo-iterators-to-bitcode=true %s -o - \
; RUN: | llvm-dis --load-bitcode-into-experimental-debuginfo-iterators=true --write-experimental-debuginfo=false \
; RUN: | FileCheck %s

;; Load and write RemoveDIs mode in llvm-dis.
; RUN: llvm-as --write-experimental-debuginfo-iterators-to-bitcode=true %s -o - \
; RUN: | llvm-dis --load-bitcode-into-experimental-debuginfo-iterators=true --write-experimental-debuginfo=true \
; RUN: | FileCheck %s --check-prefixes=RECORDS

;; Load intrinsics directly into the new format (auto-upgrade).
; RUN: llvm-as --write-experimental-debuginfo-iterators-to-bitcode=false %s -o - \
; RUN: | llvm-dis --load-bitcode-into-experimental-debuginfo-iterators=true --write-experimental-debuginfo=true \
; RUN: | FileCheck %s --check-prefixes=RECORDS

;; When preserving, we should output the format the bitcode was written in
;; regardless of the value of the write flag.
; RUN: llvm-as --write-experimental-debuginfo-iterators-to-bitcode=true %s -o - \
; RUN: | llvm-dis --preserve-input-debuginfo-format=true --write-experimental-debuginfo=false \
; RUN: | FileCheck %s --check-prefixes=RECORDS

; RUN: llvm-as --write-experimental-debuginfo-iterators-to-bitcode=false %s -o - \
; RUN: | llvm-dis --preserve-input-debuginfo-format=true --write-experimental-debuginfo=true \
; RUN: | FileCheck %s

;; Check that verify-uselistorder passes regardless of input format.
; RUN: llvm-as %s --write-experimental-debuginfo-iterators-to-bitcode=true -o - | verify-uselistorder
; RUN: verify-uselistorder %s

;; Confirm we're producing RemoveDI records from various tools.
; RUN: opt %s -o - --write-experimental-debuginfo-iterators-to-bitcode=true | llvm-bcanalyzer - | FileCheck %s --check-prefix=BITCODE
; RUN: llvm-as %s -o - --write-experimental-debuginfo-iterators-to-bitcode=true | llvm-bcanalyzer - | FileCheck %s --check-prefix=BITCODE
; BITCODE-DAG: DEBUG_RECORD_LABEL
; BITCODE-DAG: DEBUG_RECORD_VALUE
; BITCODE-DAG: DEBUG_RECORD_ASSIGN
; BITCODE-DAG: DEBUG_RECORD_DECLARE

;; Check that llvm-link doesn't explode if we give it different formats to
;; link.
;; NOTE: This test fails intermittently on linux if the llvm-as output is piped
;; into llvm-link in the RUN lines below, unless the verify-uselistorder RUN
;; lines above are removed. Write to a temporary file to avoid that weirdness.
;; NOTE2: Unfortunately, the above only stopped it occuring on my machine.
;; It failed again intermittently here:
;; https://lab.llvm.org/buildbot/#/builders/245/builds/21930
;; Allow this test to fail-over twice, until this strangeness is understood.
; ALLOW_RETRIES: 2
; RUN: llvm-as %s --experimental-debuginfo-iterators=true --write-experimental-debuginfo-iterators-to-bitcode=true -o %t
; RUN: llvm-link %t %s --experimental-debuginfo-iterators=false -o /dev/null
; RUN: llvm-as %s --experimental-debuginfo-iterators=false -o %t
; RUN: llvm-link %t %s --experimental-debuginfo-iterators=true

;; Checks inline.

@g = internal dso_local global i32 0, align 4, !dbg !0

define internal dso_local noundef i32 @_Z3funv(i32 %p, ptr %storage) !dbg !13 {
entry:
;; Dbg record at top of block, check dbg.value configurations.
; CHECK: entry:
; CHECK-NEXT: dbg.value(metadata i32 %p, metadata ![[e:[0-9]+]], metadata !DIExpression()), !dbg ![[dbg:[0-9]+]]
; CHECK-NEXT: dbg.value(metadata ![[empty:[0-9]+]], metadata ![[e]], metadata !DIExpression()), !dbg ![[dbg]]
; CHECK-NEXT: dbg.value(metadata i32 poison, metadata ![[e]], metadata !DIExpression()), !dbg ![[dbg]]
; CHECK-NEXT: dbg.value(metadata i32 1, metadata ![[f:[0-9]+]], metadata !DIExpression()), !dbg ![[dbg]]
; RECORDS: entry:
; RECORDS-NEXT: dbg_value(i32 %p, ![[e:[0-9]+]], !DIExpression(), ![[dbg:[0-9]+]])
; RECORDS-NEXT: dbg_value(![[empty:[0-9]+]], ![[e]], !DIExpression(), ![[dbg]])
; RECORDS-NEXT: dbg_value(i32 poison, ![[e]], !DIExpression(), ![[dbg]])
; RECORDS-NEXT: dbg_value(i32 1, ![[f:[0-9]+]], !DIExpression(), ![[dbg]])
  tail call void @llvm.dbg.value(metadata i32 %p, metadata !32, metadata !DIExpression()), !dbg !19
  tail call void @llvm.dbg.value(metadata !29, metadata !32, metadata !DIExpression()), !dbg !19
  tail call void @llvm.dbg.value(metadata i32 poison, metadata !32, metadata !DIExpression()), !dbg !19
  tail call void @llvm.dbg.value(metadata i32 1, metadata !33, metadata !DIExpression()), !dbg !19
;; Arglist with an argument, constant, local use before def, poison.
; CHECK-NEXT: dbg.value(metadata !DIArgList(i32 %p, i32 0, i32 %0, i32 poison), metadata ![[f]], metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_LLVM_arg, 2, DW_OP_LLVM_arg, 3, DW_OP_plus, DW_OP_minus)), !dbg ![[dbg]]
; RECORDS-NEXT: dbg_value(!DIArgList(i32 %p, i32 0, i32 %0, i32 poison), ![[f]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_LLVM_arg, 2, DW_OP_LLVM_arg, 3, DW_OP_plus, DW_OP_minus), ![[dbg]])
  tail call void @llvm.dbg.value(metadata !DIArgList(i32 %p, i32 0, i32 %0, i32 poison), metadata !33, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_LLVM_arg, 2, DW_OP_LLVM_arg, 3, DW_OP_plus, DW_OP_minus)), !dbg !19
;; Check dbg.assign use before def (value, addr and ID). Check expression order too.
; CHECK: dbg.assign(metadata i32 %0, metadata ![[i:[0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 0),
; CHECK-SAME:       metadata ![[ID:[0-9]+]], metadata ptr %a, metadata !DIExpression(DW_OP_plus_uconst, 1)), !dbg ![[dbg]]
; RECORDS: dbg_assign(i32 %0, ![[i:[0-9]+]], !DIExpression(DW_OP_plus_uconst, 0),
; RECORDS-SAME:       ![[ID:[0-9]+]], ptr %a, !DIExpression(DW_OP_plus_uconst, 1), ![[dbg]])
  tail call void @llvm.dbg.assign(metadata i32 %0, metadata !36, metadata !DIExpression(DW_OP_plus_uconst, 0), metadata !37, metadata ptr %a, metadata !DIExpression(DW_OP_plus_uconst, 1)), !dbg !19
  %a = alloca i32, align 4, !DIAssignID !37
; CHECK: %a = alloca i32, align 4, !DIAssignID ![[ID]]
;; Check dbg.declare configurations.
; CHECK-NEXT: dbg.declare(metadata ptr %a, metadata ![[a:[0-9]+]], metadata !DIExpression()), !dbg ![[dbg]]
; CHECK-NEXT: dbg.declare(metadata ![[empty:[0-9]+]], metadata ![[b:[0-9]+]], metadata !DIExpression()), !dbg ![[dbg]]
; CHECK-NEXT: dbg.declare(metadata ptr poison, metadata ![[c:[0-9]+]], metadata !DIExpression()), !dbg ![[dbg]]
; CHECK-NEXT: dbg.declare(metadata ptr null, metadata ![[d:[0-9]+]], metadata !DIExpression()), !dbg ![[dbg]]
; CHECK-NEXT: dbg.declare(metadata ptr @g, metadata ![[h:[0-9]+]], metadata !DIExpression()), !dbg ![[dbg]]
; RECORDS: %a = alloca i32, align 4, !DIAssignID ![[ID]]
;; Check dbg.declare configurations.
; RECORDS-NEXT: dbg_declare(ptr %a, ![[a:[0-9]+]], !DIExpression(), ![[dbg]])
; RECORDS-NEXT: dbg_declare(![[empty:[0-9]+]], ![[b:[0-9]+]], !DIExpression(), ![[dbg]])
; RECORDS-NEXT: dbg_declare(ptr poison, ![[c:[0-9]+]], !DIExpression(), ![[dbg]])
; RECORDS-NEXT: dbg_declare(ptr null, ![[d:[0-9]+]], !DIExpression(), ![[dbg]])
; RECORDS-NEXT: dbg_declare(ptr @g, ![[h:[0-9]+]], !DIExpression(), ![[dbg]])
  tail call void @llvm.dbg.declare(metadata ptr %a, metadata !17, metadata !DIExpression()), !dbg !19
  tail call void @llvm.dbg.declare(metadata !29, metadata !28, metadata !DIExpression()), !dbg !19
  tail call void @llvm.dbg.declare(metadata ptr poison, metadata !30, metadata !DIExpression()), !dbg !19
  tail call void @llvm.dbg.declare(metadata ptr null, metadata !31, metadata !DIExpression()), !dbg !19
  tail call void @llvm.dbg.declare(metadata ptr @g, metadata !35, metadata !DIExpression()), !dbg !19
;; Argument value dbg.declare.
; CHECK: dbg.declare(metadata ptr %storage, metadata ![[g:[0-9]+]], metadata !DIExpression()), !dbg ![[dbg]]
; RECORDS: dbg_declare(ptr %storage, ![[g:[0-9]+]], !DIExpression(), ![[dbg]])
  tail call void @llvm.dbg.declare(metadata ptr %storage, metadata !34, metadata !DIExpression()), !dbg !19
;; Use before def dbg.value.
; CHECK: dbg.value(metadata i32 %0, metadata ![[e]], metadata !DIExpression()), !dbg ![[dbg]]
; RECORDS: dbg_value(i32 %0, ![[e]], !DIExpression(), ![[dbg]])
  tail call void @llvm.dbg.value(metadata i32 %0, metadata !32, metadata !DIExpression()), !dbg !19
  %0 = load i32, ptr @g, align 4, !dbg !20
;; Non-argument local value dbg.value.
; CHECK: dbg.value(metadata i32 %0, metadata ![[e]], metadata !DIExpression()), !dbg ![[dbg]]
; RECORDS: dbg_value(i32 %0, ![[e]], !DIExpression(), ![[dbg]])
  tail call void @llvm.dbg.value(metadata i32 %0, metadata !32, metadata !DIExpression()), !dbg !19
  store i32 %0, ptr %a, align 4, !dbg !19
  %1 = load i32, ptr %a, align 4, !dbg !25
; CHECK: dbg.label(metadata ![[label:[0-9]+]]), !dbg ![[dbg]]
; RECORDS: dbg_label(![[label:[0-9]+]], ![[dbg]])
  tail call void @llvm.dbg.label(metadata !38), !dbg !19
  ret i32 %1, !dbg !27
}

; CHECK-DAG: ![[a]] = !DILocalVariable(name: "a",
; CHECK-DAG: ![[b]] = !DILocalVariable(name: "b",
; CHECK-DAG: ![[c]] = !DILocalVariable(name: "c",
; CHECK-DAG: ![[d]] = !DILocalVariable(name: "d",
; CHECK-DAG: ![[e]] = !DILocalVariable(name: "e",
; CHECK-DAG: ![[f]] = !DILocalVariable(name: "f",
; CHECK-DAG: ![[g]] = !DILocalVariable(name: "g",
; CHECK-DAG: ![[h]] = !DILocalVariable(name: "h",
; CHECK-DAG: ![[i]] = !DILocalVariable(name: "i",
; CHECK-DAG: ![[empty]] = !{}
; CHECK-DAG: ![[label]] = !DILabel

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)
declare void @llvm.dbg.label(metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7, !8, !9, !10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 19.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 8, !"PIC Level", i32 2}
!10 = !{i32 7, !"PIE Level", i32 2}
!11 = !{i32 7, !"uwtable", i32 2}
!12 = !{!"clang version 19.0.0"}
!13 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !3, file: !3, line: 2, type: !14, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !16)
!14 = !DISubroutineType(types: !15)
!15 = !{!5}
!16 = !{!17}
!17 = !DILocalVariable(name: "a", scope: !13, file: !3, line: 3, type: !5)
!18 = !DILocation(line: 3, column: 3, scope: !13)
!19 = !DILocation(line: 3, column: 7, scope: !13)
!20 = !DILocation(line: 3, column: 11, scope: !13)
!25 = !DILocation(line: 4, column: 12, scope: !13)
!26 = !DILocation(line: 5, column: 1, scope: !13)
!27 = !DILocation(line: 4, column: 5, scope: !13)
!28 = !DILocalVariable(name: "b", scope: !13, file: !3, line: 3, type: !5)
!29 = !{}
!30 = !DILocalVariable(name: "c", scope: !13, file: !3, line: 3, type: !5)
!31 = !DILocalVariable(name: "d", scope: !13, file: !3, line: 3, type: !5)
!32 = !DILocalVariable(name: "e", scope: !13, file: !3, line: 3, type: !5)
!33 = !DILocalVariable(name: "f", scope: !13, file: !3, line: 3, type: !5)
!34 = !DILocalVariable(name: "g", scope: !13, file: !3, line: 3, type: !5)
!35 = !DILocalVariable(name: "h", scope: !13, file: !3, line: 3, type: !5)
!36 = !DILocalVariable(name: "i", scope: !13, file: !3, line: 3, type: !5)
!37 = distinct !DIAssignID()
!38 = !DILabel(scope: !13, name: "label", file: !3, line: 1)

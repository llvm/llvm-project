; XFAIL: target={{.*}}-aix{{.*}}
; XFAIL: *
; RUN: %llc_dwarf -accel-tables=Apple -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump -apple-names %t | FileCheck %s --check-prefix=NUM_HASHES
; RUN: llvm-dwarfdump  --find=bb --find=cA %t | FileCheck %s --check-prefix=FOUND_VARS
; RUN: llvm-dwarfdump  --find-all-apple %t | FileCheck %s --check-prefix=ALL_ENTRIES


; The strings 'bb' and 'cA' hash to the same value under the Apple accelerator
; table hashing algorithm.
; We first test that there is exactly one bucket and one hash.
; Then we check that both values are found.

; NUM_HASHES:      Bucket count: 2
; NUM_HASHES-NEXT: Hashes count: 2
; FOUND_VARS: DW_AT_name        ("bb")
; FOUND_VARS: DW_AT_name        ("cA")

; ALL_ENTRIES: Apple accelerator entries with name = "bb":
; ALL_ENTRIES: DW_AT_name        ("bb")
; ALL_ENTRIES: Apple accelerator entries with name = "cA":
; ALL_ENTRIES: DW_AT_name        ("cA")
; ALL_ENTRIES: Apple accelerator entries with name = "some_other_hash":
; ALL_ENTRIES: DW_AT_name        ("some_other_hash")
; ALL_ENTRIES: Apple accelerator entries with name = "int":
; ALL_ENTRIES: DW_AT_name        ("int")

@bb = global i32 200, align 4, !dbg !0
@cA = global i32 10, align 4, !dbg !5
@some_other_hash = global i32 10, !dbg !17

!llvm.module.flags = !{!9, !10, !11, !12, !13}
!llvm.dbg.cu = !{!2}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "bb", scope: !2, file: !3, line: 1, type: !7, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "", emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "test.cpp", directory: "blah")
!4 = !{!0, !5, !17}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "cA", scope: !2, file: !3, line: 2, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{i32 8, !"PIC Level", i32 2}
!13 = !{i32 7, !"uwtable", i32 1}
!15 = !{!"blah"}
!16 = distinct !DIGlobalVariable(name: "some_other_hash", scope: !2, file: !3, line: 2, type: !7, isLocal: false, isDefinition: true)
!17 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression())

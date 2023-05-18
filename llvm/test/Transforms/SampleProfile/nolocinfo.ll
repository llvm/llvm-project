; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/nolocinfo.prof -S -pass-remarks=sample-profile 2>&1 | FileCheck %s

define i32 @foo(i32 %i) #0 !dbg !4 {
entry:
  %i.addr = alloca i32, align 4
  %0 = load i32, ptr %i.addr, align 4
  %cmp = icmp sgt i32 %0, 1000

; Remarks for conditional branches need debug location information for the
; referring branch. When that is not present, the compiler should not abort.
;
; CHECK: remark: nolocinfo.c:3:5: most popular destination for conditional branches at <UNKNOWN LOCATION>
  br i1 %cmp, label %if.then, label %if.end

if.then:
  ret i32 0, !dbg !18

if.end:
  ret i32 1
}

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 251335) (llvm/trunk 251344)", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "nolocinfo.c", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.8.0 (trunk 251335) (llvm/trunk 251344)"}
!15 = distinct !DILexicalBlock(scope: !4, file: !1, line: 2, column: 7)
!18 = !DILocation(line: 3, column: 5, scope: !15)

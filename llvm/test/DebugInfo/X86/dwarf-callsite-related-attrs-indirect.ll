; $ clang -O2 -S -emit-llvm indir.c -gdwarf-5
; __attribute__((disable_tail_calls)) void call_reg(void (*f)()) { f(); }
; __attribute__((disable_tail_calls)) void call_mem(void (**f)()) { (*f)(); }

; RUN: llc -mtriple=x86_64 -debugger-tune=lldb < %s -filetype=obj -o %t.o
; RUN: llvm-dwarfdump %t.o -o - | FileCheck %s -check-prefix=OBJ -implicit-check-not=DW_TAG_call_site -implicit-check-not=DW_AT_call_target
; RUN: llvm-dwarfdump -verify %t.o 2>&1 | FileCheck %s -check-prefix=VERIFY
; RUN: llvm-dwarfdump -statistics %t.o | FileCheck %s -check-prefix=STATS

; VERIFY: No errors.
; STATS: "#call site DIEs": 1,

; OBJ: DW_TAG_subprogram
; OBJ:   DW_AT_name ("call_reg")
; Function Attrs: nounwind uwtable
define dso_local void @call_reg(ptr noundef readonly captures(none) %f) local_unnamed_addr #0 !dbg !10 {
entry:
    #dbg_value(ptr %f, !17, !DIExpression(), !18)

; OBJ:   DW_TAG_call_site
; OBJ:     DW_AT_call_target
; OBJ:     DW_AT_call_return_pc
  call void (...) %f() #1, !dbg !19
  ret void, !dbg !20
}

; OBJ: DW_TAG_subprogram
; OBJ:   DW_AT_name ("call_mem")
; Function Attrs: nounwind uwtable
define dso_local void @call_mem(ptr noundef readonly captures(none) %f) local_unnamed_addr #0 !dbg !21 {
entry:
    #dbg_value(ptr %f, !26, !DIExpression(), !27)
  %0 = load ptr, ptr %f, align 8, !dbg !28, !tbaa !29
  call void (...) %0() #1, !dbg !28
  ret void, !dbg !33
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="true" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 22.0.0git (https://github.com/llvm/llvm-project 74e4a8645da91247dc8dc502771c2cc4d46f1f91)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "indir.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "4a7538b13e2edbec44f43ed5154be38c")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 22.0.0git (https://github.com/llvm/llvm-project 74e4a8645da91247dc8dc502771c2cc4d46f1f91)"}
!10 = distinct !DISubprogram(name: "call_reg", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DISubroutineType(types: !15)
!15 = !{null, null}
!16 = !{!17}
!17 = !DILocalVariable(name: "f", arg: 1, scope: !10, file: !1, line: 1, type: !13)
!18 = !DILocation(line: 0, scope: !10)
!19 = !DILocation(line: 1, column: 66, scope: !10)
!20 = !DILocation(line: 1, column: 71, scope: !10)
!21 = distinct !DISubprogram(name: "call_mem", scope: !1, file: !1, line: 2, type: !22, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !25)
!22 = !DISubroutineType(types: !23)
!23 = !{null, !24}
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!25 = !{!26}
!26 = !DILocalVariable(name: "f", arg: 1, scope: !21, file: !1, line: 2, type: !24)
!27 = !DILocation(line: 0, scope: !21)
!28 = !DILocation(line: 2, column: 67, scope: !21)
!29 = !{!30, !30, i64 0}
!30 = !{!"any pointer", !31, i64 0}
!31 = !{!"omnipotent char", !32, i64 0}
!32 = !{!"Simple C/C++ TBAA"}
!33 = !DILocation(line: 2, column: 75, scope: !21)

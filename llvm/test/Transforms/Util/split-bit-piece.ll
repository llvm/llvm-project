; Checks that llvm.dbg.declare -> llvm.dbg.value conversion utility
; (here exposed through the SROA) pass, properly inserts bit_piece expressions
; if it only describes part of the variable.
; RUN: opt -S -sroa %s | FileCheck %s

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; Function Attrs: nounwind uwtable
define hidden void @_ZN6__tsan9FastState14SetHistorySizeEi(i32 %hs) #1 align 2 {
entry:
  %hs.addr = alloca i32, align 4
  %v1 = alloca i64, align 8
  %v2 = alloca i64, align 8
  store i32 %hs, i32* %hs.addr, align 4
; CHECK: call void @llvm.dbg.value(metadata i32 %hs, i64 0, metadata !{{[0-9]+}}, metadata ![[EXPR:[0-9]+]])
; CHECK: ![[EXPR]] = !DIExpression(DW_OP_LLVM_fragment, 0
  call void @llvm.dbg.declare(metadata i64* %v1, metadata !9, metadata !12), !dbg !13
  %0 = load i32, i32* %hs.addr, align 4
  %conv = sext i32 %0 to i64
  store i64 %conv, i64* %v1, align 8
  %1 = load i64, i64* %v2, align 8
  unreachable
}

attributes #0 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 256979) (llvm/trunk 257107)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2)
!1 = !DIFile(filename: "tsan_shadow_test.cc", directory: "/tmp")
!2 = !{!3, !5}
!3 = !DICompositeType(tag: DW_TAG_class_type, name: "FastState", file: !4, line: 91, size: 64, align: 64, identifier: "_ZTSN6__tsan9FastStateE")
!4 = !DIFile(filename: "/mnt/extra/llvm/projects/compiler-rt/lib/tsan/rtl/tsan_rtl.h", directory: "/tmp")
!5 = distinct !DIDerivedType(tag: DW_TAG_typedef, name: "u64", line: 78, baseType: !6)
!6 = !DIBasicType(name: "long long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"clang version 3.8.0 (trunk 256979) (llvm/trunk 257107)"}
!9 = !DILocalVariable(name: "v1", scope: !10, file: !4, line: 136, type: !5)
!10 = distinct !DILexicalBlock(scope: !11, file: !4, line: 136, column: 5)
!11 = distinct !DISubprogram(name: "SetHistorySize", linkageName: "_ZN6__tsan9FastState14SetHistorySizeEi", scope: !3, file: !4, line: 135, isLocal: false, isDefinition: true, scopeLine: 135, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!12 = !DIExpression()
!13 = !DILocation(line: 136, column: 5, scope: !10)

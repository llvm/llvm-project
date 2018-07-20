; RUN: opt < %s -mem2reg -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local x86_fp80 @powixf2() !dbg !1 {
entry:
  %r = alloca x86_fp80, align 16
  call void @llvm.dbg.declare(metadata x86_fp80* %r, metadata !14, metadata !DIExpression()), !dbg !15
  br i1 undef, label %if.then, label %if.end, !dbg !16

if.then:                                          ; preds = %entry
; CHECK-LABEL: if.then:
; CHECK: %mul = fmul x86_fp80
; CHECK: call void @llvm.dbg.value(metadata x86_fp80 %mul, metadata {{.*}}, metadata !DIExpression())
  %mul = fmul x86_fp80 undef, undef, !dbg !18
  store x86_fp80 %mul, x86_fp80* %r, align 16, !dbg !18
  br label %if.end, !dbg !20

if.end:                                           ; preds = %if.then, %entry
; CHECK-LABEL: if.end:
; CHECK: %r.0 = phi x86_fp80
; CHECK: call void @llvm.dbg.value(metadata x86_fp80 %r.0, metadata {{.*}}, metadata !DIExpression())
  %out = load x86_fp80, x86_fp80* %r, align 16, !dbg !21
  ret x86_fp80 %out, !dbg !22
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{}
!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DISubprogram(name: "__powixf2", scope: !2, file: !2, line: 22, type: !3, isLocal: false, isDefinition: true, scopeLine: 23, flags: DIFlagPrototyped, isOptimized: true, unit: !11, retainedNodes: !13)
!2 = !DIFile(filename: "powixf2.c", directory: "")
!3 = !DISubroutineType(types: !4)
!4 = !{!5, !5, !6}
!5 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "si_int", file: !7, line: 28, baseType: !8)
!7 = !DIFile(filename: "int_types.h", directory: "")
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "int32_t", file: !9, line: 39, baseType: !10)
!9 = !DIFile(filename: "/usr/include/stdint.h", directory: "")
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 7.0.0 () ()", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !12)
!12 = !{}
!13 = !{!14}
!14 = !DILocalVariable(name: "r", scope: !1, file: !2, line: 25, type: !5)
!15 = !DILocation(line: 25, column: 17, scope: !1)
!16 = !DILocation(line: 28, column: 13, scope: !17)
!17 = distinct !DILexicalBlock(scope: !1, file: !2, line: 27, column: 5)
!18 = !DILocation(line: 29, column: 15, scope: !19)
!19 = distinct !DILexicalBlock(scope: !17, file: !2, line: 28, column: 13)
!20 = !DILocation(line: 29, column: 13, scope: !19)
!21 = !DILocation(line: 35, column: 22, scope: !1)
!22 = !DILocation(line: 35, column: 5, scope: !1)

; RUN: llc -O2 < %s | FileCheck %s
; RUN: llc -O2 -regalloc=basic < %s | FileCheck %s
source_filename = "test/CodeGen/X86/2010-05-26-DotDebugLoc.ll"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10"

%struct.a = type { i32, %struct.a* }

@llvm.used = appending global [1 x i8*] [i8* bitcast (i8* (%struct.a*)* @bar to i8*)], section "llvm.metadata"

; Function Attrs: noinline nounwind optsize ssp
define i8* @bar(%struct.a* %myvar) #0 !dbg !8 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.a* %myvar, i64 0, metadata !18, metadata !19), !dbg !20
  %0 = getelementptr inbounds %struct.a, %struct.a* %myvar, i64 0, i32 0, !dbg !21
  %1 = load i32, i32* %0, align 8, !dbg !21
  tail call void @foo(i32 %1) #0, !dbg !21
  %2 = bitcast %struct.a* %myvar to i8*, !dbg !23
  ret i8* %2, !dbg !23
}

; Function Attrs: noinline nounwind optsize ssp
declare void @foo(i32) #0

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { noinline nounwind optsize ssp }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C89, file: !1, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "foo.c", directory: "/tmp/")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = !DIGlobalVariable(name: "ret", scope: !1, file: !1, line: 7, type: !6, isLocal: false, isDefinition: true)
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !{i32 1, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "bar", linkageName: "bar", scope: !1, file: !1, line: 17, type: !9, isLocal: false, isDefinition: true, scopeLine: 17, virtualIndex: 6, isOptimized: true, unit: !0, retainedNodes: !17)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !12}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, scope: !1, file: !1, baseType: null, size: 64, align: 64)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, scope: !1, file: !1, baseType: !13, size: 64, align: 64)
!13 = !DICompositeType(tag: DW_TAG_structure_type, name: "a", scope: !1, file: !1, line: 2, size: 128, align: 64, elements: !14)
!14 = !{!15, !16}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !13, file: !1, line: 3, baseType: !6, size: 32, align: 32)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !13, file: !1, line: 4, baseType: !12, size: 64, align: 64, offset: 64)
!17 = !{!18}
!18 = !DILocalVariable(name: "myvar", arg: 1, scope: !8, file: !1, line: 17, type: !12)
!19 = !DIExpression()
!20 = !DILocation(line: 0, scope: !8)
!21 = !DILocation(line: 18, scope: !22)
!22 = distinct !DILexicalBlock(scope: !8, file: !1, line: 17)
!23 = !DILocation(line: 19, scope: !22)

; The variable bar:myvar changes registers after the first movq.
; It is cobbered by popq %rbx
; CHECK: movq
; CHECK-NEXT: [[LABEL:Ltmp[0-9]*]]
; CHECK: .loc	1 19 0
; CHECK: popq
; CHECK-NEXT: [[CLOBBER:Ltmp[0-9]*]]

; CHECK: Ldebug_loc0:
; CHECK-NEXT: .set [[SET1:.*]], Lfunc_begin0-Lfunc_begin0
; CHECK-NEXT: .quad   [[SET1]]
; CHECK-NEXT: .set [[SET2:.*]], [[LABEL]]-Lfunc_begin0
; CHECK-NEXT: .quad   [[SET2]]
; CHECK-NEXT: .short  1     ## Loc expr size
; CHECK-NEXT: .byte   85
; CHECK-NEXT: .set [[SET3:.*]], [[LABEL]]-Lfunc_begin0
; CHECK-NEXT: .quad   [[SET3]]
; CHECK-NEXT: .set [[SET4:.*]], [[CLOBBER]]-Lfunc_begin0
; CHECK-NEXT: .quad   [[SET4]]
; CHECK-NEXT: .short  1     ## Loc expr size
; CHECK-NEXT: .byte   83

; RUN: llc < %s -mcpu=generic -mtriple=i386-apple-darwin -no-integrated-as

source_filename = "test/CodeGen/X86/fpstack-debuginstr-kill.ll"

@g1 = global double 0.000000e+00, align 8, !dbg !0
@g2 = global i32 0, align 4, !dbg !7

define void @_Z16fpuop_arithmeticjj(i32, i32) !dbg !16 {
entry:
  switch i32 undef, label %sw.bb.i1921 [
  ]

sw.bb261:                                         ; No predecessors!
  unreachable

sw.bb.i1921:                                      ; preds = %entry
  switch i32 undef, label %if.end511 [
    i32 1, label %sw.bb27.i
  ]

sw.bb27.i:                                        ; preds = %sw.bb.i1921
  %conv.i.i1923 = fpext float undef to x86_fp80
  br label %if.end511

if.end511:                                        ; preds = %sw.bb27.i, %sw.bb.i1921
  %src.sroa.0.0.src.sroa.0.0.2280 = phi x86_fp80 [ %conv.i.i1923, %sw.bb27.i ], [ undef, %sw.bb.i1921 ]
  switch i32 undef, label %sw.bb992 [
    i32 3, label %sw.bb735
    i32 18, label %if.end41.i2210
  ]

sw.bb735:                                         ; preds = %if.end511
  %2 = call x86_fp80 asm sideeffect "frndint", "={st},0,~{dirflag},~{fpsr},~{flags}"(x86_fp80 %src.sroa.0.0.src.sroa.0.0.2280)
  unreachable

if.end41.i2210:                                   ; preds = %if.end511
  call void @llvm.dbg.value(metadata x86_fp80 %src.sroa.0.0.src.sroa.0.0.2280, i64 0, metadata !25, metadata !26), !dbg !27
  unreachable

sw.bb992:                                         ; preds = %if.end511
  ret void
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #0

attributes #0 = { nounwind readnone }

!llvm.dbg.cu = !{!10}
!llvm.module.flags = !{!14, !15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "g1", scope: null, file: !2, line: 5, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "f1.cpp", directory: "x87stackifier")
!3 = !DIDerivedType(tag: DW_TAG_typedef, name: "fpu_extended", file: !2, line: 3, baseType: !4)
!4 = !DIDerivedType(tag: DW_TAG_typedef, name: "fpu_register", file: !2, line: 2, baseType: !5)
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "uae_f64", file: !2, line: 1, baseType: !6)
!6 = !DIBasicType(name: "long double", size: 128, align: 128, encoding: DW_ATE_float)
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = !DIGlobalVariable(name: "g2", scope: null, file: !2, line: 6, type: !9, isLocal: false, isDefinition: true)
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !11, producer: "clang version 3.6.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !12, retainedTypes: !12, globals: !13, imports: !12)
!11 = !DIFile(filename: "fpu_ieee.cpp", directory: "x87stackifier")
!12 = !{}
!13 = !{!0, !7}
!14 = !{i32 2, !"Dwarf Version", i32 2}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = distinct !DISubprogram(name: "fpuop_arithmetic", linkageName: "_Z16fpuop_arithmeticjj", scope: !2, file: !2, line: 11, type: !17, isLocal: false, isDefinition: true, scopeLine: 13, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !10, retainedNodes: !20)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19, !19}
!19 = !DIBasicType(name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!20 = !{!21, !22, !23, !24, !25}
!21 = !DILocalVariable(arg: 1, scope: !16, file: !2, line: 11, type: !19)
!22 = !DILocalVariable(arg: 2, scope: !16, file: !2, line: 11, type: !19)
!23 = !DILocalVariable(name: "x", scope: !16, file: !2, line: 14, type: !3)
!24 = !DILocalVariable(name: "a", scope: !16, file: !2, line: 15, type: !9)
!25 = !DILocalVariable(name: "value", scope: !16, file: !2, line: 16, type: !3)
!26 = !DIExpression()
!27 = !DILocation(line: 0, scope: !16)


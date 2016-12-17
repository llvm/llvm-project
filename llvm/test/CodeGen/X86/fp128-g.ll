; RUN: llc < %s -O2 -mtriple=x86_64-linux-android -mattr=+mmx | FileCheck %s --check-prefix=X64
;
; These cases check if x86_64-linux-android works with -O2 -g,
; especially CSE matching needed by SoftenFloatRes_LOAD.
; Multiple common load patterns are included to have better coverage.
; When CSE matching fails, SoftenFloatResult and SoftenFloatRes_LOAD
; can be called in an infinite loop.

; ModuleID = 'fp128-g.c'
source_filename = "fp128-g.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux-android"

@ld_ptr = common local_unnamed_addr global fp128* null, align 8, !dbg !0

; Function Attrs: nounwind readonly uwtable
define fp128 @test_return1(fp128* nocapture readonly %ptr) local_unnamed_addr #0 !dbg !11 {
entry:
  tail call void @llvm.dbg.value(metadata fp128* %ptr, i64 0, metadata !15, metadata !16), !dbg !17
  %0 = load fp128, fp128* %ptr, align 16, !dbg !18, !tbaa !19
  ret fp128 %0, !dbg !23
; X64-LABEL: test_return1:
; X64:       .loc
; X64:       movaps     (%rdi), %xmm0
; X64:       .loc
; X64:       retq
}

; Function Attrs: nounwind readonly uwtable
define fp128 @test_return2(fp128* nocapture readonly %ptr) local_unnamed_addr #0 !dbg !24 {
entry:
  tail call void @llvm.dbg.value(metadata fp128* %ptr, i64 0, metadata !26, metadata !16), !dbg !28
  %0 = load fp128, fp128* %ptr, align 16, !dbg !29, !tbaa !19
  tail call void @llvm.dbg.value(metadata fp128 %0, i64 0, metadata !27, metadata !16), !dbg !30
  ret fp128 %0, !dbg !31
; X64-LABEL: test_return2:
; X64:       .loc
; X64:       movaps     (%rdi), %xmm0
; X64:       .loc
; X64:       retq
}

; Function Attrs: nounwind readonly uwtable
define fp128 @test_return3(fp128* nocapture readonly %ptr) local_unnamed_addr #0 !dbg !32 {
entry:
  tail call void @llvm.dbg.value(metadata fp128* %ptr, i64 0, metadata !34, metadata !16), !dbg !35
  %0 = load fp128, fp128* %ptr, align 16, !dbg !36, !tbaa !19
  %add = fadd fp128 %0, %0, !dbg !37
  ret fp128 %add, !dbg !38
; X64-LABEL: test_return3:
; X64:       .loc
; X64:       movaps     (%rdi), %xmm0
; X64:       .loc
; X64:       movaps	%xmm0, %xmm1
; X64:       callq	__addtf3
; X64:       .loc
; X64:       retq
}

; Function Attrs: norecurse nounwind readonly uwtable
define fp128 @test_return4() local_unnamed_addr #1 !dbg !39 {
entry:
  %0 = load fp128*, fp128** @ld_ptr, align 8, !dbg !42, !tbaa !43
  %1 = load fp128, fp128* %0, align 16, !dbg !45, !tbaa !19
  ret fp128 %1, !dbg !46
; X64-LABEL: test_return4:
; X64:       .loc
; X64:       movq	ld_ptr(%rip), %rax
; X64:       .loc
; X64:       movaps	(%rax), %xmm0
; X64:       .loc
; X64:       retq
}

; Function Attrs: nounwind readonly uwtable
define fp128 @test_return5() local_unnamed_addr #0 !dbg !47 {
entry:
  %0 = load fp128*, fp128** @ld_ptr, align 8, !dbg !50, !tbaa !43
  %1 = load fp128, fp128* %0, align 16, !dbg !51, !tbaa !19
  tail call void @llvm.dbg.value(metadata fp128 %1, i64 0, metadata !49, metadata !16), !dbg !52
  ret fp128 %1, !dbg !53
; X64-LABEL: test_return5:
; X64:       .loc
; X64:       movq	ld_ptr(%rip), %rax
; X64:       .loc
; X64:       movaps	(%rax), %xmm0
; X64:       .loc
; X64:       retq
}

; Function Attrs: norecurse nounwind readonly uwtable
define fp128 @test_return6() local_unnamed_addr #1 !dbg !54 {
entry:
  %0 = load fp128*, fp128** @ld_ptr, align 8, !dbg !55, !tbaa !43
  %1 = load fp128, fp128* %0, align 16, !dbg !56, !tbaa !19
  %add = fadd fp128 %1, %1, !dbg !57
  ret fp128 %add, !dbg !58
; X64-LABEL: test_return6:
; X64:       .loc
; X64:       movaps	(%rax), %xmm0
; X64:       .loc
; X64:       movaps	%xmm0, %xmm1
; X64:       callq	__addtf3
; X64:       .loc
; X64:       retq
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DIGlobalVariable(name: "ld_ptr", scope: !1, file: !2, line: 17, type: !5, isLocal: false, isDefinition: true)
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 4.0.0 (trunk 281495)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, globals: !4)
!2 = !DIFile(filename: "fp128-g.c", directory: "/disk5/chh/Debug/ld.loop")
!3 = !{}
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, align: 64)
!6 = !DIBasicType(name: "long double", size: 128, align: 128, encoding: DW_ATE_float)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"PIC Level", i32 2}
!10 = !{!"clang version 4.0.0 (trunk 281495)"}
!11 = distinct !DISubprogram(name: "test_return1", scope: !2, file: !2, line: 3, type: !12, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !1, variables: !14)
!12 = !DISubroutineType(types: !13)
!13 = !{!6, !5}
!14 = !{!15}
!15 = !DILocalVariable(name: "ptr", arg: 1, scope: !11, file: !2, line: 3, type: !5)
!16 = !DIExpression()
!17 = !DILocation(line: 3, column: 39, scope: !11)
!18 = !DILocation(line: 4, column: 12, scope: !11)
!19 = !{!20, !20, i64 0}
!20 = !{!"long double", !21, i64 0}
!21 = !{!"omnipotent char", !22, i64 0}
!22 = !{!"Simple C/C++ TBAA"}
!23 = !DILocation(line: 4, column: 5, scope: !11)
!24 = distinct !DISubprogram(name: "test_return2", scope: !2, file: !2, line: 7, type: !12, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: true, unit: !1, variables: !25)
!25 = !{!26, !27}
!26 = !DILocalVariable(name: "ptr", arg: 1, scope: !24, file: !2, line: 7, type: !5)
!27 = !DILocalVariable(name: "value", scope: !24, file: !2, line: 8, type: !6)
!28 = !DILocation(line: 7, column: 39, scope: !24)
!29 = !DILocation(line: 9, column: 14, scope: !24)
!30 = !DILocation(line: 8, column: 17, scope: !24)
!31 = !DILocation(line: 10, column: 5, scope: !24)
!32 = distinct !DISubprogram(name: "test_return3", scope: !2, file: !2, line: 13, type: !12, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: true, unit: !1, variables: !33)
!33 = !{!34}
!34 = !DILocalVariable(name: "ptr", arg: 1, scope: !32, file: !2, line: 13, type: !5)
!35 = !DILocation(line: 13, column: 39, scope: !32)
!36 = !DILocation(line: 14, column: 12, scope: !32)
!37 = !DILocation(line: 14, column: 17, scope: !32)
!38 = !DILocation(line: 14, column: 5, scope: !32)
!39 = distinct !DISubprogram(name: "test_return4", scope: !2, file: !2, line: 18, type: !40, isLocal: false, isDefinition: true, scopeLine: 18, isOptimized: true, unit: !1, variables: !3)
!40 = !DISubroutineType(types: !41)
!41 = !{!6}
!42 = !DILocation(line: 19, column: 13, scope: !39)
!43 = !{!44, !44, i64 0}
!44 = !{!"any pointer", !21, i64 0}
!45 = !DILocation(line: 19, column: 12, scope: !39)
!46 = !DILocation(line: 19, column: 5, scope: !39)
!47 = distinct !DISubprogram(name: "test_return5", scope: !2, file: !2, line: 22, type: !40, isLocal: false, isDefinition: true, scopeLine: 22, isOptimized: true, unit: !1, variables: !48)
!48 = !{!49}
!49 = !DILocalVariable(name: "value", scope: !47, file: !2, line: 23, type: !6)
!50 = !DILocation(line: 23, column: 26, scope: !47)
!51 = !DILocation(line: 23, column: 25, scope: !47)
!52 = !DILocation(line: 23, column: 17, scope: !47)
!53 = !DILocation(line: 24, column: 5, scope: !47)
!54 = distinct !DISubprogram(name: "test_return6", scope: !2, file: !2, line: 27, type: !40, isLocal: false, isDefinition: true, scopeLine: 27, isOptimized: true, unit: !1, variables: !3)
!55 = !DILocation(line: 28, column: 13, scope: !54)
!56 = !DILocation(line: 28, column: 12, scope: !54)
!57 = !DILocation(line: 28, column: 20, scope: !54)
!58 = !DILocation(line: 28, column: 5, scope: !54)

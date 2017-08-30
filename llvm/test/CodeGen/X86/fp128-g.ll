; RUN: llc < %s -O2 -mtriple=x86_64-linux-android -mattr=+mmx \
; RUN:    -enable-legalize-types-checking | FileCheck %s --check-prefix=X64
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
define fp128 @test_return1(fp128* nocapture readonly %ptr) local_unnamed_addr #0 !dbg !12 {
entry:
  tail call void @llvm.dbg.value(metadata fp128* %ptr, i64 0, metadata !16, metadata !17), !dbg !18
  %0 = load fp128, fp128* %ptr, align 16, !dbg !19, !tbaa !20
  ret fp128 %0, !dbg !24
; X64-LABEL: test_return1:
; X64:       .loc
; X64:       movaps     (%rdi), %xmm0
; X64:       .loc
; X64:       retq
}

; Function Attrs: nounwind readonly uwtable
define fp128 @test_return2(fp128* nocapture readonly %ptr) local_unnamed_addr #0 !dbg !25 {
entry:
  tail call void @llvm.dbg.value(metadata fp128* %ptr, i64 0, metadata !27, metadata !17), !dbg !29
  %0 = load fp128, fp128* %ptr, align 16, !dbg !30, !tbaa !20
  tail call void @llvm.dbg.value(metadata fp128 %0, i64 0, metadata !28, metadata !17), !dbg !31
  ret fp128 %0, !dbg !32
}

; X64-LABEL: test_return2:
; X64:       .loc
; X64:       movaps     (%rdi), %xmm0
; X64:       .loc
; X64:       retq
; Function Attrs: nounwind readonly uwtable

define fp128 @test_return3(fp128* nocapture readonly %ptr) local_unnamed_addr #0 !dbg !33 {
entry:
  tail call void @llvm.dbg.value(metadata fp128* %ptr, i64 0, metadata !35, metadata !17), !dbg !36
  %0 = load fp128, fp128* %ptr, align 16, !dbg !37, !tbaa !20
  %add = fadd fp128 %0, %0, !dbg !38
  ret fp128 %add, !dbg !39
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
define fp128 @test_return4() local_unnamed_addr #1 !dbg !40 {
entry:
  %0 = load fp128*, fp128** @ld_ptr, align 8, !dbg !43, !tbaa !44
  %1 = load fp128, fp128* %0, align 16, !dbg !46, !tbaa !20
  ret fp128 %1, !dbg !47
; X64-LABEL: test_return4:
; X64:       .loc
; X64:       movq	ld_ptr(%rip), %rax
; X64:       .loc
; X64:       movaps	(%rax), %xmm0
; X64:       .loc
; X64:       retq
}

; Function Attrs: nounwind readonly uwtable
define fp128 @test_return5() local_unnamed_addr #0 !dbg !48 {
entry:
  %0 = load fp128*, fp128** @ld_ptr, align 8, !dbg !51, !tbaa !44
  %1 = load fp128, fp128* %0, align 16, !dbg !52, !tbaa !20
  tail call void @llvm.dbg.value(metadata fp128 %1, i64 0, metadata !50, metadata !17), !dbg !53
  ret fp128 %1, !dbg !54
; X64-LABEL: test_return5:
; X64:       .loc
; X64:       movq	ld_ptr(%rip), %rax
; X64:       .loc
; X64:       movaps	(%rax), %xmm0
; X64:       .loc
; X64:       retq
}

; Function Attrs: norecurse nounwind readonly uwtable
define fp128 @test_return6() local_unnamed_addr #1 !dbg !55 {
entry:
  %0 = load fp128*, fp128** @ld_ptr, align 8, !dbg !56, !tbaa !44
  %1 = load fp128, fp128* %0, align 16, !dbg !57, !tbaa !20
  %add = fadd fp128 %1, %1, !dbg !58
  ret fp128 %add, !dbg !59
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

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "ld_ptr", scope: !2, file: !3, line: 17, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 4.0.0 (trunk 281495)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "fp128-g.c", directory: "/disk5/chh/Debug/ld.loop")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64)
!7 = !DIBasicType(name: "long double", size: 128, align: 128, encoding: DW_ATE_float)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"PIC Level", i32 2}
!11 = !{!"clang version 4.0.0 (trunk 281495)"}
!12 = distinct !DISubprogram(name: "test_return1", scope: !3, file: !3, line: 3, type: !13, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !2, variables: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{!7, !6}
!15 = !{!16}
!16 = !DILocalVariable(name: "ptr", arg: 1, scope: !12, file: !3, line: 3, type: !6)
!17 = !DIExpression()
!18 = !DILocation(line: 3, column: 39, scope: !12)
!19 = !DILocation(line: 4, column: 12, scope: !12)
!20 = !{!21, !21, i64 0}
!21 = !{!"long double", !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C/C++ TBAA"}
!24 = !DILocation(line: 4, column: 5, scope: !12)
!25 = distinct !DISubprogram(name: "test_return2", scope: !3, file: !3, line: 7, type: !13, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: true, unit: !2, variables: !26)
!26 = !{!27, !28}
!27 = !DILocalVariable(name: "ptr", arg: 1, scope: !25, file: !3, line: 7, type: !6)
!28 = !DILocalVariable(name: "value", scope: !25, file: !3, line: 8, type: !7)
!29 = !DILocation(line: 7, column: 39, scope: !25)
!30 = !DILocation(line: 9, column: 14, scope: !25)
!31 = !DILocation(line: 8, column: 17, scope: !25)
!32 = !DILocation(line: 10, column: 5, scope: !25)
!33 = distinct !DISubprogram(name: "test_return3", scope: !3, file: !3, line: 13, type: !13, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: true, unit: !2, variables: !34)
!34 = !{!35}
!35 = !DILocalVariable(name: "ptr", arg: 1, scope: !33, file: !3, line: 13, type: !6)
!36 = !DILocation(line: 13, column: 39, scope: !33)
!37 = !DILocation(line: 14, column: 12, scope: !33)
!38 = !DILocation(line: 14, column: 17, scope: !33)
!39 = !DILocation(line: 14, column: 5, scope: !33)
!40 = distinct !DISubprogram(name: "test_return4", scope: !3, file: !3, line: 18, type: !41, isLocal: false, isDefinition: true, scopeLine: 18, isOptimized: true, unit: !2, variables: !4)
!41 = !DISubroutineType(types: !42)
!42 = !{!7}
!43 = !DILocation(line: 19, column: 13, scope: !40)
!44 = !{!45, !45, i64 0}
!45 = !{!"any pointer", !22, i64 0}
!46 = !DILocation(line: 19, column: 12, scope: !40)
!47 = !DILocation(line: 19, column: 5, scope: !40)
!48 = distinct !DISubprogram(name: "test_return5", scope: !3, file: !3, line: 22, type: !41, isLocal: false, isDefinition: true, scopeLine: 22, isOptimized: true, unit: !2, variables: !49)
!49 = !{!50}
!50 = !DILocalVariable(name: "value", scope: !48, file: !3, line: 23, type: !7)
!51 = !DILocation(line: 23, column: 26, scope: !48)
!52 = !DILocation(line: 23, column: 25, scope: !48)
!53 = !DILocation(line: 23, column: 17, scope: !48)
!54 = !DILocation(line: 24, column: 5, scope: !48)
!55 = distinct !DISubprogram(name: "test_return6", scope: !3, file: !3, line: 27, type: !41, isLocal: false, isDefinition: true, scopeLine: 27, isOptimized: true, unit: !2, variables: !4)
!56 = !DILocation(line: 28, column: 13, scope: !55)
!57 = !DILocation(line: 28, column: 12, scope: !55)
!58 = !DILocation(line: 28, column: 20, scope: !55)
!59 = !DILocation(line: 28, column: 5, scope: !55)


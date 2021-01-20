; RUN: opt -S -loop-rotate < %s | FileCheck --check-prefix=FULL %s
; RUN: opt -S -loop-rotate -rotation-prepare-for-lto < %s | FileCheck --check-prefix=PREPARE %s
; RUN: opt -S -passes='require<targetir>,require<assumptions>,loop(loop-rotate)' < %s | FileCheck --check-prefix=FULL %s
; RUN: opt -S -passes='require<targetir>,require<assumptions>,loop(loop-rotate)' -rotation-prepare-for-lto < %s | FileCheck --check-prefix=PREPARE %s

; Test case to make sure loop-rotate avoids rotating during the prepare-for-lto
; stage, when the header contains a call which may be inlined during the LTO stage.
define void @test_prepare_for_lto() {
; FULL-LABEL: @test_prepare_for_lto(
; FULL-NEXT:  entry:
; FULL-NEXT:    %array = alloca [20 x i32], align 16
; FULL-NEXT:    %arrayidx = getelementptr inbounds [20 x i32], [20 x i32]* %array, i64 0, i64 0
; FULL-NEXT:    call void @may_be_inlined()
; FULL-NEXT:    br label %for.body
;
; PREPARE-LABEL: @test_prepare_for_lto(
; PREPARE-NEXT:  entry:
; PREPARE-NEXT:    %array = alloca [20 x i32], align 16
; PREPARE-NEXT:    br label %for.cond
;
entry:
  %array = alloca [20 x i32], align 16
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, 100
  %arrayidx = getelementptr inbounds [20 x i32], [20 x i32]* %array, i64 0, i64 0
  call void @may_be_inlined()
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %arrayidx, align 16
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

define void @may_be_inlined() {
  ret void
}

; Intrinsics, like @llvm.dbg.value are never inlined and should not block loop
; rotation, even when preparing for LTO.
define void @test_prepare_for_lto_intrinsic() !dbg !7 {
; FULL-LABEL: @test_prepare_for_lto_intrinsic(
; FULL-NEXT:  entry:
; FULL-NEXT:    %array = alloca [20 x i32], align 16
; FULL-NEXT:    call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !13
; FULL-NEXT:    %arrayidx = getelementptr inbounds [20 x i32], [20 x i32]* %array, i64 0, i64 0
; FULL-NEXT:    br label %for.body
;
; PREPARE-LABEL: @test_prepare_for_lto_intrinsic(
; PREPARE-NEXT:  entry:
; PREPARE-NEXT:    %array = alloca [20 x i32], align 16
; PREPARE-NEXT:    call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !13
; PREPARE-NEXT:    %arrayidx = getelementptr inbounds [20 x i32], [20 x i32]* %array, i64 0, i64 0
; PREPARE-NEXT:    br label %for.body
;
entry:
  %array = alloca [20 x i32], align 16
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  call void @llvm.dbg.value(metadata i32 %i.0, metadata !12, metadata !DIExpression()), !dbg !13
  %cmp = icmp slt i32 %i.0, 100
  %arrayidx = getelementptr inbounds [20 x i32], [20 x i32]* %array, i64 0, i64 0
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %arrayidx, align 16
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #2


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "input", arg: 1, scope: !7, file: !1, line: 2, type: !10)
!13 = !DILocation(line: 2, column: 15, scope: !7)

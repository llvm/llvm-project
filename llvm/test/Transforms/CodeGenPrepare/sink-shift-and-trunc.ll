; REQUIRES: aarch64-registered-target
; RUN: opt < %s -codegenprepare -mtriple=arm64-apple-ios -S | FileCheck %s
; RUN: opt < %s -codegenprepare -mtriple=arm64-apple-ios -S --try-experimental-debuginfo-iterators | FileCheck %s

@first_ones = external global [65536 x i8]

define i32 @fct19(i64 %arg1) #0 !dbg !6 {
; CHECK-LABEL: @fct19
entry:
  %x.sroa.1.0.extract.shift = lshr i64 %arg1, 16, !dbg !35
  %x.sroa.1.0.extract.trunc = trunc i64 %x.sroa.1.0.extract.shift to i16, !dbg !36

  %x.sroa.3.0.extract.shift = lshr i64 %arg1, 32, !dbg !37
  call void @llvm.dbg.value(metadata i64 %x.sroa.3.0.extract.shift, metadata !13, metadata !DIExpression()), !dbg !37
; CHECK: #dbg_value(i64 %arg1, {{.*}}, !DIExpression(DW_OP_constu, 32, DW_OP_shr, DW_OP_stack_value), [[shift2_loc:![0-9]+]]

  %x.sroa.5.0.extract.shift = lshr i64 %arg1, 48, !dbg !38
  %tobool = icmp eq i64 %x.sroa.5.0.extract.shift, 0, !dbg !39
  br i1 %tobool, label %if.end, label %if.then, !dbg !40

if.then:                                          ; preds = %entry
  %arrayidx3 = getelementptr inbounds [65536 x i8], ptr @first_ones, i64 0, i64 %x.sroa.5.0.extract.shift, !dbg !41
  %0 = load i8, ptr %arrayidx3, align 1, !dbg !42
  %conv = zext i8 %0 to i32, !dbg !43
  br label %return, !dbg !44

if.end:                                           ; preds = %entry
; CHECK-LABEL: if.end:
; CHECK-NEXT: lshr i64 %arg1, 32, !dbg [[shift2_loc]]
  %x.sroa.3.0.extract.trunc = trunc i64 %x.sroa.3.0.extract.shift to i16, !dbg !45
  %tobool6 = icmp eq i16 %x.sroa.3.0.extract.trunc, 0, !dbg !46
  br i1 %tobool6, label %if.end13, label %if.then7, !dbg !47

if.then7:                                         ; preds = %if.end
  %idxprom10 = and i64 %x.sroa.3.0.extract.shift, 65535, !dbg !48
  %arrayidx11 = getelementptr inbounds [65536 x i8], ptr @first_ones, i64 0, i64 %idxprom10, !dbg !49
  %1 = load i8, ptr %arrayidx11, align 1, !dbg !50
  %conv12 = zext i8 %1 to i32, !dbg !51
  %add = add nsw i32 %conv12, 16, !dbg !52
  br label %return, !dbg !53

if.end13:                                         ; preds = %if.end
; CHECK-LABEL: if.end13:
; CHECK-NEXT: [[shift1:%.*]] = lshr i64 %arg1, 16, !dbg [[shift1_loc:![0-9]+]]
; CHECK-NEXT: trunc i64 [[shift1]] to i16, !dbg [[trunc1_loc:![0-9]+]]
  %tobool16 = icmp eq i16 %x.sroa.1.0.extract.trunc, 0, !dbg !54
  br i1 %tobool16, label %return, label %if.then17, !dbg !55

if.then17:                                        ; preds = %if.end13
  %idxprom20 = and i64 %x.sroa.1.0.extract.shift, 65535, !dbg !56
  %arrayidx21 = getelementptr inbounds [65536 x i8], ptr @first_ones, i64 0, i64 %idxprom20, !dbg !57
  %2 = load i8, ptr %arrayidx21, align 1, !dbg !58
  %conv22 = zext i8 %2 to i32, !dbg !59
  %add23 = add nsw i32 %conv22, 32, !dbg !60
  br label %return, !dbg !61

return:                                           ; preds = %if.then17, %if.end13, %if.then7, %if.then
  %retval.0 = phi i32 [ %conv, %if.then ], [ %add, %if.then7 ], [ %add23, %if.then17 ], [ 64, %if.end13 ], !dbg !62
  ret i32 %retval.0, !dbg !63
}

; CodeGenPrepare was erasing the unused lshr instruction, but then further
; processing the instruction after it was freed. If this bug is still present,
; this test will always crash in an LLVM built with ASAN enabled, and may
; crash even if ASAN is not enabled.

define i32 @shift_unused(i32 %a) {
; CHECK-LABEL: @shift_unused(
; CHECK-NEXT:  BB2:
; CHECK-NEXT:    ret i32 [[A:%.*]]
;
  %as = lshr i32 %a, 3
  br label %BB2

BB2:
  ret i32 %a
}

; CHECK: [[shift1_loc]] = !DILocation(line: 1
; CHECK: [[trunc1_loc]] = !DILocation(line: 2
; CHECK: [[shift2_loc]] = !DILocation(line: 3

declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readonly ssp }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "sink-shift-and-trunc.ll", directory: "/")
!2 = !{}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "fct19", linkageName: "fct19", scope: null, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!13}
!10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!13 = !DILocalVariable(name: "3", scope: !6, file: !1, line: 3, type: !10)
!35 = !DILocation(line: 1, column: 1, scope: !6)
!36 = !DILocation(line: 2, column: 1, scope: !6)
!37 = !DILocation(line: 3, column: 1, scope: !6)
!38 = !DILocation(line: 4, column: 1, scope: !6)
!39 = !DILocation(line: 5, column: 1, scope: !6)
!40 = !DILocation(line: 6, column: 1, scope: !6)
!41 = !DILocation(line: 7, column: 1, scope: !6)
!42 = !DILocation(line: 8, column: 1, scope: !6)
!43 = !DILocation(line: 9, column: 1, scope: !6)
!44 = !DILocation(line: 10, column: 1, scope: !6)
!45 = !DILocation(line: 11, column: 1, scope: !6)
!46 = !DILocation(line: 12, column: 1, scope: !6)
!47 = !DILocation(line: 13, column: 1, scope: !6)
!48 = !DILocation(line: 14, column: 1, scope: !6)
!49 = !DILocation(line: 15, column: 1, scope: !6)
!50 = !DILocation(line: 16, column: 1, scope: !6)
!51 = !DILocation(line: 17, column: 1, scope: !6)
!52 = !DILocation(line: 18, column: 1, scope: !6)
!53 = !DILocation(line: 19, column: 1, scope: !6)
!54 = !DILocation(line: 20, column: 1, scope: !6)
!55 = !DILocation(line: 21, column: 1, scope: !6)
!56 = !DILocation(line: 22, column: 1, scope: !6)
!57 = !DILocation(line: 23, column: 1, scope: !6)
!58 = !DILocation(line: 24, column: 1, scope: !6)
!59 = !DILocation(line: 25, column: 1, scope: !6)
!60 = !DILocation(line: 26, column: 1, scope: !6)
!61 = !DILocation(line: 27, column: 1, scope: !6)
!62 = !DILocation(line: 28, column: 1, scope: !6)
!63 = !DILocation(line: 29, column: 1, scope: !6)

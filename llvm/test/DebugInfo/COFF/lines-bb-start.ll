; RUN: llc -O0 < %s | FileCheck %s

source_filename = "t.c"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.0.24215"

@str_const = internal unnamed_addr constant [4 x i8] c"str\00", align 1

declare i32 @puts(i8*)
declare void @use_i32(i32*)
declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone

; We had a line info quality issue where the LEA for the string constant had no
; location info, so the .cv_loc directive appeared after it. Now we have logic
; that tries to emit the first valid location to the top of each MBB.

define void @lea_str_loc(i1 %cond) !dbg !8 {
entry:
  br i1 %cond, label %if.then, label %if.end, !dbg !17

if.then:                                          ; preds = %entry
  br label %return, !dbg !18

if.end:                                           ; preds = %entry
  %call = call i32 @puts(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @str_const, i32 0, i32 0)), !dbg !19
  br label %return, !dbg !20

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !20
}

; The t.c:5 line marker should appear immediately after the BB label.

; CHECK-LABEL: _lea_str_loc:
; CHECK:         .cv_loc {{.*}} # t.c:4:5
; CHECK:         jmp     LBB{{.*}}
; CHECK: LBB0_{{.*}}:                                 # %if.end
; CHECK-NEXT:    .cv_loc {{.*}} # t.c:5:3
; CHECK-NEXT:    leal    _str_const, %[[reg:[^ ]*]]
; CHECK-NEXT:    movl    %[[reg]], (%esp)
; CHECK-NEXT:    calll   _puts

define void @instr_no_loc(i1 %cond) !dbg !21 {
entry:
  br i1 %cond, label %if.then, label %if.end, !dbg !22

if.then:                                          ; preds = %entry
  br label %return, !dbg !23

if.end:                                           ; preds = %entry
  call void asm sideeffect "nop", ""()
  %call = call i32 @puts(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @str_const, i32 0, i32 0)), !dbg !24
  br label %return, !dbg !25

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !25
}

; CHECK-LABEL: _instr_no_loc:
; CHECK:         .cv_loc {{.*}} # t.c:4:5
; CHECK:         jmp     LBB{{.*}}
; CHECK: LBB1_{{.*}}:                                 # %if.end
; CHECK-NEXT:    .cv_loc {{.*}} # t.c:5:3
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    leal    _str_const, %[[reg:[^ ]*]]
; CHECK-NEXT:    movl    %[[reg]], (%esp)
; CHECK-NEXT:    calll   _puts

define void @lea_dbg_value(i1 %cond) !dbg !30 {
entry:
  %value = alloca i32
  store i32 42, i32* %value
  br i1 %cond, label %if.then, label %if.end, !dbg !31

if.then:                                          ; preds = %entry
  br label %return, !dbg !32

if.end:                                           ; preds = %entry
  call void @llvm.dbg.value(metadata i32* %value, metadata !35, metadata !13), !dbg !34
  call void @use_i32(i32* %value), !dbg !33
  br label %return, !dbg !34

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !34
}

; CHECK-LABEL: _lea_dbg_value:
; CHECK:         .cv_loc {{.*}} # t.c:4:5
; CHECK:         jmp     LBB{{.*}}
; CHECK: LBB2_{{.*}}:                                 # %if.end
; CHECK-NEXT:    .cv_loc {{.*}} # t.c:5:3
; CHECK:         leal 4(%esp), %[[reg:[^ ]*]]
; CHECK:         movl    %[[reg]], (%esp)
; CHECK:         calll   _use_i32

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "b32df088e991f1996b4e4deb3855c14b")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"CodeView", i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{!"clang version 5.0.0 "}
!8 = distinct !DISubprogram(name: "lea_str_loc", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIExpression(DW_OP_deref)
!14 = !DILocation(line: 2, column: 12, scope: !8)
!15 = !DILocation(line: 3, column: 7, scope: !16)
!16 = distinct !DILexicalBlock(scope: !8, file: !1, line: 3, column: 7)
!17 = !DILocation(line: 3, column: 7, scope: !8)
!18 = !DILocation(line: 4, column: 5, scope: !16)
!19 = !DILocation(line: 5, column: 3, scope: !8)
!20 = !DILocation(line: 6, column: 1, scope: !8)
!21 = distinct !DISubprogram(name: "instr_no_loc", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!22 = !DILocation(line: 3, column: 7, scope: !21)
!23 = !DILocation(line: 4, column: 5, scope: !21)
!24 = !DILocation(line: 5, column: 3, scope: !21)
!25 = !DILocation(line: 6, column: 1, scope: !21)
!30 = distinct !DISubprogram(name: "lea_dbg_value", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!31 = !DILocation(line: 3, column: 7, scope: !30)
!32 = !DILocation(line: 4, column: 5, scope: !30)
!33 = !DILocation(line: 5, column: 3, scope: !30)
!34 = !DILocation(line: 6, column: 1, scope: !30)
!35 = !DILocalVariable(name: "v", scope: !30, file: !1, line: 3, type: !11)

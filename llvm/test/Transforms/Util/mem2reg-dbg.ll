; RUN: opt -mem2reg -S < %s | FileCheck %s

; Test that a @llvm.dbg.value node is created to describe the value returned by a phi node when
; lowering a @llvm.dbg.declare node

; Created from the C code, compiled with -O0 -g:
;
; int func(int a)
; {
;         int c = 1;
;         if (a < 0 ) {
;                 c = 12;
;         }
;         return c;
; }

; Function Attrs: nounwind
define i32 @func(i32 %a) #0 !dbg !8 {
entry:
  %a.addr = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !12, metadata !13), !dbg !14
  call void @llvm.dbg.declare(metadata i32* %c, metadata !15, metadata !13), !dbg !16
  store i32 1, i32* %c, align 4, !dbg !16
  %0 = load i32, i32* %a.addr, align 4, !dbg !17
  %cmp = icmp slt i32 %0, 0, !dbg !19
  br i1 %cmp, label %if.then, label %if.end, !dbg !20

if.then:                                          ; preds = %entry
  store i32 12, i32* %c, align 4, !dbg !21
  br label %if.end, !dbg !23

if.end:                                           ; preds = %if.then, %entry
  %1 = load i32, i32* %c, align 4, !dbg !24
; CHECK: [[PHI:%.*]] = phi i32 [ 12, {{.*}} ], [ 1, {{.*}} ]
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 [[PHI]], i64 0, metadata !15, metadata !13), !dbg !16
  ret i32 %1, !dbg !25
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "a.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"min_enum_size", i32 4}
!7 = !{!"clang"}
!8 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!13 = !DIExpression()
!14 = !DILocation(line: 1, column: 14, scope: !8)
!15 = !DILocalVariable(name: "c", scope: !8, file: !1, line: 3, type: !11)
!16 = !DILocation(line: 3, column: 6, scope: !8)
!17 = !DILocation(line: 4, column: 6, scope: !18)
!18 = distinct !DILexicalBlock(scope: !8, file: !1, line: 4, column: 6)
!19 = !DILocation(line: 4, column: 8, scope: !18)
!20 = !DILocation(line: 4, column: 6, scope: !8)
!21 = !DILocation(line: 5, column: 5, scope: !22)
!22 = distinct !DILexicalBlock(scope: !18, file: !1, line: 4, column: 14)
!23 = !DILocation(line: 6, column: 2, scope: !22)
!24 = !DILocation(line: 7, column: 9, scope: !8)
!25 = !DILocation(line: 7, column: 2, scope: !8)


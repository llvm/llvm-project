; RUN: llc -march=mips -mcpu=mips32r2 -O1 -filetype=obj -relocation-model=pic <%s | \
; RUN:    llvm-dwarfdump -debug-dump=all - | FileCheck %s -check-prefix=F0
; RUN: llc -march=mips -mcpu=mips32r2 -O1 -filetype=obj -relocation-model=pic <%s | \
; RUN:    llvm-dwarfdump -debug-dump=all - | FileCheck %s -check-prefix=F1

; void foo(int *);
;
; int f0(int a, int b, int c, int d, int e) {
;   int x = a + b + c + d + e;
;   foo(&x);
;   return x;
; }
;
; int f1(int a, int b, int c, int d, int e) {
;   int x __attribute__((aligned(16))) = a + b + c + d + e;
;   foo(&x);
;   return x;
; }

declare void @llvm.lifetime.start(i64, i8* nocapture)
declare void @llvm.lifetime.end(i64, i8* nocapture)

declare void @foo(i32*)

; F0: DW_AT_name {{.*}}"e"
; F0: DW_TAG_variable
; F0-NEXT: DW_AT_location [DW_FORM_sec_offset]   ([[LOC:.*]])
; F0-NEXT: DW_AT_name [DW_FORM_strp]     ( .debug_str[0x0000006b] = "x")
;
; x -> DW_OP_reg1(51)
; F0: [[LOC]]: Beginning address offset: 0x0000000000000028
; F0:             Ending address offset: 0x000000000000002c
; F0:              Location description: 51

define i32 @f0(i32 signext %a, i32 signext %b, i32 signext %c, i32 signext %d, i32 signext %e) !dbg !4 {
entry:
  %x = alloca i32, align 4
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !9,  metadata !DIExpression()), !dbg !27
  tail call void @llvm.dbg.value(metadata i32 %b, i64 0, metadata !10, metadata !DIExpression()), !dbg !28
  tail call void @llvm.dbg.value(metadata i32 %c, i64 0, metadata !11, metadata !DIExpression()), !dbg !29
  tail call void @llvm.dbg.value(metadata i32 %d, i64 0, metadata !12, metadata !DIExpression()), !dbg !30
  tail call void @llvm.dbg.value(metadata i32 %e, i64 0, metadata !13, metadata !DIExpression()), !dbg !31
  %0 = bitcast i32* %x to i8*, !dbg !32
  call void @llvm.lifetime.start(i64 4, i8* %0) #4, !dbg !32
  %add = add nsw i32 %b, %a, !dbg !33
  %add1 = add nsw i32 %add, %c, !dbg !34
  %add2 = add nsw i32 %add1, %d, !dbg !35
  %add3 = add nsw i32 %add2, %e, !dbg !36
  tail call void @llvm.dbg.value(metadata i32 %add3, i64 0, metadata !14, metadata !DIExpression()), !dbg !37
  store i32 %add3, i32* %x, align 4, !dbg !37, !tbaa !38
  tail call void @llvm.dbg.value(metadata i32* %x, i64 0, metadata !14, metadata !26), !dbg !37
  call void @foo(i32* nonnull %x) #4, !dbg !42
  call void @llvm.dbg.value(metadata i32* %x, i64 0, metadata !14, metadata !26), !dbg !37
  %1 = load i32, i32* %x, align 4, !dbg !43, !tbaa !38
  call void @llvm.lifetime.end(i64 4, i8* %0) #4, !dbg !44
  ret i32 %1, !dbg !45
}


; F1: DW_AT_name {{.*}}"x"
; F1: DW_AT_name {{.*}}"e"
; F1: DW_TAG_variable
; F1-NEXT: DW_AT_location [DW_FORM_sec_offset]   ([[LOC:.*]])
; F1-NEXT: DW_AT_name [DW_FORM_strp]     ( .debug_str[0x0000006b] = "x")

; x -> DW_OP_reg1(51)
; F1: [[LOC]]: Beginning address offset: 0x0000000000000080
; F1:             Ending address offset: 0x0000000000000084
; F1:              Location description: 51

define i32 @f1(i32 signext %a, i32 signext %b, i32 signext %c, i32 signext %d, i32 signext %e) !dbg !15 {
entry:
  %x = alloca i32, align 16
  tail call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !17, metadata !DIExpression()), !dbg !46
  tail call void @llvm.dbg.value(metadata i32 %b, i64 0, metadata !18, metadata !DIExpression()), !dbg !47
  tail call void @llvm.dbg.value(metadata i32 %c, i64 0, metadata !19, metadata !DIExpression()), !dbg !48
  tail call void @llvm.dbg.value(metadata i32 %d, i64 0, metadata !20, metadata !DIExpression()), !dbg !49
  tail call void @llvm.dbg.value(metadata i32 %e, i64 0, metadata !21, metadata !DIExpression()), !dbg !50
  %0 = bitcast i32* %x to i8*, !dbg !51
  call void @llvm.lifetime.start(i64 4, i8* %0) #4, !dbg !51
  %add = add nsw i32 %b, %a, !dbg !52
  %add1 = add nsw i32 %add, %c, !dbg !53
  %add2 = add nsw i32 %add1, %d, !dbg !54
  %add3 = add nsw i32 %add2, %e, !dbg !55
  tail call void @llvm.dbg.value(metadata i32 %add3, i64 0, metadata !22, metadata !DIExpression()), !dbg !56
  store i32 %add3, i32* %x, align 16, !dbg !56, !tbaa !38
  tail call void @llvm.dbg.value(metadata i32* %x, i64 0, metadata !22, metadata !26), !dbg !56
  call void @foo(i32* nonnull %x) #4, !dbg !57
  call void @llvm.dbg.value(metadata i32* %x, i64 0, metadata !22, metadata !26), !dbg !56
  %1 = load i32, i32* %x, align 16, !dbg !58, !tbaa !38
  call void @llvm.lifetime.end(i64 4, i8* %0) #4, !dbg !59
  ret i32 %1, !dbg !60
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23, !24}
!llvm.ident = !{!25}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 251783) (llvm/trunk 251781)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/home/vk/repos/tmp/dwarf")
!2 = !{}
!4 = distinct !DISubprogram(name: "f0", scope: !1, file: !1, line: 4, type: !5, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !8)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7, !7, !7, !7, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{!9, !10, !11, !12, !13, !14}
!9 = !DILocalVariable(name: "a", arg: 1, scope: !4, file: !1, line: 4, type: !7)
!10 = !DILocalVariable(name: "b", arg: 2, scope: !4, file: !1, line: 4, type: !7)
!11 = !DILocalVariable(name: "c", arg: 3, scope: !4, file: !1, line: 4, type: !7)
!12 = !DILocalVariable(name: "d", arg: 4, scope: !4, file: !1, line: 4, type: !7)
!13 = !DILocalVariable(name: "e", arg: 5, scope: !4, file: !1, line: 4, type: !7)
!14 = !DILocalVariable(name: "x", scope: !4, file: !1, line: 5, type: !7)
!15 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 11, type: !5, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !16)
!16 = !{!17, !18, !19, !20, !21, !22}
!17 = !DILocalVariable(name: "a", arg: 1, scope: !15, file: !1, line: 11, type: !7)
!18 = !DILocalVariable(name: "b", arg: 2, scope: !15, file: !1, line: 11, type: !7)
!19 = !DILocalVariable(name: "c", arg: 3, scope: !15, file: !1, line: 11, type: !7)
!20 = !DILocalVariable(name: "d", arg: 4, scope: !15, file: !1, line: 11, type: !7)
!21 = !DILocalVariable(name: "e", arg: 5, scope: !15, file: !1, line: 11, type: !7)
!22 = !DILocalVariable(name: "x", scope: !15, file: !1, line: 12, type: !7)
!23 = !{i32 2, !"Dwarf Version", i32 4}
!24 = !{i32 2, !"Debug Info Version", i32 3}
!25 = !{!"clang version 3.8.0 (trunk 251783) (llvm/trunk 251781)"}
!26 = !DIExpression(DW_OP_deref)
!27 = !DILocation(line: 4, column: 12, scope: !4)
!28 = !DILocation(line: 4, column: 19, scope: !4)
!29 = !DILocation(line: 4, column: 26, scope: !4)
!30 = !DILocation(line: 4, column: 33, scope: !4)
!31 = !DILocation(line: 4, column: 40, scope: !4)
!32 = !DILocation(line: 5, column: 3, scope: !4)
!33 = !DILocation(line: 5, column: 13, scope: !4)
!34 = !DILocation(line: 5, column: 17, scope: !4)
!35 = !DILocation(line: 5, column: 21, scope: !4)
!36 = !DILocation(line: 5, column: 25, scope: !4)
!37 = !DILocation(line: 5, column: 7, scope: !4)
!38 = !{!39, !39, i64 0}
!39 = !{!"int", !40, i64 0}
!40 = !{!"omnipotent char", !41, i64 0}
!41 = !{!"Simple C/C++ TBAA"}
!42 = !DILocation(line: 6, column: 3, scope: !4)
!43 = !DILocation(line: 7, column: 10, scope: !4)
!44 = !DILocation(line: 8, column: 1, scope: !4)
!45 = !DILocation(line: 7, column: 3, scope: !4)
!46 = !DILocation(line: 11, column: 12, scope: !15)
!47 = !DILocation(line: 11, column: 19, scope: !15)
!48 = !DILocation(line: 11, column: 26, scope: !15)
!49 = !DILocation(line: 11, column: 33, scope: !15)
!50 = !DILocation(line: 11, column: 40, scope: !15)
!51 = !DILocation(line: 12, column: 3, scope: !15)
!52 = !DILocation(line: 12, column: 42, scope: !15)
!53 = !DILocation(line: 12, column: 46, scope: !15)
!54 = !DILocation(line: 12, column: 50, scope: !15)
!55 = !DILocation(line: 12, column: 54, scope: !15)
!56 = !DILocation(line: 12, column: 7, scope: !15)
!57 = !DILocation(line: 13, column: 3, scope: !15)
!58 = !DILocation(line: 14, column: 10, scope: !15)
!59 = !DILocation(line: 15, column: 1, scope: !15)
!60 = !DILocation(line: 14, column: 3, scope: !15)

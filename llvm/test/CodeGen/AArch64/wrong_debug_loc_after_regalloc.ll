; RUN: llc -O3 -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -print-after virtregrewriter < %s >%t 2>&1 && FileCheck <%t %s 

; This test checks that DBG_VALUE instruction placed correctly.
; Specifically: if Register Allocator inserts additional instructions
; in the beginning of basic block then it should not break placement 
; of DBG_VALUE for loop index variable. That DBG_VALUE instruction 
; for "i" variable should be placed before any real loop instruction. 
; https://reviews.llvm.org/D62650

; Created from the following C source: 

; cat test_debug_val.cpp
;
; void func(int, ...);
;
; int array[0x100];
;
; int main( int argc, char **argv )    
; {    
;    int var = 56;
;
;    int a1 = array[1]; int a2 = array[2]; int a3 = array[3]; int a4 = array[4];
;    int a5 = array[5]; int a6 = array[6]; int a7 = array[7]; int a8 = array[8];
;    int a9 = array[9]; int a10 = array[10];
; 
;    for( int i = 0; i < 0x100; i++ ) {    
; 
;        array[i] = var;
; 
;        func(0, i, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 );
;    }
; 
;    return 0;
; }
;
;
; clang -O3 -g -c --target=aarch64-unknown-linux -std=gnu++14 test_debug_val.cpp -emit-llvm -S -o -


; CHECK:  bb.2.for.body
; CHECK-NEXT: predecessors
; CHECK-NEXT: successors
; CHECK-NEXT: liveins
; CHECK-NOT: MOV
; CHECK: DBG_VALUE $[[REG:[xw][0-9]+]], $noreg, !"i"
; CHECK: MOV

; ModuleID = 'test_debug_val.cpp'
source_filename = "test_debug_val.cpp"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux"

@array = dso_local local_unnamed_addr global [256 x i32] zeroinitializer, align 4, !dbg !0

; Function Attrs: norecurse
define dso_local i32 @main(i32 %argc, ptr nocapture readnone %argv) local_unnamed_addr #0 !dbg !14 {
entry:
  call void @llvm.dbg.value(metadata i32 %argc, metadata !21, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata ptr %argv, metadata !22, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 56, metadata !23, metadata !DIExpression()), !dbg !36
  %0 = load i32, ptr getelementptr inbounds ([256 x i32], ptr @array, i64 0, i64 1), align 4, !dbg !37
  call void @llvm.dbg.value(metadata i32 %0, metadata !24, metadata !DIExpression()), !dbg !36
  %1 = load i32, ptr getelementptr inbounds ([256 x i32], ptr @array, i64 0, i64 2), align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 %1, metadata !25, metadata !DIExpression()), !dbg !36
  %2 = load i32, ptr getelementptr inbounds ([256 x i32], ptr @array, i64 0, i64 3), align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %2, metadata !26, metadata !DIExpression()), !dbg !36
  %3 = load i32, ptr getelementptr inbounds ([256 x i32], ptr @array, i64 0, i64 4), align 4, !dbg !44
  call void @llvm.dbg.value(metadata i32 %3, metadata !27, metadata !DIExpression()), !dbg !36
  %4 = load i32, ptr getelementptr inbounds ([256 x i32], ptr @array, i64 0, i64 5), align 4, !dbg !45
  call void @llvm.dbg.value(metadata i32 %4, metadata !28, metadata !DIExpression()), !dbg !36
  %5 = load i32, ptr getelementptr inbounds ([256 x i32], ptr @array, i64 0, i64 6), align 4, !dbg !46
  call void @llvm.dbg.value(metadata i32 %5, metadata !29, metadata !DIExpression()), !dbg !36
  %6 = load i32, ptr getelementptr inbounds ([256 x i32], ptr @array, i64 0, i64 7), align 4, !dbg !47
  call void @llvm.dbg.value(metadata i32 %6, metadata !30, metadata !DIExpression()), !dbg !36
  %7 = load i32, ptr getelementptr inbounds ([256 x i32], ptr @array, i64 0, i64 8), align 4, !dbg !48
  call void @llvm.dbg.value(metadata i32 %7, metadata !31, metadata !DIExpression()), !dbg !36
  %8 = load i32, ptr getelementptr inbounds ([256 x i32], ptr @array, i64 0, i64 9), align 4, !dbg !49
  call void @llvm.dbg.value(metadata i32 %8, metadata !32, metadata !DIExpression()), !dbg !36
  %9 = load i32, ptr getelementptr inbounds ([256 x i32], ptr @array, i64 0, i64 10), align 4, !dbg !50
  call void @llvm.dbg.value(metadata i32 %9, metadata !33, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.value(metadata i32 0, metadata !34, metadata !DIExpression()), !dbg !51
  br label %for.body, !dbg !52

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 0, !dbg !53

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !34, metadata !DIExpression()), !dbg !51
  %arrayidx = getelementptr inbounds [256 x i32], ptr @array, i64 0, i64 %indvars.iv, !dbg !54
  store i32 56, ptr %arrayidx, align 4, !dbg !57
  %10 = trunc i64 %indvars.iv to i32, !dbg !58
  tail call void (i32, ...) @_Z4funciz(i32 0, i32 %10, i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9), !dbg !58
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !59
  call void @llvm.dbg.value(metadata i32 undef, metadata !34, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !51
  %exitcond = icmp eq i64 %indvars.iv.next, 256, !dbg !60
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !dbg !52, !llvm.loop !61
}

declare dso_local void @_Z4funciz(i32, ...) local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind uwtable }
attributes #2 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "array", scope: !2, file: !3, line: 4, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 9.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "test_debug_val.cpp", directory: "")
!4 = !{}
!5 = !{!0}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 8192, elements: !8)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DISubrange(count: 256)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 9.0.0 "}
!14 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 6, type: !15, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !20)
!15 = !DISubroutineType(types: !16)
!16 = !{!7, !7, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!20 = !{!21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34}
!21 = !DILocalVariable(name: "argc", arg: 1, scope: !14, file: !3, line: 6, type: !7)
!22 = !DILocalVariable(name: "argv", arg: 2, scope: !14, file: !3, line: 6, type: !17)
!23 = !DILocalVariable(name: "var", scope: !14, file: !3, line: 8, type: !7)
!24 = !DILocalVariable(name: "a1", scope: !14, file: !3, line: 10, type: !7)
!25 = !DILocalVariable(name: "a2", scope: !14, file: !3, line: 10, type: !7)
!26 = !DILocalVariable(name: "a3", scope: !14, file: !3, line: 10, type: !7)
!27 = !DILocalVariable(name: "a4", scope: !14, file: !3, line: 10, type: !7)
!28 = !DILocalVariable(name: "a5", scope: !14, file: !3, line: 11, type: !7)
!29 = !DILocalVariable(name: "a6", scope: !14, file: !3, line: 11, type: !7)
!30 = !DILocalVariable(name: "a7", scope: !14, file: !3, line: 11, type: !7)
!31 = !DILocalVariable(name: "a8", scope: !14, file: !3, line: 11, type: !7)
!32 = !DILocalVariable(name: "a9", scope: !14, file: !3, line: 12, type: !7)
!33 = !DILocalVariable(name: "a10", scope: !14, file: !3, line: 12, type: !7)
!34 = !DILocalVariable(name: "i", scope: !35, file: !3, line: 14, type: !7)
!35 = distinct !DILexicalBlock(scope: !14, file: !3, line: 14, column: 4)
!36 = !DILocation(line: 0, scope: !14)
!37 = !DILocation(line: 10, column: 13, scope: !14)
!42 = !DILocation(line: 10, column: 32, scope: !14)
!43 = !DILocation(line: 10, column: 51, scope: !14)
!44 = !DILocation(line: 10, column: 70, scope: !14)
!45 = !DILocation(line: 11, column: 13, scope: !14)
!46 = !DILocation(line: 11, column: 32, scope: !14)
!47 = !DILocation(line: 11, column: 51, scope: !14)
!48 = !DILocation(line: 11, column: 70, scope: !14)
!49 = !DILocation(line: 12, column: 13, scope: !14)
!50 = !DILocation(line: 12, column: 33, scope: !14)
!51 = !DILocation(line: 0, scope: !35)
!52 = !DILocation(line: 14, column: 4, scope: !35)
!53 = !DILocation(line: 21, column: 4, scope: !14)
!54 = !DILocation(line: 16, column: 8, scope: !55)
!55 = distinct !DILexicalBlock(scope: !56, file: !3, line: 14, column: 37)
!56 = distinct !DILexicalBlock(scope: !35, file: !3, line: 14, column: 4)
!57 = !DILocation(line: 16, column: 17, scope: !55)
!58 = !DILocation(line: 18, column: 8, scope: !55)
!59 = !DILocation(line: 14, column: 32, scope: !56)
!60 = !DILocation(line: 14, column: 22, scope: !56)
!61 = distinct !{!61, !52, !62}
!62 = !DILocation(line: 19, column: 4, scope: !35)

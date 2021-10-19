; Test set representation in DWARF debug info:

; RUN: llc -debugger-tune=gdb -dwarf-version=4 -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -debug-info %t.o | FileCheck %s --check-prefix=CHECK

; ModuleID = 'Main.mb'
source_filename = "../src/Main.m3"
target datalayout = "e-m:e-p:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%M_Const_struct = type { [7 x i8], [1 x i8], [4 x i8], [4 x i8], i8* (i64)*, i8*, void ()*, i8*, [8 x i8], [14 x i8], [2 x i8] }
%M_Main_struct = type { i8*, [32 x i8], i8*, [24 x i8], i8*, [8 x i8], i8* (i64)*, i64, [8 x i8], i8* ()*, i8*, [8 x i8], i8* ()*, [8 x i8] }

@M_Const = internal constant %M_Const_struct { [7 x i8] c"Main_M3", [1 x i8] zeroinitializer, [4 x i8] c"Test", [4 x i8] zeroinitializer, i8* (i64)* @Main_M3, i8* getelementptr inbounds (%M_Const_struct, %M_Const_struct* @M_Const, i32 0, i32 0, i32 0), void ()* @Main__Test, i8* getelementptr inbounds (i8, i8* getelementptr inbounds (%M_Const_struct, %M_Const_struct* @M_Const, i32 0, i32 0, i32 0), i64 8), [8 x i8] zeroinitializer, [14 x i8] c"../src/Main.m3", [2 x i8] zeroinitializer }, align 8
@M_Main = internal global %M_Main_struct { i8* getelementptr inbounds (i8, i8* getelementptr inbounds (%M_Const_struct, %M_Const_struct* @M_Const, i32 0, i32 0, i32 0), i64 56), [32 x i8] zeroinitializer, i8* getelementptr inbounds (i8, i8* getelementptr inbounds (%M_Const_struct, %M_Const_struct* @M_Const, i32 0, i32 0, i32 0), i64 16), [24 x i8] zeroinitializer, i8* getelementptr inbounds (i8, i8* bitcast (%M_Main_struct* @M_Main to i8*), i64 104), [8 x i8] zeroinitializer, i8* (i64)* @Main_M3, i64 3, [8 x i8] zeroinitializer, i8* ()* @Main_I3, i8* getelementptr inbounds (i8, i8* bitcast (%M_Main_struct* @M_Main to i8*), i64 128), [8 x i8] zeroinitializer, i8* ()* @RTHooks_I3, [8 x i8] zeroinitializer }, align 8
@m3_jmpbuf_size = external global i64, align 8

declare i8* @Main_I3()

declare i8* @RTHooks_I3()

; Function Attrs: uwtable
define void @Main__Test() #0 !dbg !5 {
entry:
  %as = alloca i64, align 8
  %bs = alloca i64, align 8
  br label %second, !dbg !21

second:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata i64* %as, metadata !22, metadata !DIExpression()), !dbg !25
  call void @llvm.dbg.declare(metadata i64* %bs, metadata !26, metadata !DIExpression()), !dbg !25
  store i64 36028797018972298, i64* %as, align 8, !dbg !28
  store i64 197, i64* %bs, align 8, !dbg !29
  ret void, !dbg !21
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i8* @alloca()

; Function Attrs: uwtable
define i8* @Main_M3(i64 %mode) #0 !dbg !30 {
entry:
  %mode1 = alloca i64, align 8
  store i64 %mode, i64* %mode1, align 8
  br label %second, !dbg !36

second:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata i64* %mode1, metadata !37, metadata !DIExpression()), !dbg !38
  %v.3 = load i64, i64* %mode1, align 8, !dbg !38
  %icmp = icmp eq i64 %v.3, 0, !dbg !38
  br i1 %icmp, label %if_1, label %else_1, !dbg !38

else_1:                                           ; preds = %second
  call void @Main__Test(), !dbg !36
  br label %if_1, !dbg !36

if_1:                                             ; preds = %else_1, %second
  ret i8* bitcast (%M_Main_struct* @M_Main to i8*), !dbg !36
}

attributes #0 = { uwtable "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.ident = !{!0}
!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!18, !19, !20}

!0 = !{!"versions- cm3: d5.10.0 llvm: 9.0"}
!1 = distinct !DICompileUnit(language: DW_LANG_Modula3, file: !2, producer: "cm3", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3)
!2 = !DIFile(filename: "Main.m3", directory: "/home/cm3/settest/src")
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Enum", scope: !5, file: !2, line: 11, size: 8, align: 8, elements: !9)
!5 = distinct !DISubprogram(name: "Test", linkageName: "Main__Test", scope: !2, file: !2, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{}
!9 = !{!10, !11, !12, !13, !14, !15, !16, !17}
!10 = !DIEnumerator(name: "alpha", value: 0)
!11 = !DIEnumerator(name: "beta", value: 1)
!12 = !DIEnumerator(name: "gamma", value: 2)
!13 = !DIEnumerator(name: "delta", value: 3)
!14 = !DIEnumerator(name: "epsilon", value: 4)
!15 = !DIEnumerator(name: "theta", value: 5)
!16 = !DIEnumerator(name: "psi", value: 6)
!17 = !DIEnumerator(name: "zeta", value: 7)
!18 = !{i64 2, !"Dwarf Version", i64 4}
!19 = !{i64 2, !"Debug Info Version", i64 3}
!20 = !{i64 2, !"wchar_size", i64 2}
!21 = !DILocation(line: 20, scope: !5)
!22 = !DILocalVariable(name: "as", scope: !5, file: !2, line: 11, type: !23)
!23 = !DIDerivedType(tag: DW_TAG_set_type, name: "SS", scope: !2, file: !2, line: 11, baseType: !24, size: 64, align: 64)
!24 = !DIBasicType(name: "SR", size: 8, encoding: DW_ATE_signed)
; CHECK:         DW_TAG_set_type
; CHECK:           DW_AT_type{{.*}}"SR"
; CHECK:           DW_AT_name      ("SS")
; CHECK:           DW_AT_byte_size (0x08)
!25 = !DILocation(line: 11, scope: !5)
!26 = !DILocalVariable(name: "bs", scope: !5, file: !2, line: 11, type: !27)
!27 = !DIDerivedType(tag: DW_TAG_set_type, name: "ST", scope: !2, file: !2, line: 11, baseType: !4, size: 64, align: 64)
; CHECK:         DW_TAG_set_type
; CHECK:           DW_AT_type{{.*}}"Enum"
; CHECK:           DW_AT_name      ("ST")
; CHECK:           DW_AT_byte_size (0x08)
!28 = !DILocation(line: 17, scope: !5)
!29 = !DILocation(line: 18, scope: !5)
!30 = distinct !DISubprogram(name: "Main_M3", linkageName: "Main_M3", scope: !2, file: !2, line: 22, type: !31, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !8)
!31 = !DISubroutineType(types: !32)
!32 = !{!33, !35}
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "ADDR", baseType: !34, size: 64, align: 64)
!34 = !DICompositeType(tag: DW_TAG_class_type, name: "ADDR__HeapObject", scope: !5, file: !2, line: 22, size: 64, align: 64, elements: !7, identifier: "AJWxb1")
!35 = !DIBasicType(name: "INTEGER", size: 64, encoding: DW_ATE_signed)
!36 = !DILocation(line: 23, scope: !30)
!37 = !DILocalVariable(name: "mode", arg: 1, scope: !30, file: !2, line: 22, type: !35)
!38 = !DILocation(line: 22, scope: !30)

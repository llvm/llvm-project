; Test set representation in DWARF debug info:

; RUN: llc -debugger-tune=gdb -dwarf-version=4 -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump -debug-info %t.o | FileCheck %s --check-prefix=CHECK

; ModuleID = 'Main.mb'
source_filename = "../src/Main.m3"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-p:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%M_Const_struct = type { [8 x i8], i64, [16 x i8], [7 x i8], [1 x i8], [4 x i8], [4 x i8], ptr, ptr, ptr, ptr, [8 x i8], [14 x i8], [2 x i8] }
%M_Main_struct = type { ptr, [32 x i8], ptr, [24 x i8], ptr, [8 x i8], ptr, i64, [8 x i8], ptr, ptr, [8 x i8], ptr, [8 x i8] }
%struct.0 = type { [32 x i8] }

@M_Const = internal constant %M_Const_struct { [8 x i8] zeroinitializer, i64 65546, [16 x i8] zeroinitializer, [7 x i8] c"Main_M3", [1 x i8] zeroinitializer, [4 x i8] c"Test", [4 x i8] zeroinitializer, ptr @Main_M3, ptr getelementptr inbounds (i8, ptr @M_Const, i64 32), ptr @Main__Test, ptr getelementptr inbounds (i8, ptr @M_Const, i64 40), [8 x i8] zeroinitializer, [14 x i8] c"../src/Main.m3", [2 x i8] zeroinitializer }, align 8
@M_Main = internal global %M_Main_struct { ptr getelementptr inbounds (i8, ptr @M_Const, i64 88), [32 x i8] zeroinitializer, ptr getelementptr inbounds (i8, ptr @M_Const, i64 48), [24 x i8] zeroinitializer, ptr getelementptr inbounds (i8, ptr @M_Main, i64 104), [8 x i8] zeroinitializer, ptr @Main_M3, i64 3, [8 x i8] zeroinitializer, ptr @Main_I3, ptr getelementptr inbounds (i8, ptr @M_Main, i64 128), [8 x i8] zeroinitializer, ptr @RTHooks_I3, [8 x i8] zeroinitializer }, align 8
@m3_jmpbuf_size = external global i64, align 8

declare ptr @__m3_personality_v0()

declare ptr @Main_I3()

declare ptr @RTHooks_I3()

define void @Main__Test() #0 !dbg !18 {
entry:
  %as = alloca i64, align 8
  %bs = alloca i8, align 1
  %sc = alloca %struct.0, align 8
  %sb = alloca i8, align 1
  br label %second, !dbg !22

second:                                           ; preds = %entry
    #dbg_declare(ptr %as, !23, !DIExpression(), !27)
    #dbg_declare(ptr %bs, !28, !DIExpression(), !27)
    #dbg_declare(ptr %sc, !30, !DIExpression(), !27)
    #dbg_declare(ptr %sb, !33, !DIExpression(), !27)
  store i8 3, ptr %sb, align 1, !dbg !36
  store i64 36028797018972298, ptr %as, align 8, !dbg !37
  store i8 85, ptr %bs, align 1, !dbg !38
  call void @llvm.memmove.p0.p0.i64(ptr align 8 %sc, ptr align 8 @M_Const, i64 32, i1 false), !dbg !39
  ret void, !dbg !22
}

declare ptr @alloca()

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly captures(none), ptr readonly captures(none), i64, i1 immarg) #1

define ptr @Main_M3(i64 %mode) #0 !dbg !40 {
entry:
  %mode1 = alloca i64, align 8
  store i64 %mode, ptr %mode1, align 8
  br label %second, !dbg !45

second:                                           ; preds = %entry
    #dbg_declare(ptr %mode1, !46, !DIExpression(), !47)
  %v.3 = load i64, ptr %mode1, align 8, !dbg !47
  %icmp = icmp eq i64 %v.3, 0, !dbg !47
  br i1 %icmp, label %if_1, label %else_1, !dbg !47

else_1:                                           ; preds = %second
  call void @Main__Test(), !dbg !45
  br label %if_1, !dbg !45

if_1:                                             ; preds = %else_1, %second
  ret ptr @M_Main, !dbg !45
}

attributes #0 = { "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14, !15, !16, !17}

!0 = distinct !DICompileUnit(language: DW_LANG_Modula3, file: !1, producer: "cm3", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "Main.m3", directory: "/home/peter/cm3/settest/src")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Enum", scope: !1, file: !1, line: 1, size: 8, align: 8, elements: !4)
!4 = !{!5, !6, !7, !8, !9, !10, !11, !12}
!5 = !DIEnumerator(name: "alpha", value: 0)
!6 = !DIEnumerator(name: "beta", value: 1)
!7 = !DIEnumerator(name: "gamma", value: 2)
!8 = !DIEnumerator(name: "delta", value: 3)
!9 = !DIEnumerator(name: "epsilon", value: 4)
!10 = !DIEnumerator(name: "theta", value: 5)
!11 = !DIEnumerator(name: "psi", value: 6)
!12 = !DIEnumerator(name: "zeta", value: 7)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 2, !"wchar_size", i32 2}
!16 = !{i32 2, !"PIC Level", i32 2}
!17 = !{i32 2, !"PIE Level", i32 2}
!18 = distinct !DISubprogram(name: "Test", linkageName: "Main__Test", scope: !1, file: !1, line: 18, type: !19, scopeLine: 18, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !21)
!19 = !DISubroutineType(types: !20)
!20 = !{null}
!21 = !{}
!22 = !DILocation(line: 30, scope: !18)
!23 = !DILocalVariable(name: "as", scope: !18, file: !1, line: 19, type: !24)
!24 = !DIDerivedType(tag: DW_TAG_set_type, name: "SS", scope: !1, file: !1, line: 1, baseType: !25, size: 64, align: 64)
; CHECK:         DW_TAG_set_type
; CHECK:           DW_AT_type{{.*}}"SR"
; CHECK:           DW_AT_name      ("SS")
; CHECK:           DW_AT_byte_size (0x08)
!25 = !DISubrangeType(name: "SR", scope: !1, file: !1, line: 1, size: 8, align: 8, baseType: !26, lowerBound: i64 0, upperBound: i64 55)
!26 = !DIBasicType(name: "SR_BASE", size: 8, encoding: DW_ATE_signed)
!27 = !DILocation(line: 18, scope: !18)
!28 = !DILocalVariable(name: "bs", scope: !18, file: !1, line: 19, type: !29)
!29 = !DIDerivedType(tag: DW_TAG_set_type, name: "ST", scope: !1, file: !1, line: 1, baseType: !3, size: 8, align: 8)
; CHECK:         DW_TAG_set_type
; CHECK:           DW_AT_type{{.*}}"Enum"
; CHECK:           DW_AT_name      ("ST")
; CHECK:           DW_AT_byte_size (0x01)
!30 = !DILocalVariable(name: "sc", scope: !18, file: !1, line: 19, type: !31)
!31 = !DIDerivedType(tag: DW_TAG_set_type, name: "SC", scope: !1, file: !1, line: 1, baseType: !32, size: 256, align: 64)
; CHECK:         DW_TAG_set_type
; CHECK:           DW_AT_type{{.*}}"CHAR"
; CHECK:           DW_AT_name      ("SC")
; CHECK:           DW_AT_byte_size (0x20)
!32 = !DIBasicType(name: "CHAR", size: 8, encoding: DW_ATE_unsigned_char)
!33 = !DILocalVariable(name: "sb", scope: !18, file: !1, line: 19, type: !34)
!34 = !DIDerivedType(tag: DW_TAG_set_type, name: "SB", scope: !1, file: !1, line: 1, baseType: !35, size: 8, align: 8)
!35 = !DIBasicType(name: "BOOLEAN", size: 8, encoding: DW_ATE_boolean)
!36 = !DILocation(line: 25, scope: !18)
!37 = !DILocation(line: 26, scope: !18)
!38 = !DILocation(line: 27, scope: !18)
!39 = !DILocation(line: 28, scope: !18)
!40 = distinct !DISubprogram(name: "Main_M3", linkageName: "Main_M3", scope: !1, file: !1, line: 32, type: !41, scopeLine: 32, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !21)
!41 = !DISubroutineType(types: !42)
!42 = !{!43, !44}
!43 = !DICompositeType(tag: DW_TAG_class_type, name: "ADDR", scope: !1, file: !1, line: 1, size: 64, align: 64, elements: !21, identifier: "AJWxb1")
!44 = !DIBasicType(name: "INTEGER", size: 64, encoding: DW_ATE_signed)
!45 = !DILocation(line: 33, scope: !40)
!46 = !DILocalVariable(name: "mode", arg: 1, scope: !40, file: !1, line: 32, type: !44)
!47 = !DILocation(line: 32, scope: !40)

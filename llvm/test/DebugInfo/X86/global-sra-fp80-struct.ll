; RUN: opt -S -globalopt < %s | FileCheck %s
source_filename = "struct.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.mystruct = type { x86_fp80, i32, [12 x i8] }

; Generated from:
;
; static struct mystruct {
;     long double a;
;     int b;
; } static_struct;
; void __attribute__((nodebug)) foo(int in) { static_struct.a = in; }
; void __attribute__((nodebug)) bar(int in) { static_struct.b = in; }
; int main(int argc, char **argv)
; {
;     foo(argv[0][1]);
;     bar(argv[0][1]);
;     return (static_struct.a + static_struct.b) > 0;
; }
;
; using clang -O0 -g2 -S -emit-llvm

@static_struct = internal global %struct.mystruct zeroinitializer, align 16, !dbg !0

; CHECK: @static_struct.0 = internal unnamed_addr global x86_fp80 0xK00000000000000000000, align 16, !dbg ![[EL0:.*]]
; CHECK: @static_struct.1 = internal unnamed_addr global i32 0, align 16, !dbg ![[EL1:.*]]

; CHECK: ![[EL0]] = !DIGlobalVariableExpression(var: ![[VAR:.*]], expr: !DIExpression(DW_OP_LLVM_fragment, 0, 128))
; CHECK: ![[VAR]] = distinct !DIGlobalVariable(name: "static_struct"
; CHECK: ![[EL1]] = !DIGlobalVariableExpression(var: ![[VAR]], expr: !DIExpression(DW_OP_LLVM_fragment, 128, 32))

; Function Attrs: noinline nounwind optnone uwtable
define void @foo(i32 %in) #0 {
entry:
  %in.addr = alloca i32, align 4
  store i32 %in, i32* %in.addr, align 4
  %0 = load i32, i32* %in.addr, align 4
  %conv = sitofp i32 %0 to x86_fp80
  store x86_fp80 %conv, x86_fp80* getelementptr inbounds (%struct.mystruct, %struct.mystruct* @static_struct, i32 0, i32 0), align 16
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define void @bar(i32 %in) #0 {
entry:
  %in.addr = alloca i32, align 4
  store i32 %in, i32* %in.addr, align 4
  %0 = load i32, i32* %in.addr, align 4
  store i32 %0, i32* getelementptr inbounds (%struct.mystruct, %struct.mystruct* @static_struct, i32 0, i32 1), align 16
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define i32 @main(i32 %argc, i8** %argv) #0 !dbg !16 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !22, metadata !DIExpression()), !dbg !23
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !24, metadata !DIExpression()), !dbg !25
  %0 = load i8**, i8*** %argv.addr, align 8, !dbg !26
  %arrayidx = getelementptr inbounds i8*, i8** %0, i64 0, !dbg !26
  %1 = load i8*, i8** %arrayidx, align 8, !dbg !26
  %arrayidx1 = getelementptr inbounds i8, i8* %1, i64 1, !dbg !26
  %2 = load i8, i8* %arrayidx1, align 1, !dbg !26
  %conv = sext i8 %2 to i32, !dbg !26
  call void @foo(i32 %conv), !dbg !27
  %3 = load i8**, i8*** %argv.addr, align 8, !dbg !28
  %arrayidx2 = getelementptr inbounds i8*, i8** %3, i64 0, !dbg !28
  %4 = load i8*, i8** %arrayidx2, align 8, !dbg !28
  %arrayidx3 = getelementptr inbounds i8, i8* %4, i64 1, !dbg !28
  %5 = load i8, i8* %arrayidx3, align 1, !dbg !28
  %conv4 = sext i8 %5 to i32, !dbg !28
  call void @bar(i32 %conv4), !dbg !29
  %6 = load x86_fp80, x86_fp80* getelementptr inbounds (%struct.mystruct, %struct.mystruct* @static_struct, i32 0, i32 0), align 16, !dbg !30
  %7 = load i32, i32* getelementptr inbounds (%struct.mystruct, %struct.mystruct* @static_struct, i32 0, i32 1), align 16, !dbg !31
  %conv5 = sitofp i32 %7 to x86_fp80, !dbg !32
  %add = fadd x86_fp80 %6, %conv5, !dbg !33
  %cmp = fcmp ogt x86_fp80 %add, 0xK00000000000000000000, !dbg !34
  %conv6 = zext i1 %cmp to i32, !dbg !34
  ret i32 %conv6, !dbg !35
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone uwtable }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "static_struct", scope: !2, file: !3, line: 4, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "struct.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "mystruct", file: !3, line: 1, size: 256, elements: !7)
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !6, file: !3, line: 2, baseType: !9, size: 128)
!9 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !6, file: !3, line: 3, baseType: !11, size: 32, offset: 128)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{!"clang version 7.0.0"}
!16 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 7, type: !17, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !2, retainedNodes: !4)
!17 = !DISubroutineType(types: !18)
!18 = !{!11, !11, !19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64)
!21 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!22 = !DILocalVariable(name: "argc", arg: 1, scope: !16, file: !3, line: 7, type: !11)
!23 = !DILocation(line: 7, column: 14, scope: !16)
!24 = !DILocalVariable(name: "argv", arg: 2, scope: !16, file: !3, line: 7, type: !19)
!25 = !DILocation(line: 7, column: 27, scope: !16)
!26 = !DILocation(line: 9, column: 9, scope: !16)
!27 = !DILocation(line: 9, column: 5, scope: !16)
!28 = !DILocation(line: 10, column: 9, scope: !16)
!29 = !DILocation(line: 10, column: 5, scope: !16)
!30 = !DILocation(line: 11, column: 27, scope: !16)
!31 = !DILocation(line: 11, column: 45, scope: !16)
!32 = !DILocation(line: 11, column: 31, scope: !16)
!33 = !DILocation(line: 11, column: 29, scope: !16)
!34 = !DILocation(line: 11, column: 48, scope: !16)
!35 = !DILocation(line: 11, column: 5, scope: !16)

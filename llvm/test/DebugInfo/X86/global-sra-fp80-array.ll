; RUN: opt -S -globalopt < %s | FileCheck %s
source_filename = "array.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.mystruct = type { i32, i64 }

; Generated from:
;
; static long double array[2];
; void __attribute__((nodebug)) foo(int in) { array[0] = in; }
; void __attribute__((nodebug)) bar(int in) { array[1] = in; }
; int main(int argc, char **argv)
; {
;   foo(argv[0][1]);
;   bar(argv[0][1]);
;   return (array[0] + array[1]) > 0;
; }
;
; using clang -O0 -g2 -S -emit-llvm

@array = internal global [2 x x86_fp80] zeroinitializer, align 16, !dbg !0

; CHECK: @array.0 = internal unnamed_addr global x86_fp80 0xK00000000000000000000, align 16, !dbg ![[EL0:.*]]
; CHECK: @array.1 = internal unnamed_addr global x86_fp80 0xK00000000000000000000, align 16, !dbg ![[EL1:.*]]
;
; CHECK: ![[EL0]] = !DIGlobalVariableExpression(var: ![[VAR:.*]], expr: !DIExpression(DW_OP_LLVM_fragment, 0, 128))
; CHECK: ![[VAR]] = distinct !DIGlobalVariable(name: "array"
; CHECK: ![[EL1]] = !DIGlobalVariableExpression(var: ![[VAR]], expr: !DIExpression(DW_OP_LLVM_fragment, 128, 128))

; Function Attrs: noinline nounwind optnone uwtable
define void @foo(i32 %in) #0 {
entry:
  %in.addr = alloca i32, align 4
  store i32 %in, i32* %in.addr, align 4
  %0 = load i32, i32* %in.addr, align 4
  %conv = sitofp i32 %0 to x86_fp80
  store x86_fp80 %conv, x86_fp80* getelementptr inbounds ([2 x x86_fp80], [2 x x86_fp80]* @array, i64 0, i64 0), align 16
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define void @bar(i32 %in) #0 {
entry:
  %in.addr = alloca i32, align 4
  store i32 %in, i32* %in.addr, align 4
  %0 = load i32, i32* %in.addr, align 4
  %conv = sitofp i32 %0 to x86_fp80
  store x86_fp80 %conv, x86_fp80* getelementptr inbounds ([2 x x86_fp80], [2 x x86_fp80]* @array, i64 0, i64 1), align 16
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define i32 @main(i32 %argc, i8** %argv) #0 !dbg !14 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !21, metadata !DIExpression()), !dbg !22
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !23, metadata !DIExpression()), !dbg !24
  %0 = load i8**, i8*** %argv.addr, align 8, !dbg !25
  %arrayidx = getelementptr inbounds i8*, i8** %0, i64 0, !dbg !25
  %1 = load i8*, i8** %arrayidx, align 8, !dbg !25
  %arrayidx1 = getelementptr inbounds i8, i8* %1, i64 1, !dbg !25
  %2 = load i8, i8* %arrayidx1, align 1, !dbg !25
  %conv = sext i8 %2 to i32, !dbg !25
  call void @foo(i32 %conv), !dbg !26
  %3 = load i8**, i8*** %argv.addr, align 8, !dbg !27
  %arrayidx2 = getelementptr inbounds i8*, i8** %3, i64 0, !dbg !27
  %4 = load i8*, i8** %arrayidx2, align 8, !dbg !27
  %arrayidx3 = getelementptr inbounds i8, i8* %4, i64 1, !dbg !27
  %5 = load i8, i8* %arrayidx3, align 1, !dbg !27
  %conv4 = sext i8 %5 to i32, !dbg !27
  call void @bar(i32 %conv4), !dbg !28
  %6 = load x86_fp80, x86_fp80* getelementptr inbounds ([2 x x86_fp80], [2 x x86_fp80]* @array, i64 0, i64 0), align 16, !dbg !29
  %7 = load x86_fp80, x86_fp80* getelementptr inbounds ([2 x x86_fp80], [2 x x86_fp80]* @array, i64 0, i64 1), align 16, !dbg !30
  %add = fadd x86_fp80 %6, %7, !dbg !31
  %cmp = fcmp ogt x86_fp80 %add, 0xK00000000000000000000, !dbg !32
  %conv5 = zext i1 %cmp to i32, !dbg !32
  ret i32 %conv5, !dbg !33
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone uwtable }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "array", scope: !2, file: !3, line: 1, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "array.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 256, elements: !8)
!7 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!8 = !{!9}
!9 = !DISubrange(count: 2)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 7.0.0"}
!14 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 4, type: !15, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !2, variables: !4)
!15 = !DISubroutineType(types: !16)
!16 = !{!17, !17, !18}
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!20 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!21 = !DILocalVariable(name: "argc", arg: 1, scope: !14, file: !3, line: 4, type: !17)
!22 = !DILocation(line: 4, column: 14, scope: !14)
!23 = !DILocalVariable(name: "argv", arg: 2, scope: !14, file: !3, line: 4, type: !18)
!24 = !DILocation(line: 4, column: 27, scope: !14)
!25 = !DILocation(line: 6, column: 7, scope: !14)
!26 = !DILocation(line: 6, column: 3, scope: !14)
!27 = !DILocation(line: 7, column: 7, scope: !14)
!28 = !DILocation(line: 7, column: 3, scope: !14)
!29 = !DILocation(line: 8, column: 11, scope: !14)
!30 = !DILocation(line: 8, column: 22, scope: !14)
!31 = !DILocation(line: 8, column: 20, scope: !14)
!32 = !DILocation(line: 8, column: 32, scope: !14)
!33 = !DILocation(line: 8, column: 3, scope: !14)

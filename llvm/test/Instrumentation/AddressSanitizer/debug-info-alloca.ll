; Checks that asan prologue does not add debug locations, which would
; fool findPrologueEndLoc because it sets the end of the prologue to the
; first instruction.  Breaking on the instrumented function in a debugger
; would then stop at that instruction, before the prologue is finished.

; RUN: opt < %s -passes='asan-pipeline' -S | FileCheck %s
; 1: void f(int *arg) {
; 2: }
; 3: int main(int argc, char **argv) {
; 4:   f(&argc);
; 5: }
; clang 1.cc -g -S -emit-llvm -o - | sed 's/#0 = {/#0 = { sanitize_address/'

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local i32 @main(i32 %argc, i8** %argv) #0 !dbg !15 {
entry:
; No suffix like !dbg !123
; CHECK: %asan_local_stack_base = alloca i64, align 8{{$}}
; CHECK:     %3 = call i64 @__asan_stack_malloc_0(i64 64){{$}}
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !21, metadata !DIExpression()), !dbg !22
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !23, metadata !DIExpression()), !dbg !24
  call void @f(i32* %argc.addr), !dbg !25
  ret i32 0, !dbg !26
}

define dso_local void @f(i32* %arg) #0 !dbg !7 {
entry:
  %arg.addr = alloca i32*, align 8
  store i32* %arg, i32** %arg.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %arg.addr, metadata !12, metadata !DIExpression()), !dbg !13
  ret void, !dbg !14
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { sanitize_address noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (git@github.com:llvm/llvm-project 1ac700cdef787383ad49a0e37d9894491ef19480)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "2.c", directory: "/home/builduser")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0 (git@github.com:llvm/llvm-project 1ac700cdef787383ad49a0e37d9894491ef19480)"}
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "arg", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!13 = !DILocation(line: 1, column: 13, scope: !7)
!14 = !DILocation(line: 2, column: 1, scope: !7)
!15 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !16, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!16 = !DISubroutineType(types: !17)
!17 = !{!11, !11, !18}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!20 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!21 = !DILocalVariable(name: "argc", arg: 1, scope: !15, file: !1, line: 3, type: !11)
!22 = !DILocation(line: 3, column: 14, scope: !15)
!23 = !DILocalVariable(name: "argv", arg: 2, scope: !15, file: !1, line: 3, type: !18)
!24 = !DILocation(line: 3, column: 27, scope: !15)
!25 = !DILocation(line: 4, column: 3, scope: !15)
!26 = !DILocation(line: 5, column: 1, scope: !15)

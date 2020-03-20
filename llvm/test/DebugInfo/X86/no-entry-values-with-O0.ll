; RUN: llc -O0  -dwarf-version=5 -debugger-tune=lldb -march=x86-64 -filetype=obj < %s \
; RUN:  | llvm-dwarfdump - | FileCheck --implicit-check-not=DW_OP_entry_value %s
; RUN: llc -O0  -dwarf-version=5 -debugger-tune=gdb -march=x86-64 -filetype=obj < %s \
; RUN:  | llvm-dwarfdump - | FileCheck --implicit-check-not=DW_OP_entry_value %s

; The call-site-params are created iff corresponding DISubprogram contains
; the AllCallsDescribed DIFlag.
; CHECK-NOT: DW_TAG_call_site_param

; Genarated with:
; clang -gdwarf-5 -O0 test.c -S -emit-llvm
;
; ModuleID = 'test.c'
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @fn1(i32 %x, i32 %y) !dbg !7 {
entry:
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  %u = alloca i32, align 4
  %a = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %x.addr, metadata !11, metadata !DIExpression()), !dbg !12
  store i32 %y, i32* %y.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %y.addr, metadata !13, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.declare(metadata i32* %u, metadata !15, metadata !DIExpression()), !dbg !16
  %0 = load i32, i32* %x.addr, align 4, !dbg !16
  %1 = load i32, i32* %y.addr, align 4, !dbg !16
  %add = add nsw i32 %0, %1, !dbg !16
  store i32 %add, i32* %u, align 4, !dbg !16
  %2 = load i32, i32* %x.addr, align 4, !dbg !17
  %cmp = icmp sgt i32 %2, 1, !dbg !17
  br i1 %cmp, label %if.then, label %if.else, !dbg !16

if.then:                                          ; preds = %entry
  %3 = load i32, i32* %u, align 4, !dbg !17
  %add1 = add nsw i32 %3, 1, !dbg !17
  store i32 %add1, i32* %u, align 4, !dbg !17
  br label %if.end, !dbg !17

if.else:                                          ; preds = %entry
  %4 = load i32, i32* %u, align 4, !dbg !17
  %add2 = add nsw i32 %4, 2, !dbg !17
  store i32 %add2, i32* %u, align 4, !dbg !17
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @llvm.dbg.declare(metadata i32* %a, metadata !19, metadata !DIExpression()), !dbg !16
  store i32 7, i32* %a, align 4, !dbg !16
  %5 = load i32, i32* %a, align 4, !dbg !16
  call void @fn2(i32 %5), !dbg !16
  %6 = load i32, i32* %u, align 4, !dbg !16
  %dec = add nsw i32 %6, -1, !dbg !16
  store i32 %dec, i32* %u, align 4, !dbg !16
  ret void, !dbg !16
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare dso_local void @fn2(i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 11.0.0"}
!7 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "x", arg: 1, scope: !7, file: !1, line: 5, type: !10)
!12 = !DILocation(line: 5, column: 10, scope: !7)
!13 = !DILocalVariable(name: "y", arg: 2, scope: !7, file: !1, line: 5, type: !10)
!14 = !DILocation(line: 5, column: 17, scope: !7)
!15 = !DILocalVariable(name: "u", scope: !7, file: !1, line: 6, type: !10)
!16 = !DILocation(line: 6, column: 7, scope: !7)
!17 = !DILocation(line: 7, column: 7, scope: !18)
!18 = distinct !DILexicalBlock(scope: !7, file: !1, line: 7, column: 7)
!19 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 11, type: !10)

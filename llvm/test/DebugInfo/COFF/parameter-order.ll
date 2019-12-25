; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s --check-prefix=ASM

; Make sure variables come out in the right order. windbg builds the prototype
; from the order of these records.

; C++ source to regenerate:
; $ cat t.cpp
; int f(int a, int b, int c) {
;   int d = 4;
;   int e = 5;
;   return a + b + c + d + e;
; }
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll

; ASM: .short  4414                    # Record kind: S_LOCAL
; ASM: .long   116                     # TypeIndex
; ASM: .short  1                       # Flags
; ASM: .asciz  "a"
; ASM: .cv_def_range

; ASM: .short  4414                    # Record kind: S_LOCAL
; ASM: .long   116                     # TypeIndex
; ASM: .short  1                       # Flags
; ASM: .asciz  "b"
; ASM: .cv_def_range

; ASM: .short  4414                    # Record kind: S_LOCAL
; ASM: .long   116                     # TypeIndex
; ASM: .short  1                       # Flags
; ASM: .asciz  "c"
; ASM: .cv_def_range

; ASM: .short  4414                    # Record kind: S_LOCAL
; ASM: .long   116                     # TypeIndex
; ASM: .short  0                       # Flags
; ASM: .asciz  "d"
; ASM: .cv_def_range

; ASM: .short  4414                    # Record kind: S_LOCAL
; ASM: .long   116                     # TypeIndex
; ASM: .short  0                       # Flags
; ASM: .asciz  "e"
; ASM: .cv_def_range


; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.23918"

; Function Attrs: nounwind uwtable
define i32 @"\01?f@@YAHHHH@Z"(i32 %a, i32 %b, i32 %c) #0 !dbg !7 {
entry:
  %c.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %a.addr = alloca i32, align 4
  %d = alloca i32, align 4
  %e = alloca i32, align 4
  store i32 %c, i32* %c.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %c.addr, metadata !11, metadata !12), !dbg !13
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !14, metadata !12), !dbg !15
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !16, metadata !12), !dbg !17
  call void @llvm.dbg.declare(metadata i32* %d, metadata !18, metadata !12), !dbg !19
  store i32 4, i32* %d, align 4, !dbg !19
  call void @llvm.dbg.declare(metadata i32* %e, metadata !20, metadata !12), !dbg !21
  store i32 5, i32* %e, align 4, !dbg !21
  %0 = load i32, i32* %a.addr, align 4, !dbg !22
  %1 = load i32, i32* %b.addr, align 4, !dbg !23
  %add = add nsw i32 %0, %1, !dbg !24
  %2 = load i32, i32* %c.addr, align 4, !dbg !25
  %add1 = add nsw i32 %add, %2, !dbg !26
  %3 = load i32, i32* %d, align 4, !dbg !27
  %add2 = add nsw i32 %add1, %3, !dbg !28
  %4 = load i32, i32* %e, align 4, !dbg !29
  %add3 = add nsw i32 %add2, %4, !dbg !30
  ret i32 %add3, !dbg !31
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 273683) (llvm/trunk 273691)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 3.9.0 (trunk 273683) (llvm/trunk 273691)"}
!7 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAHHHH@Z", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "c", arg: 3, scope: !7, file: !1, line: 1, type: !10)
!12 = !DIExpression()
!13 = !DILocation(line: 1, column: 25, scope: !7)
!14 = !DILocalVariable(name: "b", arg: 2, scope: !7, file: !1, line: 1, type: !10)
!15 = !DILocation(line: 1, column: 18, scope: !7)
!16 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!17 = !DILocation(line: 1, column: 11, scope: !7)
!18 = !DILocalVariable(name: "d", scope: !7, file: !1, line: 2, type: !10)
!19 = !DILocation(line: 2, column: 7, scope: !7)
!20 = !DILocalVariable(name: "e", scope: !7, file: !1, line: 3, type: !10)
!21 = !DILocation(line: 3, column: 7, scope: !7)
!22 = !DILocation(line: 4, column: 10, scope: !7)
!23 = !DILocation(line: 4, column: 14, scope: !7)
!24 = !DILocation(line: 4, column: 12, scope: !7)
!25 = !DILocation(line: 4, column: 18, scope: !7)
!26 = !DILocation(line: 4, column: 16, scope: !7)
!27 = !DILocation(line: 4, column: 22, scope: !7)
!28 = !DILocation(line: 4, column: 20, scope: !7)
!29 = !DILocation(line: 4, column: 26, scope: !7)
!30 = !DILocation(line: 4, column: 24, scope: !7)
!31 = !DILocation(line: 4, column: 3, scope: !7)


; REQUIRES: x86_64-linux
; RUN: rm -rf %t && mkdir %t
; RUN: mkdir -p %t/gnuLink
; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -filetype=obj -o main.o
; RUN: llvm-objcopy --only-keep-debug main.o main.o.debuginfo
; RUN: llvm-objcopy --strip-debug --add-gnu-debuglink=main.o.debuginfo main.o
; RUN: llvm-dwarfdump --debug-info -r 0  main.o | FileCheck --check-prefix=DWARFDUMP %s
; RUN: llvm-gsymutil --convert main.o -o main.gsym | FileCheck --check-prefix=GSYM %s

; DWARFDUMP: DW_TAG_compile_unit
; GSYM: Loaded 2 functions from DWARF.

;; Testing that llvm-dwarfdump and llvm-gsymutil work on a binary from which debug information
;; was stripped after gnu_debug_link is created.
;;clang++ -g2 -O0 main.cpp -c -emit-llvm -S
;; int foo(int i) {
;;   return i + 1;
;; }
;;
;; int main() {
;;   int j = 3;
;;   j = foo(j) + 1;
;;   return j;
;; }


; ModuleID = 'main.cpp'
source_filename = "main.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z3fooi(i32 noundef %i) #0 !dbg !10 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, ptr %i.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %i.addr, metadata !15, metadata !DIExpression()), !dbg !16
  %0 = load i32, ptr %i.addr, align 4, !dbg !17
  %add = add nsw i32 %0, 1, !dbg !18
  ret i32 %add, !dbg !19
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: mustprogress noinline norecurse nounwind optnone uwtable
define dso_local noundef i32 @main() #2 !dbg !20 {
entry:
  %retval = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  call void @llvm.dbg.declare(metadata ptr %j, metadata !23, metadata !DIExpression()), !dbg !24
  store i32 3, ptr %j, align 4, !dbg !24
  %0 = load i32, ptr %j, align 4, !dbg !25
  %call = call noundef i32 @_Z3fooi(i32 noundef %0), !dbg !26
  %add = add nsw i32 %call, 1, !dbg !27
  store i32 %add, ptr %j, align 4, !dbg !28
  %1 = load i32, ptr %j, align 4, !dbg !29
  ret i32 %1, !dbg !30
}

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress noinline norecurse nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 18.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "main.cpp", directory: "/gnuLink", checksumkind: CSK_MD5, checksum: "cc30cb527607311c4a8449e92acdf1c5")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 18.0.0git"}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{}
!15 = !DILocalVariable(name: "i", arg: 1, scope: !10, file: !1, line: 1, type: !13)
!16 = !DILocation(line: 1, column: 13, scope: !10)
!17 = !DILocation(line: 2, column: 10, scope: !10)
!18 = !DILocation(line: 2, column: 12, scope: !10)
!19 = !DILocation(line: 2, column: 3, scope: !10)
!20 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 5, type: !21, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!21 = !DISubroutineType(types: !22)
!22 = !{!13}
!23 = !DILocalVariable(name: "j", scope: !20, file: !1, line: 6, type: !13)
!24 = !DILocation(line: 6, column: 7, scope: !20)
!25 = !DILocation(line: 7, column: 11, scope: !20)
!26 = !DILocation(line: 7, column: 7, scope: !20)
!27 = !DILocation(line: 7, column: 14, scope: !20)
!28 = !DILocation(line: 7, column: 5, scope: !20)
!29 = !DILocation(line: 8, column: 10, scope: !20)
!30 = !DILocation(line: 8, column: 3, scope: !20)

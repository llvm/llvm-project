;; Test to make sure that memprof works and do cloning right when exists direct recursion.
;; Original code looks like:
;; After modify alloc type, there are two direct recursion with hot and cold alloc type.
;; a.cpp
;; extern void foo(int);
;; int b = 10;
;; int* a;
;; int main(){
;;     foo(1);
;;     b = 10;
;;     foo(2);
;;     return 0;
;; }
;; b.cpp
;; extern int b;
;; extern int *a;
;; void foo(int c){
;;     a = new int[1];
;;     if (c&1) {
;;         for (int i = 0; i < 100; ++i)
;;             a[0] = 1;
;;     }
;;     --b;
;;     if (b) {
;;         foo(c);
;;     }
;; }

; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/b.ll >%t/b.o
; RUN: opt -thinlto-bc %t/a.ll >%t/a.o

; RUN: llvm-lto2 run %t/b.o %t/a.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=%t/b.o,_Z3fooi,plx \
; RUN:  -r=%t/b.o,a \
; RUN:  -r=%t/b.o,b \
; RUN:  -r=%t/b.o,_Znam \
; RUN:  -r=%t/a.o,main,plx \
; RUN:  -r=%t/a.o,_Z3fooi \
; RUN:  -r=%t/a.o,a,plx \
; RUN:  -r=%t/a.o,b,plx \
; RUN:  -memprof-dump-ccg \
; RUN:  -o %t2.out 2>&1

; RUN: llvm-dis %t/b.o.thinlto.bc -o - | FileCheck %s --check-prefix=SUMMARY

;; Test direct recursion don't cause assert failed and do cloning right.
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:  -memprof-import-summary=%t/b.o.thinlto.bc \
; RUN:  %t/b.o -S | FileCheck %s --check-prefix=IR

; SUMMARY: stackIds: (1985258834072910425, 2841526434899864997)
; SUMMARY-NOT: stackIds: (1985258834072910425, 1985258834072910425, 2841526434899864997)

; IR: _Z3fooi
; IR: _Z3fooi.memprof.1
; IR: "memprof"="notcold" 
; IR: "memprof"="cold" 

;--- b.ll
; ModuleID = 'b.cpp'
source_filename = "b.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = external local_unnamed_addr global ptr, align 8
@b = external local_unnamed_addr global i32, align 4

; Function Attrs: mustprogress uwtable
define dso_local void @_Z3fooi(i32 noundef %0) local_unnamed_addr #0 !dbg !9 {
  br label %2, !dbg !12

2:                                                ; preds = %7, %1
  %3 = tail call noalias noundef nonnull dereferenceable(4) ptr @_Znam(i64 noundef 4) #2, !dbg !13, !memprof !14, !callsite !55
  store ptr %3, ptr @a, align 8, !dbg !56, !tbaa !57
  %4 = and i32 %0, 1, !dbg !61
  %5 = icmp eq i32 %4, 0, !dbg !62
  br i1 %5, label %7, label %6, !dbg !62

6:                                                ; preds = %2
  store i32 1, ptr %3, align 4, !tbaa !63
  br label %7, !dbg !65

7:                                                ; preds = %6, %2
  %8 = load i32, ptr @b, align 4, !dbg !65, !tbaa !63
  %9 = add nsw i32 %8, -1, !dbg !65
  store i32 %9, ptr @b, align 4, !dbg !65, !tbaa !63
  %10 = icmp eq i32 %9, 0, !dbg !66
  br i1 %10, label %11, label %2, !dbg !66

11:                                               ; preds = %7
  ret void, !dbg !67
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) local_unnamed_addr #1

attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nobuiltin allocsize(0) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { builtin allocsize(0) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 18.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "b.cpp", directory: "/", checksumkind: CSK_MD5, checksum: "8fa6c585f9d2c35f1a82b920e64bbda2")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang version 18.0.0"}
!9 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !1, file: !1, line: 4, type: !10, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !DILocation(line: 12, column: 9, scope: !9)
!13 = !DILocation(line: 5, column: 9, scope: !9)
!14 = !{!15, !17, !19, !21, !23, !25, !27, !29, !31, !33, !35, !37, !39, !41, !43, !45, !47, !49, !51, !53}
!15 = !{!16, !"hot"}
!16 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 2841526434899864997}
!17 = !{!18, !"cold"}
!18 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 8256520048276991898}
!19 = !{!20, !"hot"}
!20 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 2841526434899864997}
!21 = !{!22, !"cold"}
!22 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 8256520048276991898}
!23 = !{!24, !"hot"}
!24 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 2841526434899864997}
!25 = !{!26, !"cold"}
!26 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 8256520048276991898}
!27 = !{!28, !"hot"}
!28 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 2841526434899864997}
!29 = !{!30, !"cold"}
!30 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 8256520048276991898}
!31 = !{!32, !"hot"}
!32 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 2841526434899864997}
!33 = !{!34, !"cold"}
!34 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 8256520048276991898}
!35 = !{!36, !"hot"}
!36 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 2841526434899864997}
!37 = !{!38, !"cold"}
!38 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 8256520048276991898}
!39 = !{!40, !"hot"}
!40 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 2841526434899864997}
!41 = !{!42, !"cold"}
!42 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 1985258834072910425, i64 8256520048276991898}
!43 = !{!44, !"hot"}
!44 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 2841526434899864997}
!45 = !{!46, !"cold"}
!46 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 1985258834072910425, i64 8256520048276991898}
!47 = !{!48, !"hot"}
!48 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 2841526434899864997}
!49 = !{!50, !"cold"}
!50 = !{i64 -1057479539165743997, i64 1985258834072910425, i64 8256520048276991898}
!51 = !{!52, !"hot"}
!52 = !{i64 -1057479539165743997, i64 2841526434899864997}
!53 = !{!54, !"cold"}
!54 = !{i64 -1057479539165743997, i64 8256520048276991898}
!55 = !{i64 -1057479539165743997}
!56 = !DILocation(line: 5, column: 7, scope: !9)
!57 = !{!58, !58, i64 0}
!58 = !{!"any pointer", !59, i64 0}
!59 = !{!"omnipotent char", !60, i64 0}
!60 = !{!"Simple C++ TBAA"}
!61 = !DILocation(line: 6, column: 10, scope: !9)
!62 = !DILocation(line: 6, column: 9, scope: !9)
!63 = !{!64, !64, i64 0}
!64 = !{!"int", !59, i64 0}
!65 = !DILocation(line: 10, column: 5, scope: !9)
!66 = !DILocation(line: 11, column: 9, scope: !9)
!67 = !DILocation(line: 14, column: 1, scope: !9)

;--- a.ll
; ModuleID = 'a.cpp'
source_filename = "a.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@b = dso_local local_unnamed_addr global i32 10, align 4
@a = dso_local local_unnamed_addr global ptr null, align 8

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 !dbg !9 {
  tail call void @_Z3fooi(i32 noundef 1), !dbg !12, !callsite !13
  store i32 10, ptr @b, align 4, !dbg !14, !tbaa !15
  tail call void @_Z3fooi(i32 noundef 2), !dbg !19, !callsite !20
  ret i32 0, !dbg !21
}

declare !dbg !22 void @_Z3fooi(i32 noundef) local_unnamed_addr #1

attributes #0 = { mustprogress norecurse uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 18.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "a.cpp", directory: "/", checksumkind: CSK_MD5, checksum: "16ecbfa723a07d69c0374cfc704a7c44")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang version 18.0.0"}
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 4, type: !10, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !DILocation(line: 5, column: 5, scope: !9)
!13 = !{i64 2841526434899864997}
!14 = !DILocation(line: 6, column: 7, scope: !9)
!15 = !{!16, !16, i64 0}
!16 = !{!"int", !17, i64 0}
!17 = !{!"omnipotent char", !18, i64 0}
!18 = !{!"Simple C++ TBAA"}
!19 = !DILocation(line: 7, column: 5, scope: !9)
!20 = !{i64 8256520048276991898}
!21 = !DILocation(line: 8, column: 5, scope: !9)
!22 = !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !1, file: !1, line: 1, type: !10, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
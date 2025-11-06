; RUN: opt -mcpu=cyclone -mtriple=arm64-apple-ios -passes=loop-data-prefetch \
; RUN:     -pass-remarks=loop-data-prefetch -S -max-prefetch-iters-ahead=100 \
; RUN:     -pass-remarks-with-hotness \
; RUN:     < %s 2>&1 | FileCheck %s

; ModuleID = '/tmp/s.c'
source_filename = "/tmp/s.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios5.0.0"

;   1	struct MyStruct {
;   2	  int field;
;   3	  char kk[2044];
;   4	} *my_struct;
;   5
;   6	int f(struct MyStruct *p, int N) {
;   7	  int total = 0;
;   8	  for (int i = 0; i < N; i++) {
;   9	    total += my_struct[i].field;
;  10	  }
;  11	  return total;
;  12	}

; CHECK: remark: /tmp/s.c:9:27: prefetched memory access (hotness: 600)

%struct.MyStruct = type { i32, [2044 x i8] }

@my_struct = common global ptr null, align 8

define i32 @f(ptr nocapture readnone %p, i32 %N) !dbg !6 !prof !21 {
entry:
  %cmp6 = icmp sgt i32 %N, 0, !dbg !8
  br i1 %cmp6, label %for.body.lr.ph, label %for.cond.cleanup, !dbg !9, !prof !22

for.body.lr.ph:                                   ; preds = %entry
  %0 = load ptr, ptr @my_struct, align 8, !dbg !10, !tbaa !11
  br label %for.body, !dbg !9

for.cond.cleanup:                                 ; preds = %for.body, %entry
  %total.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %total.0.lcssa, !dbg !15

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %total.07 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %field = getelementptr inbounds %struct.MyStruct, ptr %0, i64 %indvars.iv, i32 0, !dbg !16
  %1 = load i32, ptr %field, align 4, !dbg !16, !tbaa !17
  %add = add nsw i32 %1, %total.07, !dbg !20
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !9
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !9
  %exitcond = icmp eq i32 %lftr.wideiv, %N, !dbg !9
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !dbg !9, !prof !23
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/s.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"PIC Level", i32 2}
!5 = !{!"clang version 3.9.0"}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 6, type: !7, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 8, column: 21, scope: !6)
!9 = !DILocation(line: 8, column: 3, scope: !6)
!10 = !DILocation(line: 9, column: 14, scope: !6)
!11 = !{!12, !12, i64 0}
!12 = !{!"any pointer", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C/C++ TBAA"}
!15 = !DILocation(line: 11, column: 3, scope: !6)
!16 = !DILocation(line: 9, column: 27, scope: !6)
!17 = !{!18, !19, i64 0}
!18 = !{!"MyStruct", !19, i64 0, !13, i64 4}
!19 = !{!"int", !13, i64 0}
!20 = !DILocation(line: 9, column: 11, scope: !6)
!21 = !{!"function_entry_count", i64 6}
!22 = !{!"branch_weights", i32 2000, i32 1}
!23 = !{!"branch_weights", i32 1, i32 99}

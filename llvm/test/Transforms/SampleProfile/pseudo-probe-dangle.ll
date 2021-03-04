; REQUIRES: x86_64-linux
; RUN: opt < %s -passes='pseudo-probe,jump-threading' -S -o %t
; RUN: FileCheck %s < %t --check-prefix=JT
; RUN: llc -pseudo-probe-for-profiling -function-sections <%t -filetype=asm | FileCheck %s --check-prefix=ASM
; RUN: opt < %s -passes='pseudo-probe' -S -o %t1
; RUN: llc -pseudo-probe-for-profiling -stop-after=tailduplication <%t1 | FileCheck %s --check-prefix=MIR-tail
; RUN: opt < %s -passes='pseudo-probe,simplifycfg' -S | FileCheck %s --check-prefix=SC

declare i32 @f1()

define i32 @foo(i1 %cond) {
; JT-LABEL: @foo(
; JT: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 1, i32 0, i64 -1)
; ASM: pseudoprobe	6699318081062747564 1 0 0
	%call = call i32 @f1()
	br i1 %cond, label %T, label %F
T:
	br label %Merge
F:
	br label %Merge
Merge:
;; Check branch T and F are gone, and their probes (probe 2 and 3) are dangling.
; JT-LABEL-NO: T
; JT-LABEL-NO: F
; JT-LABEL: Merge
; JT: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 3, i32 2, i64 -1)
; JT: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 2, i32 2, i64 -1)
; JT: call void @llvm.pseudoprobe(i64 [[#GUID:]], i64 4, i32 0, i64 -1)
; ASM: .pseudoprobe	6699318081062747564 3 0 2
; ASM: .pseudoprobe	6699318081062747564 2 0 2
; ASM: .pseudoprobe	6699318081062747564 4 0 0
	ret i32 %call
}

;; Check block T and F are gone, and their probes (probe 2 and 3) are dangling.
; MIR-tail: bb.0
; MIR-tail: PSEUDO_PROBE [[#GUID:]], 1, 0, 0
; MIR-tail: PSEUDO_PROBE [[#GUID:]], 2, 0, 2
; MIR-tail: PSEUDO_PROBE [[#GUID:]], 3, 0, 2
; MIR-tail: PSEUDO_PROBE [[#GUID:]], 4, 0, 0

define i32 @test(i32 %a, i32 %b, i32 %c) {
;; Check block bb1 and bb2 are gone, and their probes (probe 2 and 3) are dangling.
; SC-LABEL: @test(
; SC-LABEL-NO: bb1
; SC-LABEL-NO: bb2
; SC:    [[T1:%.*]] = icmp eq i32 [[B:%.*]], 0
; SC-DAG:    call void @llvm.pseudoprobe(i64 [[#GUID3:]], i64 2, i32 2, i64 -1)
; SC-DAG:    call void @llvm.pseudoprobe(i64 [[#GUID3]], i64 3, i32 2, i64 -1)
; SC:    [[T2:%.*]] = icmp sgt i32 [[C:%.*]], 1
; SC:    [[T3:%.*]] = add i32 [[A:%.*]], 1
; SC:    [[SPEC_SELECT:%.*]] = select i1 [[T2]], i32 [[T3]], i32 [[A]]
; SC:    [[T4:%.*]] = select i1 [[T1]], i32 [[SPEC_SELECT]], i32 [[B]]
; SC:    [[T5:%.*]] = sub i32 [[T4]], 1
; SC:    ret i32 [[T5]]

entry:
  %t1 = icmp eq i32 %b, 0
  br i1 %t1, label %bb1, label %bb3

bb1:
  %t2 = icmp sgt i32 %c, 1
  br i1 %t2, label %bb2, label %bb3

bb2:
  %t3 = add i32 %a, 1
  br label %bb3

bb3:
  %t4 = phi i32 [ %b, %entry ], [ %a, %bb1 ], [ %t3, %bb2 ]
  %t5 = sub i32 %t4, 1
  ret i32 %t5
}

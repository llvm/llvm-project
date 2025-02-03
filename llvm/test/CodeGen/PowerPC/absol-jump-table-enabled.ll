; NOTE: This test case generates a jump table on PowerPC big and little endian
; NOTE: then verifies that the command line option to enable absolute jump
; NOTE: table works correctly.
; RUN:  llc -mtriple=powerpc64le-unknown-linux-gnu -ppc-min-jump-table-entries=4 -o - \
; RUN:      -ppc-use-absolute-jumptables -ppc-asm-full-reg-names \
; RUN:      -verify-machineinstrs %s | FileCheck %s -check-prefix=CHECK-LE
; RUN:  llc -mtriple=powerpc64-unknown-linux-gnu -ppc-min-jump-table-entries=4 -o - \
; RUN:      -ppc-use-absolute-jumptables -ppc-asm-full-reg-names \
; RUN:      -verify-machineinstrs %s | FileCheck %s -check-prefix=CHECK-BE
; RUN:  llc -mtriple=powerpc64-ibm-aix-xcoff -ppc-min-jump-table-entries=4 -o - \
; RUN:      -ppc-use-absolute-jumptables -ppc-asm-full-reg-names \
; RUN:      -verify-machineinstrs %s | FileCheck %s -check-prefix=CHECK-AIX
; RUN:  llc -mtriple=powerpc64le-unknown-linux-gnu -ppc-min-jump-table-entries=4 -o - \
; RUN:      -ppc-use-absolute-jumptables=true --relocation-model=pic < %s | FileCheck %s \
; RUN:      -check-prefix=CHECK-A-PIC-LE
; RUN:  llc -mtriple=powerpc64le-unknown-linux-gnu -ppc-min-jump-table-entries=4 -o - \
; RUN:      -ppc-use-absolute-jumptables=false --relocation-model=pic < %s | FileCheck %s \
; RUN:      -check-prefix=CHECK-R-PIC-LE
; RUN:  llc -mtriple=powerpc64-unknown-linux-gnu -ppc-min-jump-table-entries=4 -o - \
; RUN:      -ppc-use-absolute-jumptables=true --relocation-model=pic < %s | FileCheck %s \
; RUN:      -check-prefix=CHECK-A-PIC-BE
; RUN:  llc -mtriple=powerpc64-unknown-linux-gnu -ppc-min-jump-table-entries=4 -o - \
; RUN:      -ppc-use-absolute-jumptables=false --relocation-model=pic < %s | FileCheck %s \
; RUN:      -check-prefix=CHECK-R-PIC-BE
; RUN:  llc -mtriple=powerpc64-ibm-aix-xcoff -ppc-min-jump-table-entries=4 -o - \
; RUN:      -ppc-use-absolute-jumptables=true --relocation-model=pic < %s | FileCheck %s \
; RUN:      -check-prefix=CHECK-A-PIC-AIX
; RUN:  llc -mtriple=powerpc64-ibm-aix-xcoff -ppc-min-jump-table-entries=4 -o - \
; RUN:      -ppc-use-absolute-jumptables=false --relocation-model=pic < %s | FileCheck %s \
; RUN:      -check-prefix=CHECK-R-PIC-AIX
; RUN:  llc -mtriple=powerpc64le-unknown-linux-gnu -ppc-min-jump-table-entries=4 -o - \
; RUN:      -ppc-use-absolute-jumptables=true --relocation-model=static < %s | FileCheck %s \
; RUN:      -check-prefix=CHECK-A-STATIC-LE
; RUN:  llc -mtriple=powerpc64le-unknown-linux-gnu -ppc-min-jump-table-entries=4 -o - \
; RUN:      -ppc-use-absolute-jumptables=false --relocation-model=static < %s | FileCheck %s \
; RUN:      -check-prefix=CHECK-R-STATIC-LE
; RUN:  llc -mtriple=powerpc64-unknown-linux-gnu -ppc-min-jump-table-entries=4 -o - \
; RUN:      -ppc-use-absolute-jumptables=true --relocation-model=static < %s | FileCheck %s \
; RUN:      -check-prefix=CHECK-A-STATIC-BE
; RUN:  llc -mtriple=powerpc64-unknown-linux-gnu -ppc-min-jump-table-entries=4 -o - \
; RUN:      -ppc-use-absolute-jumptables=false --relocation-model=static < %s | FileCheck %s \
; RUN:      -check-prefix=CHECK-R-STATIC-BE

%struct.node = type { i8, ptr }

; Function Attrs: norecurse nounwind readonly
define zeroext i32 @jumpTableTest(ptr readonly %list) {
; CHECK-LE-LABEL: jumpTableTest:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE:       rldic r[[REG:[0-9]+]], r[[REG]], 3, 29
; CHECK-LE:       ldx r[[REG]], r[[REG1:[0-9]+]], r[[REG]]
; CHECK-LE:       mtctr r[[REG]]
; CHECK-LE:       bctr
; CHECK-LE:       blr
;
; CHECK-BE-LABEL: jumpTableTest:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE:       rldic r[[REG:[0-9]+]], r[[REG]], 2, 30
; CHECK-BE:       lwax r[[REG]], r[[REG1:[0-9]+]], r[[REG]]
; CHECK-BE:       mtctr r[[REG]]
; CHECK-BE:       bctr
; CHECK-BE:       blr
;
; CHECK-AIX-LABEL: jumpTableTest:
; CHECK-AIX:       # %bb.0: # %entry
; CHECK-AIX:       rldic r[[REG:[0-9]+]], r[[REG]], 2, 30
; CHECK-AIX:       lwax r[[REG]], r[[REG1:[0-9]+]], r[[REG]]
; CHECK-AIX:       mtctr r[[REG]]
; CHECK-AIX:       bctr
; CHECK-AIX:       blr
;
; CHECK-A-PIC-LE-LABEL:       .LJTI0_0:
; CHECK-A-PIC-LE:             .long   .LBB0_6-.LJTI0_0
;
; CHECK-R-PIC-LE-LABEL:       .LJTI0_0:
; CHECK-R-PIC-LE:             .long   .LBB0_6-.LJTI0_0
;
; CHECK-A-PIC-BE-LABEL:       .LJTI0_0:
; CHECK-A-PIC-BE:             .long   .LBB0_9-.LJTI0_0
;
; CHECK-R-PIC-BE-LABEL:       .LJTI0_0:
; CHECK-R-PIC-BE:             .long   .LBB0_9-.LJTI0_0
;
; CHECK-A-PIC-AIX-LABEL:      L..JTI0_0:
; CHECK-A-PIC-AIX:            .vbyte  4, L..BB0_9-L..JTI0_0
;
; CHECK-R-PIC-AIX-LABEL:      L..JTI0_0:
; CHECK-R-PIC-AIX:            .vbyte  4, L..BB0_9-L..JTI0_0
;
; CHECK-A-STATIC-LE-LABEL:    .LJTI0_0:
; CHECK-A-STATIC-LE:          .quad   .LBB0_6
;
; CHECK-R-STATIC-LE-LABEL:    .LJTI0_0:
; CHECK-R-STATIC-LE:          .long   .LBB0_6-.LJTI0_0
;
; CHECK-A-STATIC-BE-LABEL:    .LJTI0_0:
; CHECK-A-STATIC-BE:          .quad   .LBB0_9
;
; CHECK-R-STATIC-BE-LABEL:    .LJTI0_0:
; CHECK-R-STATIC-BE:          .long   .LBB0_9-.LJTI0_0
entry:
  %cmp36 = icmp eq ptr %list, null
  br i1 %cmp36, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %sw.epilog
  %result.038 = phi i32 [ %result.1, %sw.epilog ], [ 0, %entry ]
  %current.037 = phi ptr [ %spec.store.select, %sw.epilog ], [ %list, %entry ]
  %next1 = getelementptr inbounds %struct.node, ptr %current.037, i64 0, i32 1
  %0 = load ptr, ptr %next1, align 8
  %cmp2 = icmp eq ptr %0, %current.037
  %spec.store.select = select i1 %cmp2, ptr null, ptr %0
  %1 = load i8, ptr %current.037, align 8
  switch i8 %1, label %sw.epilog [
    i8 1, label %sw.bb
    i8 2, label %sw.bb3
    i8 3, label %sw.bb5
    i8 4, label %sw.bb7
    i8 5, label %sw.bb9
    i8 6, label %sw.bb11
    i8 7, label %sw.bb13
    i8 8, label %sw.bb15
    i8 9, label %sw.bb17
  ]

sw.bb:                                            ; preds = %while.body
  %add = add nsw i32 %result.038, 13
  br label %sw.epilog

sw.bb3:                                           ; preds = %while.body
  %add4 = add nsw i32 %result.038, 5
  br label %sw.epilog

sw.bb5:                                           ; preds = %while.body
  %add6 = add nsw i32 %result.038, 2
  br label %sw.epilog

sw.bb7:                                           ; preds = %while.body
  %add8 = add nsw i32 %result.038, 7
  br label %sw.epilog

sw.bb9:                                           ; preds = %while.body
  %add10 = add nsw i32 %result.038, 11
  br label %sw.epilog

sw.bb11:                                          ; preds = %while.body
  %add12 = add nsw i32 %result.038, 17
  br label %sw.epilog

sw.bb13:                                          ; preds = %while.body
  %add14 = add nsw i32 %result.038, 16
  br label %sw.epilog

sw.bb15:                                          ; preds = %while.body
  %add16 = add nsw i32 %result.038, 81
  br label %sw.epilog

sw.bb17:                                          ; preds = %while.body
  %add18 = add nsw i32 %result.038, 72
  br label %sw.epilog

sw.epilog:                                        ; preds = %while.body, %sw.bb17, %sw.bb15, %sw.bb13, %sw.bb11, %sw.bb9, %sw.bb7, %sw.bb5, %sw.bb3, %sw.bb
  %result.1 = phi i32 [ %result.038, %while.body ], [ %add18, %sw.bb17 ], [ %add16, %sw.bb15 ], [ %add14, %sw.bb13 ], [ %add12, %sw.bb11 ], [ %add10, %sw.bb9 ], [ %add8, %sw.bb7 ], [ %add6, %sw.bb5 ], [ %add4, %sw.bb3 ], [ %add, %sw.bb ]
  %cmp = icmp eq ptr %spec.store.select, null
  br i1 %cmp, label %while.end, label %while.body

while.end:                                        ; preds = %sw.epilog, %entry
  %result.0.lcssa = phi i32 [ 0, %entry ], [ %result.1, %sw.epilog ]
  ret i32 %result.0.lcssa
}


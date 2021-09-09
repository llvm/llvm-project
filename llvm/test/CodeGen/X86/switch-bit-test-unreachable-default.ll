; RUN: llc -mtriple=x86_64-- -global-isel=0 -print-after=finalize-isel \
; RUN:   -stop-after=finalize-isel %s -o /dev/null 2>&1 | \
; RUN:   FileCheck %s --check-prefix=CHECK-SDISEL
; RUN: llc -mtriple=x86_64-- -global-isel=1 -print-after=finalize-isel \
; RUN:   -stop-after=finalize-isel %s -o /dev/null 2>&1 | \
; RUN:   FileCheck %s --check-prefix=CHECK-GISEL

; PR50080
define i32 @baz(i32 %0) {
; FIXME: Get rid of this conditional jump and bit test in bb.5.
; FIXME: bb.2 should not have bb.5 as a predecessor.
; CHECK-SDISEL: bb.0 (%ir-block.1):
; CHECK-SDISEL:   successors: %bb.4(0x80000000); %bb.4(100.00%)
; CHECK-SDISEL:   liveins: $edi
; CHECK-SDISEL:   %1:gr32 = COPY $edi
; CHECK-SDISEL:   %2:gr32 = MOV32r0 implicit-def dead $eflags
; CHECK-SDISEL:   %3:gr32 = COPY %1:gr32
; CHECK-SDISEL: bb.4 (%ir-block.1):
; CHECK-SDISEL: ; predecessors: %bb.0
; CHECK-SDISEL:   successors: %bb.3(0x55555555), %bb.5(0x2aaaaaab); %bb.3(66.67%), %bb.5(33.33%)
; CHECK-SDISEL:   %4:gr32 = MOV32ri 13056
; CHECK-SDISEL:   BT32rr killed %4:gr32, %3:gr32, implicit-def $eflags
; CHECK-SDISEL:   JCC_1 %bb.3, 2, implicit $eflags
; CHECK-SDISEL: bb.5 (%ir-block.1):
; CHECK-SDISEL: ; predecessors: %bb.4
; CHECK-SDISEL:   successors: %bb.1(0x80000000), %bb.2(0x00000000); %bb.1(100.00%), %bb.2(0.00%)
; CHECK-SDISEL:   %5:gr32 = MOV32ri 48
; CHECK-SDISEL:   BT32rr killed %5:gr32, %3:gr32, implicit-def $eflags
; CHECK-SDISEL:   JCC_1 %bb.1, 2, implicit $eflags
; CHECK-SDISEL:   JMP_1 %bb.2
; CHECK-SDISEL: bb.1.sw.epilog8:
; CHECK-SDISEL: ; predecessors: %bb.5
; CHECK-SDISEL:   successors: %bb.3(0x80000000); %bb.3(100.00%)
; CHECK-SDISEL:   %6:gr32 = MOV32ri 1
; CHECK-SDISEL:   JMP_1 %bb.3
; CHECK-SDISEL: bb.2.if.then.unreachabledefault:
; CHECK-SDISEL: ; predecessors: %bb.5
; CHECK-SDISEL: bb.3.return:
; CHECK-SDISEL: ; predecessors: %bb.4, %bb.1
; CHECK-SDISEL:   %0:gr32 = PHI %2:gr32, %bb.4, %6:gr32, %bb.1
; CHECK-SDISEL:   $eax = COPY %0:gr32
; CHECK-SDISEL:   RET 0, $eax


; FIXME: Get rid of this conditional jump and bit test in bb.6.
; FIXME: bb.3 should not have bb.6 as a predecessor.
; CHECK-GISEL: bb.1 (%ir-block.1):
; CHECK-GISEL:   successors: %bb.5(0x80000000); %bb.5(100.00%)
; CHECK-GISEL:   liveins: $edi
; CHECK-GISEL:   %0:gr32 = COPY $edi
; CHECK-GISEL:   %16:gr32 = MOV32ri 1
; CHECK-GISEL:   %17:gr32 = MOV32r0 implicit-def $eflags
; CHECK-GISEL:   %2:gr32 = SUB32ri8 %0:gr32(tied-def 0), 0, implicit-def $eflags
; CHECK-GISEL: bb.5 (%ir-block.1):
; CHECK-GISEL: ; predecessors: %bb.1
; CHECK-GISEL:   successors: %bb.4(0x55555555), %bb.6(0x2aaaaaab); %bb.4(66.67%), %bb.6(33.33%)
; CHECK-GISEL:   %3:gr32 = MOV32ri 1
; CHECK-GISEL:   %21:gr8 = COPY %2.sub_8bit:gr32
; CHECK-GISEL:   $cl = COPY %21:gr8
; CHECK-GISEL:   %4:gr32 = SHL32rCL %3:gr32(tied-def 0), implicit-def $eflags, implicit $cl
; CHECK-GISEL:   %6:gr32 = AND32ri %4:gr32(tied-def 0), 13056, implicit-def $eflags
; CHECK-GISEL:   %7:gr32 = MOV32r0 implicit-def $eflags
; CHECK-GISEL:   CMP32rr %6:gr32, %7:gr32, implicit-def $eflags
; CHECK-GISEL:   %20:gr8 = SETCCr 5, implicit $eflags
; CHECK-GISEL:   TEST8ri %20:gr8, 1, implicit-def $eflags
; CHECK-GISEL:   JCC_1 %bb.4, 5, implicit $eflags
; CHECK-GISEL: bb.6 (%ir-block.1):
; CHECK-GISEL: ; predecessors: %bb.5
; CHECK-GISEL:   successors: %bb.2(0x80000000), %bb.3(0x00000000); %bb.2(100.00%), %bb.3(0.00%)
; CHECK-GISEL:   %9:gr32 = MOV32ri 1
; CHECK-GISEL:   %19:gr8 = COPY %2.sub_8bit:gr32
; CHECK-GISEL:   $cl = COPY %19:gr8
; CHECK-GISEL:   %10:gr32 = SHL32rCL %9:gr32(tied-def 0), implicit-def $eflags, implicit $cl
; CHECK-GISEL:   %12:gr32 = AND32ri8 %10:gr32(tied-def 0), 48, implicit-def $eflags
; CHECK-GISEL:   %13:gr32 = MOV32r0 implicit-def $eflags
; CHECK-GISEL:   CMP32rr %12:gr32, %13:gr32, implicit-def $eflags
; CHECK-GISEL:   %18:gr8 = SETCCr 5, implicit $eflags
; CHECK-GISEL:   TEST8ri %18:gr8, 1, implicit-def $eflags
; CHECK-GISEL:   JCC_1 %bb.2, 5, implicit $eflags
; CHECK-GISEL:   JMP_1 %bb.3
; CHECK-GISEL: bb.2.sw.epilog8:
; CHECK-GISEL: ; predecessors: %bb.6
; CHECK-GISEL:   successors: %bb.4(0x80000000); %bb.4(100.00%)
; CHECK-GISEL:   JMP_1 %bb.4
; CHECK-GISEL: bb.3.if.then.unreachabledefault:
; CHECK-GISEL: ; predecessors: %bb.6
; CHECK-GISEL: bb.4.return:
; CHECK-GISEL: ; predecessors: %bb.5, %bb.2
; CHECK-GISEL:   %15:gr32 = PHI %16:gr32, %bb.2, %17:gr32, %bb.5
; CHECK-GISEL:   $eax = COPY %15:gr32
; CHECK-GISEL:   RET 0, implicit $eax

  switch i32 %0, label %if.then.unreachabledefault [
    i32 4, label %sw.epilog8
    i32 5, label %sw.epilog8
    i32 8, label %sw.bb2
    i32 9, label %sw.bb2
    i32 12, label %sw.bb4
    i32 13, label %sw.bb4
  ]

sw.bb2:
  br label %return

sw.bb4:
  br label %return

sw.epilog8:
  br label %return

if.then.unreachabledefault:
  unreachable

return:
  %retval.0 = phi i32 [ 1, %sw.epilog8 ], [ 0, %sw.bb2 ], [ 0, %sw.bb4 ]
  ret i32 %retval.0
}

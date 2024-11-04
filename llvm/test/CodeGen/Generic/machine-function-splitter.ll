; REQUIRES: aarch64-registered-target
; REQUIRES: x86-registered-target

; COM: Machine function splitting with FDO profiles
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -split-machine-functions | FileCheck %s -check-prefixes=MFS-DEFAULTS,MFS-DEFAULTS-X86
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -split-machine-functions -mfs-psi-cutoff=0 -mfs-count-threshold=2000 | FileCheck %s --dump-input=always -check-prefixes=MFS-OPTS1,MFS-OPTS1-X86
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -split-machine-functions -mfs-psi-cutoff=950000 | FileCheck %s -check-prefixes=MFS-OPTS2,MFS-OPTS2-X86
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -split-machine-functions -mfs-split-ehcode | FileCheck %s -check-prefixes=MFS-EH-SPLIT,MFS-EH-SPLIT-X86
; RUN: llc < %s -mtriple=x86_64 -split-machine-functions -O0 -mfs-psi-cutoff=0 -mfs-count-threshold=10000 | FileCheck %s -check-prefixes=MFS-O0,MFS-O0-X86

; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -aarch64-min-jump-table-entries=4 -enable-split-machine-functions | FileCheck %s -check-prefixes=MFS-DEFAULTS,MFS-DEFAULTS-AARCH64
; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -aarch64-min-jump-table-entries=4 -enable-split-machine-functions -mfs-psi-cutoff=0 -mfs-count-threshold=2000 | FileCheck %s --dump-input=always -check-prefixes=MFS-OPTS1,MFS-OPTS1-AARCH64
; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -aarch64-min-jump-table-entries=4 -enable-split-machine-functions -mfs-psi-cutoff=950000 | FileCheck %s -check-prefixes=MFS-OPTS2,MFS-OPTS2-AARCH64
; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -aarch64-min-jump-table-entries=4 -enable-split-machine-functions -mfs-split-ehcode | FileCheck %s -check-prefixes=MFS-EH-SPLIT,MFS-EH-SPLIT-AARCH64
; RUN: llc < %s -mtriple=aarch64 -enable-split-machine-functions -aarch64-redzone | FileCheck %s -check-prefixes=MFS-REDZONE-AARCH64

; COM: Machine function splitting with AFDO profiles
; RUN: sed 's/InstrProf/SampleProfile/g' %s > %t.ll
; RUN: llc < %t.ll -mtriple=x86_64-unknown-linux-gnu -split-machine-functions | FileCheck %s --check-prefix=FSAFDO-MFS
; RUN: llc < %t.ll -mtriple=x86_64-unknown-linux-gnu -split-machine-functions | FileCheck %s --check-prefix=FSAFDO-MFS2

define void @foo1(i1 zeroext %0) nounwind !prof !14 !section_prefix !15 {
;; Check that cold block is moved to .text.split.
; MFS-DEFAULTS-LABEL:         foo1
; MFS-DEFAULTS:               .section        .text.split.foo1
; MFS-DEFAULTS-NEXT:          foo1.cold:
; MFS-DEFAULTS-X86-NOT:       callq   bar
; MFS-DEFAULTS-X86-NEXT:      callq   baz
; MFS-DEFAULTS-AARCH64-NOT:   bl      bar
; MFS-DEFAULTS-AARCH64-NEXT:  bl      baz
  br i1 %0, label %2, label %4, !prof !17

2:                                                ; preds = %1
  %3 = call i32 @bar()
  br label %6

4:                                                ; preds = %1
  %5 = call i32 @baz()
  br label %6

6:                                                ; preds = %4, %2
  %7 = tail call i32 @qux()
  ret void
}

define void @foo2(i1 zeroext %0) nounwind !prof !23 !section_prefix !16 {
;; Check that function marked unlikely is not split.
; MFS-DEFAULTS-LABEL: foo2
; MFS-DEFAULTS-NOT:   foo2.cold:
  br i1 %0, label %2, label %4, !prof !17

2:                                                ; preds = %1
  %3 = call i32 @bar()
  br label %6

4:                                                ; preds = %1
  %5 = call i32 @baz()
  br label %6

6:                                                ; preds = %4, %2
  %7 = tail call i32 @qux()
  ret void
}

define void @foo3(i1 zeroext %0) nounwind !section_prefix !15 {
;; Check that function without profile data is not split.
; MFS-DEFAULTS-LABEL: foo3
; MFS-DEFAULTS-NOT:   foo3.cold:
  br i1 %0, label %2, label %4

2:                                                ; preds = %1
  %3 = call i32 @bar()
  br label %6

4:                                                ; preds = %1
  %5 = call i32 @baz()
  br label %6

6:                                                ; preds = %4, %2
  %7 = tail call i32 @qux()
  ret void
}

define void @foo4(i1 zeroext %0, i1 zeroext %1) nounwind !prof !20 {
;; Check that count threshold works.
; MFS-OPTS1-LABEL:         foo4
; MFS-OPTS1:               .section        .text.split.foo4
; MFS-OPTS1-NEXT:          foo4.cold:
; MFS-OPTS1-X86-NOT:       callq    bar
; MFS-OPTS1-X86-NOT:       callq    baz
; MFS-OPTS1-X86-NEXT:      callq    bam
; MFS-OPTS1-AARCH64-NOT:   bl       bar
; MFS-OPTS1-AARCH64-NOT:   bl       baz
; MFS-OPTS1-AARCH64-NEXT:  bl       bam
  br i1 %0, label %3, label %7, !prof !18

3:
  %4 = call i32 @bar()
  br label %7

5:
  %6 = call i32 @baz()
  br label %7

7:
  br i1 %1, label %8, label %10, !prof !19

8:
  %9 = call i32 @bam()
  br label %12

10:
  %11 = call i32 @baz()
  br label %12

12:
  %13 = tail call i32 @qux()
  ret void
}

define void @foo5(i1 zeroext %0, i1 zeroext %1) nounwind !prof !20 {
;; Check that profile summary info cutoff works.
; MFS-OPTS2-LABEL:         foo5
; MFS-OPTS2:               .section        .text.split.foo5
; MFS-OPTS2-NEXT:               foo5.cold:
; MFS-OPTS2-X86-NOT:       callq    bar
; MFS-OPTS2-X86-NOT:       callq    baz
; MFS-OPTS2-X86-NEXT:      callq    bam
; MFS-OPTS2-AARCH64-NOT:   bl       bar
; MFS-OPTS2-AARCH64-NOT:   bl       baz
; MFS-OPTS2-AARCH64-NEXT:  bl       bam
  br i1 %0, label %3, label %7, !prof !21

3:
  %4 = call i32 @bar()
  br label %7

5:
  %6 = call i32 @baz()
  br label %7

7:
  br i1 %1, label %8, label %10, !prof !22

8:
  %9 = call i32 @bam()
  br label %12

10:
  %11 = call i32 @baz()
  br label %12

12:
  %13 = call i32 @qux()
  ret void
}

define void @foo6(i1 zeroext %0) nounwind section "nosplit" !prof !14 {
;; Check that function with section attribute is not split.
; MFS-DEFAULTS-LABEL: foo6
; MFS-DEFAULTS-NOT:   foo6.cold:
  br i1 %0, label %2, label %4, !prof !17

2:                                                ; preds = %1
  %3 = call i32 @bar()
  br label %6

4:                                                ; preds = %1
  %5 = call i32 @baz()
  br label %6

6:                                                ; preds = %4, %2
  %7 = tail call i32 @qux()
  ret void
}

define i32 @foo7(i1 zeroext %0) personality ptr @__gxx_personality_v0 !prof !14 {
;; Check that a single cold ehpad is split out.
; MFS-DEFAULTS-LABEL:         foo7
; MFS-DEFAULTS:               .section        .text.split.foo7,"ax",@progbits
; MFS-DEFAULTS-NEXT:          foo7.cold:
; MFS-DEFAULTS-X86:           callq   baz
; MFS-DEFAULTS-X86:           callq   _Unwind_Resume@PLT
; MFS-DEFAULTS-AARCH64:       bl      baz
entry:
  invoke void @_Z1fv()
          to label %try.cont unwind label %lpad

lpad:
  %1 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIi
  resume { ptr, i32 } %1

try.cont:
  br i1 %0, label %2, label %4, !prof !17

2:                                                ; preds = try.cont
  %3 = call i32 @bar()
  br label %6

4:                                                ; preds = %1
  %5 = call i32 @baz()
  br label %6

6:                                                ; preds = %4, %2
  %7 = tail call i32 @qux()
  ret i32 %7
}

define i32 @foo8(i1 zeroext %0) personality ptr @__gxx_personality_v0 !prof !14 {
;; Check that all ehpads are treated as hot if one of them is hot.
; MFS-DEFAULTS-LABEL:         foo8
; MFS-DEFAULTS-X86:           callq   _Unwind_Resume@PLT
; MFS-DEFAULTS-X86:           callq   _Unwind_Resume@PLT
; MFS-DEFAULTS:               .section        .text.split.foo8,"ax",@progbits
; MFS-DEFAULTS-NEXT:          foo8.cold:
; MFS-DEFAULTS-X86:           callq   baz
; MFS-DEFAULTS-AARCH64:       bl      baz

;; Check that all ehpads are by default treated as cold with -mfs-split-ehcode.
; MFS-EH-SPLIT-LABEL:         foo8
; MFS-EH-SPLIT-X86:           callq   baz
; MFS-EH-SPLIT-AARCH64:       bl      baz
; MFS-EH-SPLIT-X86:           .section        .text.split.foo8,"ax",@progbits
; MFS-EH-SPLIT-X86-NEXT:      foo8.cold:
; MFS-EH-SPLIT-X86:           callq   _Unwind_Resume@PLT
; MFS-EH-SPLIT-X86:           callq   _Unwind_Resume@PLT
entry:
  invoke void @_Z1fv()
          to label %try.cont unwind label %lpad1

lpad1:
  %1 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIi
  resume { ptr, i32 } %1

try.cont:
  br i1 %0, label %hot, label %cold, !prof !17

hot:
  %2 = call i32 @bar()
  invoke void @_Z1fv()
          to label %exit unwind label %lpad2, !prof !21

lpad2:
  %3 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIi
  resume { ptr, i32 } %3

cold:
  %4 = call i32 @baz()
  br label %exit

exit:
  %5 = tail call i32 @qux()
  ret i32 %5
}

define void @foo9(i1 zeroext %0) nounwind #0 !prof !14 {
;; Check that function with section attribute is not split.
; MFS-DEFAULTS-LABEL: foo9
; MFS-DEFAULTS-NOT:   foo9.cold:
  br i1 %0, label %2, label %4, !prof !17

2:                                                ; preds = %1
  %3 = call i32 @bar()
  br label %6

4:                                                ; preds = %1
  %5 = call i32 @baz()
  br label %6

6:                                                ; preds = %4, %2
  %7 = tail call i32 @qux()
  ret void
}

define i32 @foo10(i1 zeroext %0) personality ptr @__gxx_personality_v0 !prof !14 {
;; Check that nop is inserted just before the EH pad if it's beginning a section.
; MFS-DEFAULTS-LABEL:         foo10
; MFS-DEFAULTS-X86-LABEL:     callq   baz
; MFS-DEFAULTS-AARCH64:       bl      baz
; MFS-DEFAULTS-X86:           .section        .text.split.foo10,"ax",@progbits
; MFS-DEFAULTS-X86-NEXT:      foo10.cold:
; MFS-DEFAULTS-X86:           nop
; MFS-DEFAULTS-X86:           callq   _Unwind_Resume@PLT
entry:
  invoke void @_Z1fv()
          to label %try.cont unwind label %lpad, !prof !17

lpad:
  %1 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIi
  resume { ptr, i32 } %1

try.cont:
  %2 = call i32 @baz()
  ret i32 %2
}

define void @foo11(i1 zeroext %0) personality ptr @__gxx_personality_v0 {
;; Check that function having landing pads are split with mfs-split-ehcode
;; even in the absence of profile data
; MFS-EH-SPLIT-LABEL:         foo11
; MFS-EH-SPLIT-X86:           .section        .text.split.foo11,"ax",@progbits
; MFS-EH-SPLIT-X86-NEXT:      foo11.cold:
; MFS-EH-SPLIT-X86:           nop
; MFS-EH-SPLIT-X86:           callq   _Unwind_Resume@PLT
entry:
  invoke void @_Z1fv()
        to label %2 unwind label %lpad

lpad:
  %1 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIi
  resume { ptr, i32 } %1

2:                                                ; preds = entry
  %3 = tail call i32 @qux()
  ret void
}

define i32 @foo12(i1 zeroext %0) personality ptr @__gxx_personality_v0 !prof !14 {
;; Check that all code reachable from ehpad is split out with cycles.
; MFS-EH-SPLIT-LABEL:         foo12
; MFS-EH-SPLIT:               .section        .text.split.foo12,"ax",@progbits
; MFS-EH-SPLIT-NEXT:          foo12.cold:
; MFS-EH-SPLIT-X86:           callq   bar
; MFS-EH-SPLIT-X86:           callq   baz
; MFS-EH-SPLIT-X86:           callq   qux
; MFS-EH-SPLIT-AARCH64:       bl      bar
; MFS-EH-SPLIT-AARCH64:       bl      baz
; MFS-EH-SPLIT-AARCH64:       bl      qux
entry:
  invoke void @_Z1fv()
          to label %8 unwind label %lpad

lpad:
  %1 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIi
  br label %2

2:                                                ; preds = lpad
  %3 = call i32 @bar()
  br i1 %0, label %4, label %6

4:                                                ; preds = lpad
  %5 = call i32 @baz()
  br label %6

6:                                                ; preds = %4, %2
  %7 = tail call i32 @qux()
  br i1 %0, label %2, label %8

8:                                                ; preds = %6
  ret i32 0
}

define i32 @foo13(i1 zeroext %0) personality ptr @__gxx_personality_v0 !prof !14{
;; Check that all code reachable from EH
;; that is also reachable from outside EH pad
;; is not touched.
; MFS-EH-SPLIT-LABEL:         foo13
; MFS-EH-SPLIT-X86:           callq   bam
; MFS-EH-SPLIT-AARCH64:       bl      bam
; MFS-EH-SPLIT:               .section        .text.split.foo13,"ax",@progbits
; MFS-EH-SPLIT-NEXT:          foo13.cold:
; MFS-EH-SPLIT-X86:           callq   baz
; MFS-EH-SPLIT-X86:           callq   bar
; MFS-EH-SPLIT-X86:           callq   qux
; MFS-EH-SPLIT-AARCH64:       bl      baz
; MFS-EH-SPLIT-AARCH64:       bl      bar
; MFS-EH-SPLIT-AARCH64:       bl      qux
entry:
  invoke void @_Z1fv()
          to label %try.cont unwind label %lpad, !prof !17

lpad:
  %1 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIi
  br i1 %0, label %2, label %4, !prof !17

2:                                                ; preds = lpad
  %3 = call i32 @bar()
  br label %6

4:                                                ; preds = lpad
  %5 = call i32 @baz()
  br label %6

6:                                                ; preds = %4, %2
  %7 = tail call i32 @qux()
  br i1 %0, label %2, label %try.cont, !prof !17

try.cont:                                        ; preds = %entry
  %8 = call i32 @bam()
  ret i32 %8
}

define void @foo14(i1 zeroext %0, i1 zeroext %1) nounwind !prof !24 {
; FSAFDO-MFS: .section	.text.split.foo14,"ax"
; FSAFDO-MFS: foo14.cold:
  br i1 %0, label %3, label %7, !prof !25

3:
  %4 = call i32 @bar()
  br label %7

5:
  %6 = call i32 @baz()
  br label %7

7:
  br i1 %1, label %8, label %10, !prof !26

8:
  %9 = call i32 @bam()
  br label %12

10:
  %11 = call i32 @baz()
  br label %12

12:
  %13 = tail call i32 @qux()
  ret void
}

define void @foo15(i1 zeroext %0, i1 zeroext %1) nounwind !prof !27 {
;; HasAccurateProfile is false, foo15 is hot, but no profile data for
;; blocks, no split should happen.
; FSAFDO-MFS2-NOT: .section	.text.split.foo15,"ax"
; FSAFDO-MFS2-NOT: foo15.cold:
  br i1 %0, label %3, label %7

3:
  %4 = call i32 @bar()
  br label %7

5:
  %6 = call i32 @baz()
  br label %7

7:
  br i1 %1, label %8, label %10

8:
  %9 = call i32 @bam()
  br label %12

10:
  %11 = call i32 @baz()
  br label %12

12:
  %13 = tail call i32 @qux()
  ret void
}

define void @foo16(i1 zeroext %0) nounwind !prof !14 !section_prefix !15 {
;; Check that an unconditional branch is only appended to a block
;; if it would fall through to the wrong block otherwise.
; MFS-O0-LABEL:               foo16
; MFS-O0-X86:                 jmp
; MFS-O0-X86-NOT:             jmp
; MFS-O0-AARCH64:                 br
; MFS-O0-AARCH64:                 br
; MFS-O0-AARCH64-NOT:             br
; MFS-O0:                     .section        .text.split.foo16
; MFS-O0-NEXT:                foo16.cold
  %2 = call i32 @baz()
  br i1 false, label %3, label %5, !prof !25

3:                                                ; preds = %1
  %4 = call i32 @bar()
  unreachable

5:                                                ; preds = %1
  %6 = tail call i32 @qux()
  ret void
}

define i32 @foo17(i1 zeroext %0, i32 %a, i32 %b) nounwind !prof !14 !section_prefix !15 {
;; Check that cold blocks in functions with red zones aren't split.
; MFS-DEFAULTS-LABEL:        foo17
; MFS-DEFAULTS-X86:          foo17.cold:
; MFS-REDZONE-AARCH64-NOT:   foo17.cold:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %x = alloca i32, align 4

  br i1 %0, label %2, label %3, !prof !17

2:                                                ; preds = %1
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  br label %4

3:                                                ; preds = %1
  store i32 %a, ptr %b.addr, align 4
  store i32 %b, ptr %a.addr, align 4
  br label %4

4:                                                ; preds = %3, %2
  %tmp = load i32, ptr %a.addr, align 4
  %tmp1 = load i32, ptr %b.addr, align 4
  %add = add nsw i32 %tmp, %tmp1
  store i32 %add, ptr %x, align 4
  %tmp2 = load i32, ptr %x, align 4
  ret i32 %tmp2
}

define i32 @foo18(i32 %in) !prof !14 !section_prefix !15 {
;; Check that a cold block targeted by a jump table is not split
;; on AArch64.
; MFS-DEFAULTS-LABEL:        foo18
; MFS-DEFAULTS:              .section        .text.split.foo18
; MFS-DEFAULTS-NEXT:         foo18.cold:
; MFS-DEFAULTS-SAME:           %common.ret
; MFS-DEFAULTS-X86-DAG:        jmp     qux
; MFS-DEFAULTS-X86-DAG:        jmp     bam
; MFS-DEFAULTS-AARCH64-NOT:    b       bar
; MFS-DEFAULTS-AARCH64-NOT:    b       baz
; MFS-DEFAULTS-AARCH64-NOT:    b       qux
; MFS-DEFAULTS-AARCH64-NOT:    b       bam

  switch i32 %in, label %common.ret [
    i32 0, label %hot1
    i32 1, label %hot2
    i32 2, label %cold1
    i32 3, label %cold2
  ], !prof !28

common.ret:                                       ; preds = %0
  ret i32 0

hot1:                                             ; preds = %0
  %1 = tail call i32 @bar()
  ret i32 %1

hot2:                                             ; preds = %0
  %2 = tail call i32 @baz()
  ret i32 %2

cold1:                                            ; preds = %0
  %3 = tail call i32 @bam()
  ret i32 %3

cold2:                                            ; preds = %0
  %4 = tail call i32 @qux()
  ret i32 %4
}

define i32 @foo19(i32 %in) !prof !14 !section_prefix !15 {
;; Check that a cold block that contains a jump table dispatch is
;; not split on AArch64.
; MFS-DEFAULTS-LABEL:        foo19
; MFS-DEFAULTS:              .section        .text.split.foo19
; MFS-DEFAULTS-NEXT:         foo19.cold:
; MFS-DEFAULTS-X86:            .LJTI18_0
; MFS-DEFAULTS-AARCH64-NOT:    .LJTI18_0
; MFS-DEFAULTS:              .section        .rodata
; MFS-DEFAULTS:                .LJTI18_0
  %cmp = icmp sgt i32 %in, 3
  br i1 %cmp, label %hot, label %cold_switch, !prof !17

hot:                                              ; preds = %0
ret i32 1

cold_switch:                                      ; preds = %0
  switch i32 %in, label %common.ret [
    i32 0, label %hot1
    i32 1, label %hot2
    i32 2, label %cold1
    i32 3, label %cold2
  ], !prof !28

common.ret:                                       ; preds = %0
  ret i32 0

hot1:                                             ; preds = %0
  %1 = tail call i32 @bar()
  ret i32 %1

hot2:                                             ; preds = %0
  %2 = tail call i32 @baz()
  ret i32 %2

cold1:                                            ; preds = %0
  %3 = tail call i32 @bam()
  ret i32 %3

cold2:                                            ; preds = %0
  %4 = tail call i32 @qux()
  ret i32 %4
}

define void @foo20(i1 zeroext %0) !prof !14 !section_prefix !15 {
;; Check that blocks containing or targeted by asm goto aren't split.
; MFS-DEFAULTS-LABEL:        foo20
; MFS-DEFAULTS-AARCH64-NOT:  foo20.cold:
; MFS-DEFAULTS-X86:          .section        .text.split.foo20
; MFS-DEFAULTS-X86:          foo20.cold:
; MFS-DEFAULTS-X86-DAG:      # %cold_asm
; MFS-DEFAULTS-X86-DAG:      # %cold_asm_target

  br i1 %0, label %hot, label %cold_asm, !prof !17

hot:
  %2 = call i32 @bar()
  ret void

cold_asm:
    callbr void asm sideeffect "nop", "!i"() #3
          to label %asm.fallthrough [label %cold_asm_target]

asm.fallthrough:
    br label %cold_asm_target

cold_asm_target:
  %3 = call i32 @baz()
  ret void
}

declare i32 @bar()
declare i32 @baz()
declare i32 @bam()
declare i32 @qux()
declare void @_Z1fv()
declare i32 @__gxx_personality_v0(...)

@_ZTIi = external constant ptr

attributes #0 = { "implicit-section-name"="nosplit" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 5}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999900, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 7000}
!15 = !{!"function_section_prefix", !"hot"}
!16 = !{!"function_section_prefix", !"unlikely"}
!17 = !{!"branch_weights", i32 7000, i32 0}
!18 = !{!"branch_weights", i32 3000, i32 4000}
!19 = !{!"branch_weights", i32 1000, i32 6000}
!20 = !{!"function_entry_count", i64 10000}
!21 = !{!"branch_weights", i32 6000, i32 4000}
!22 = !{!"branch_weights", i32 80, i32 9920}
!23 = !{!"function_entry_count", i64 7}
!24 = !{!"function_entry_count", i64 10000}
!25 = !{!"branch_weights", i32 0, i32 7000}
!26 = !{!"branch_weights", i32 1000, i32 6000}
!27 = !{!"function_entry_count", i64 10000}
!28 = !{!"branch_weights", i32 0, i32 4000, i32 4000, i32 0, i32 0}

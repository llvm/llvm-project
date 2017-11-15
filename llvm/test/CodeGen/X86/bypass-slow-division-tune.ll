; Check that a division is bypassed when appropriate only.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mcpu=atom       < %s | FileCheck -check-prefixes=ATOM,CHECK %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mcpu=silvermont < %s | FileCheck -check-prefixes=REST,CHECK %s
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake    < %s | FileCheck -check-prefixes=REST,CHECK %s
; RUN: llc -profile-summary-huge-working-set-size-threshold=1 -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake    < %s | FileCheck -check-prefixes=HUGEWS %s

; Verify that div32 is bypassed only for Atoms.
define i32 @div32(i32 %a, i32 %b) {
entry:
; ATOM-LABEL: div32:
; ATOM: orl   %{{.*}}, [[REG:%[a-z]+]]
; ATOM: testl $-256, [[REG]]
; ATOM: divb
;
; REST-LABEL: div32:
; REST-NOT: divb
;
  %div = sdiv i32 %a, %b
  ret i32 %div
}

; Verify that div64 is always bypassed.
define i64 @div64(i64 %a, i64 %b) {
entry:
; CHECK-LABEL: div64:
; CHECK:     orq     %{{.*}}, [[REG:%[a-z]+]]
; CHECK:     shrq    $32, [[REG]]
; CHECK:     divl
;
  %div = sdiv i64 %a, %b
  ret i64 %div
}


; Verify that no extra code is generated when optimizing for size.

define i64 @div64_optsize(i64 %a, i64 %b) optsize {
; CHECK-LABEL: div64_optsize:
; CHECK-NOT: divl
; CHECK: ret
  %div = sdiv i64 %a, %b
  ret i64 %div
}

define i64 @div64_hugews(i64 %a, i64 %b) {
; HUGEWS-LABEL: div64_hugews:
; HUGEWS-NOT: divl
; HUGEWS: ret
  %div = sdiv i64 %a, %b
  ret i64 %div
}

define i32 @div32_optsize(i32 %a, i32 %b) optsize {
; CHECK-LABEL: div32_optsize:
; CHECK-NOT: divb
; CHECK: ret
  %div = sdiv i32 %a, %b
  ret i32 %div
}

define i32 @div32_minsize(i32 %a, i32 %b) minsize {
; CHECK-LABEL: div32_minsize:
; CHECK-NOT: divb
; CHECK: ret
  %div = sdiv i32 %a, %b
  ret i32 %div
}

!llvm.module.flags = !{!1}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 1000}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1000}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 1000, i32 1}
!13 = !{i32 999000, i64 1000, i32 3}
!14 = !{i32 999999, i64 5, i32 3}

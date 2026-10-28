; RUN: opt < %s -passes=constmerge -S | FileCheck %s

; Constants with an explicit section are only merged with other constants
; if their output section is same.

; Same output section, different input sections: merged.
; CHECK-NOT: @T1A =
; CHECK: @T1B = internal unnamed_addr constant i32 1, section ".rodata.T1B" #0
@T1A = internal unnamed_addr constant i32 1, section ".rodata.T1A" #0
@T1B = internal unnamed_addr constant i32 1, section ".rodata.T1B" #0

; Different output sections: not merged.
; CHECK: @T2A = internal unnamed_addr constant i32 2, section ".rodata.T2A" #0
; CHECK: @T2B = internal unnamed_addr constant i32 2, section ".rodata.T2B" #1
@T2A = internal unnamed_addr constant i32 2, section ".rodata.T2A" #0
@T2B = internal unnamed_addr constant i32 2, section ".rodata.T2B" #1

; Explicit section without a known output section: not merged.
; CHECK: @T3A = internal unnamed_addr constant i32 3, section ".rodata.T3A"{{$}}
; CHECK: @T3B = internal unnamed_addr constant i32 3, section ".rodata.T3B"{{$}}
@T3A = internal unnamed_addr constant i32 3, section ".rodata.T3A"
@T3B = internal unnamed_addr constant i32 3, section ".rodata.T3B"

; Known output section vs. no section at all: not merged.
; CHECK: @T4A = internal unnamed_addr constant i32 4, section ".rodata.T4A" #0
; CHECK: @T4B = internal unnamed_addr constant i32 4{{$}}
@T4A = internal unnamed_addr constant i32 4, section ".rodata.T4A" #0
@T4B = internal unnamed_addr constant i32 4

; Same output section, but neither is unnamed_addr: not merged.
; CHECK: @T5A = internal constant i32 5, section ".rodata.T5A" #0
; CHECK: @T5B = internal constant i32 5, section ".rodata.T5B" #0
@T5A = internal constant i32 5, section ".rodata.T5A" #0
@T5B = internal constant i32 5, section ".rodata.T5B" #0

; Merging @T6S1 into @T6S2 makes @T6P1 and @T6P2 identical, so they are merged
; in the next iteration.
; CHECK-NOT: @T6S1 =
; CHECK: @T6S2 = internal unnamed_addr constant i32 6, section ".rodata.T6S2" #0
; CHECK-NOT: @T6P1 =
; CHECK: @T6P2 = internal unnamed_addr constant ptr @T6S2, {{.*}} #0
@T6S1 = internal unnamed_addr constant i32 6, section ".rodata.T6S1" #0
@T6S2 = internal unnamed_addr constant i32 6, section ".rodata.T6S2" #0
@T6P1 = internal unnamed_addr constant ptr @T6S1, section ".rodata.T6P1" #0
@T6P2 = internal unnamed_addr constant ptr @T6S2, section ".rodata.T6P2" #0

; Checking that users of the erased globals are now pointing to the canonical
; globals
; CHECK-LABEL: define void @use(
; CHECK: store ptr @T1B, ptr %P
; CHECK: store ptr @T1B, ptr %P
; CHECK: store ptr @T2A, ptr %P
; CHECK: store ptr @T2B, ptr %P
; CHECK: store ptr @T3A, ptr %P
; CHECK: store ptr @T3B, ptr %P
; CHECK: store ptr @T4A, ptr %P
; CHECK: store ptr @T4B, ptr %P
; CHECK: store ptr @T5A, ptr %P
; CHECK: store ptr @T5B, ptr %P
; CHECK: store ptr @T6P2, ptr %P
; CHECK: store ptr @T6P2, ptr %P
define void @use(ptr %P) {
  store ptr @T1A, ptr %P
  store ptr @T1B, ptr %P
  store ptr @T2A, ptr %P
  store ptr @T2B, ptr %P
  store ptr @T3A, ptr %P
  store ptr @T3B, ptr %P
  store ptr @T4A, ptr %P
  store ptr @T4B, ptr %P
  store ptr @T5A, ptr %P
  store ptr @T5B, ptr %P
  store ptr @T6P1, ptr %P
  store ptr @T6P2, ptr %P
  ret void
}

; CHECK: attributes #0 = { "linker_output_section"=".out_a" }
; CHECK: attributes #1 = { "linker_output_section"=".out_b" }
attributes #0 = { "linker_output_section"=".out_a" }
attributes #1 = { "linker_output_section"=".out_b" }

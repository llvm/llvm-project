; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=large -o %t
; RUN: llvm-readelf -S %t | FileCheck %s

; Verify that anonymous constant pool entries get SHF_X86_64_LARGE
; and are placed in .lrodata.cst* sections under the large code model.

; CHECK: .lrodata.cst16 {{.*}} AMl
; CHECK: .lrodata.cst4  {{.*}} AMl

; Also verify the suffixed path (via -partition-static-data-sections).
; The .hot suffix requires profile information (see !prof metadata below)
; so that the partitioner can distinguish hot from cold constant pool entries.
; RUN: llc < %s -relocation-model=pic -code-model=large \
; RUN:     -partition-static-data-sections -o - | FileCheck %s --check-prefix=SUFFIX

; SUFFIX: .section .lrodata.cst16.hot.,"aMl",@progbits,16
; SUFFIX: .section .lrodata.cst4,"aMl",@progbits,4

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux"

define void @vec_add(ptr %out, ptr %a, ptr %b, i32 %n) !prof !17 {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %idx = sext i32 %iv to i64
  %pa = getelementptr float, ptr %a, i64 %idx
  %pb = getelementptr float, ptr %b, i64 %idx
  %po = getelementptr float, ptr %out, i64 %idx
  %va = load <4 x float>, ptr %pa, align 4
  %vb = load <4 x float>, ptr %pb, align 4
  %add = fadd <4 x float> %va, %vb
  ; Adding a vector splat of 1.0 forces a constant pool entry in .rodata.cst16
  %ones = fadd <4 x float> %add, <float 1.0, float 1.0, float 1.0, float 1.0>
  store <4 x float> %ones, ptr %po, align 4
  %iv.next = add i32 %iv, 4
  %done = icmp sge i32 %iv.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; A scalar float constant forces a .rodata.cst4 entry.
define float @scalar_const(float %x) {
  %r = fadd float %x, 1.0
  ret float %r
}

!llvm.module.flags = !{!1}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10, !11, !12}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 1460617}
!5 = !{!"MaxCount", i64 849536}
!6 = !{!"MaxInternalCount", i64 32769}
!7 = !{!"MaxFunctionCount", i64 849536}
!8 = !{!"NumCounts", i64 23784}
!9 = !{!"NumFunctions", i64 3301}
!10 = !{!"IsPartialProfile", i64 0}
!11 = !{!"PartialProfileRatio", double 0.000000e+00}
!12 = !{!"DetailedSummary", !13}
!13 = !{!14, !15}
!14 = !{i32 990000, i64 166, i32 73}
!15 = !{i32 999999, i64 1, i32 1463}
!16 = !{!"function_entry_count", i64 1}
!17 = !{!"function_entry_count", i64 100000}

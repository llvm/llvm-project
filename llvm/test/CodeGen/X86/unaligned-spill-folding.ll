; RUN: split-file %s %t
; RUN: cat %t/main.ll %t/align4.ll > %t/a2.ll
; RUN: cat %t/main.ll %t/align16.ll > %t/b2.ll
; RUN: llc -mtriple=i386-unknown-freebsd -mcpu=core2 -relocation-model=pic < %t/a2.ll | FileCheck %s -check-prefix=UNALIGNED
; RUN: llc -mtriple=i386-unknown-freebsd -mcpu=core2 -relocation-model=pic < %t/b2.ll | FileCheck %s -check-prefix=ALIGNED
; RUN: llc -mtriple=i386-unknown-freebsd -mcpu=core2 -stackrealign -relocation-model=pic < %t/a2.ll | FileCheck %s -check-prefix=FORCEALIGNED

;--- main.ll
@arr = internal unnamed_addr global [32 x i32] zeroinitializer, align 16

; PR12250
define i32 @test1() {
vector.ph:
  br label %vector.body

vector.body:
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds [32 x i32], [32 x i32]* @arr, i32 0, i32 %index
  %1 = bitcast i32* %0 to <4 x i32>*
  %wide.load = load <4 x i32>, <4 x i32>* %1, align 16
  %2 = add nsw <4 x i32> %wide.load, <i32 10, i32 10, i32 10, i32 10>
  %3 = xor <4 x i32> %2, <i32 123345, i32 123345, i32 123345, i32 123345>
  %4 = add nsw <4 x i32> %3, <i32 112, i32 112, i32 112, i32 112>
  %5 = xor <4 x i32> %4, <i32 543345, i32 543345, i32 543345, i32 543345>
  %6 = add nsw <4 x i32> %5, <i32 73, i32 73, i32 73, i32 73>
  %7 = xor <4 x i32> %6, <i32 345987, i32 345987, i32 345987, i32 345987>
  %8 = add nsw <4 x i32> %7, <i32 48, i32 48, i32 48, i32 48>
  %9 = xor <4 x i32> %8, <i32 123987, i32 123987, i32 123987, i32 123987>
  store <4 x i32> %9, <4 x i32>* %1, align 16
  %index.next = add i32 %index, 4
  %10 = icmp eq i32 %index.next, 32
  br i1 %10, label %middle.block, label %vector.body

middle.block:
  ret i32 0

; We can't fold the spill into a padd unless the stack is aligned. Just spilling
; doesn't force stack realignment though
; UNALIGNED-LABEL: @test1
; UNALIGNED-NOT: andl $-{{..}}, %esp
; UNALIGNED: movdqu {{.*}} # 16-byte Spill
; UNALIGNED-NOT: paddd {{.*}} # 16-byte Folded Reload

; ALIGNED-LABEL: @test1
; ALIGNED-NOT: andl $-{{..}}, %esp
; ALIGNED: movdqa {{.*}} # 16-byte Spill
; ALIGNED: paddd {{.*}} # 16-byte Folded Reload

; FORCEALIGNED-LABEL: @test1
; FORCEALIGNED: andl $-{{..}}, %esp
; FORCEALIGNED: movdqa {{.*}} # 16-byte Spill
; FORCEALIGNED: paddd {{.*}} # 16-byte Folded Reload
}
!llvm.module.flags = !{!0}
;--- align4.ll
!0 = !{i32 2, !"override-stack-alignment", i32 4}
;--- align16.ll
!0 = !{i32 2, !"override-stack-alignment", i32 16}

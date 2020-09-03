; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu  -O3 | FileCheck %s

; Function Attrs: uwtable
; When tail-duplicating during placement, we work backward from blocks with
; multiple successors. In this case, the block dup1 gets duplicated into dup2
; and if.then64, and then the block dup2 only gets duplicated into land.lhs.true.

define void @partial_tail_dup(i1 %a1, i1 %a2, i32* %a4, i32* %a5, i8* %a6, i32 %a7) #0 align 2  !prof !1 {
; CHECK-LABEL: partial_tail_dup:
; CHECK:        # %bb.0: # %entry
; CHECK-NEXT:   .p2align 4, 0x90
; CHECK-NEXT:   .LBB0_1: # %for.cond
; CHECK-NEXT:   # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:	  testb	$1, %dil
; CHECK-NEXT:	  je	.LBB0_3
; CHECK-NEXT:   # %bb.2: # %land.lhs.true
; CHECK-NEXT:   # in Loop: Header=BB0_1 Depth=1
; CHECK-NEXT:	  movl	$10, (%rdx)
; CHECK-NEXT:	  movl	$2, (%rcx)
; CHECK-NEXT:	  testl	%r9d, %r9d
; CHECK-NEXT:	  je	.LBB0_1
; CHECK-NEXT:	  jmp	.LBB0_8
; CHECK-NEXT:	  .p2align	4, 0x90
; CHECK-NEXT:   .LBB0_6: # %dup2
; CHECK-NEXT:   # in Loop: Header=BB0_1 Depth=1
; CHECK-NEXT:	  movl	$2, (%rcx)
; CHECK-NEXT:	  testl	%r9d, %r9d
; CHECK-NEXT:	  je	.LBB0_1
; CHECK-NEXT:	  jmp	.LBB0_8
; CHECK-NEXT:    .p2align 4, 0x90
; CHECK-NEXT:  .LBB0_3: # %if.end56
; CHECK-NEXT:    # in Loop: Header=BB0_1 Depth=1
; CHECK-NEXT:    testb $1, %sil
; CHECK-NEXT:    je .LBB0_5
; CHECK-NEXT:  # %bb.4: # %if.then64
; CHECK-NEXT:    # in Loop: Header=BB0_1 Depth=1
; CHECK-NEXT:    movb $1, (%r8)
; CHECK-NEXT:    testl %r9d, %r9d
; CHECK-NEXT:    je .LBB0_1
; CHECK-NEXT:    jmp .LBB0_8
; CHECK-NEXT:  .LBB0_5: # %if.end70
; CHECK-NEXT:    # in Loop: Header=BB0_1 Depth=1
; CHECK-NEXT:    movl $12, (%rdx)
; CHECK-NEXT:    jne .LBB0_6  
; CHECK-NEXT:  .LBB0_8: # %for.end
; CHECK-NEXT:    retq
entry:
  br label %for.cond

for.cond:                                      
  br i1 %a1, label %land.lhs.true, label %if.end56

land.lhs.true:                                   
  store i32 10, i32* %a4, align 8
  br label %dup2

if.end56:                                        
  br i1 %a2, label %if.then64, label %if.end70, !prof !2

if.then64:                                       
  store i8 1, i8* %a6, align 1
  br label %dup1

if.end70:                                        
  store i32 12, i32* %a4, align 8
  br i1 %a2, label %dup2, label %for.end

dup2:                                            
  store i32 2, i32* %a5, align 4
  br label %dup1

dup1:                                            
  %val = load i32, i32* %a4, align 8
  %switch = icmp ult i32 %a7, 1
  br i1 %switch, label %for.cond, label %for.end, !prof !3

for.end:                                         
  ret void
}

attributes #0 = { uwtable }

!1 = !{!"function_entry_count", i64 2}
!2 = !{!"branch_weights", i32 5, i32 1}
!3 = !{!"branch_weights", i32 5, i32 1}

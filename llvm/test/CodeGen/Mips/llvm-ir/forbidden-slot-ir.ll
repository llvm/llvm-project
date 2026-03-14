target triple = "mipsisa32r6el-unknown-linux-gnu"

; RUN: llc -filetype=asm %s -o - | FileCheck %s --check-prefix=MIPSELR6
; Function Attrs: noinline nounwind optnone uwtable
define i1 @foo0() nounwind {
; MIPSELR6:      bnezc	$1, $BB0_2
; MIPSELR6-NEXT: nop
; MIPSELR6:      jr	$ra
entry:
  %0 = icmp eq i32 0, 1
  br i1 %0, label %2, label %3
  ret i1 %0
2:                                                
  ret i1 %0
3:                                                
  ret i1 %0
}

define i32 @foo1() nounwind {
; MIPSELR6:      addiu	$2, $2, 1
; MIPSELR6-NEXT: .set	noreorder
; MIPSELR6-NEXT: beqzc	$2, $tmp0
; MIPSELR6-NEXT: nop
; MIPSELR6-NEXT: .set	reorder
; MIPSELR6:      jrc	$ra
entry:
  %0 = tail call i32 asm "1: addiu $0, $0, 1; beqzc $0, 1b", "=r"() nounwind
  ret i32 %0
}

define i32 @foo2() nounwind {
; MIPSELR6:      .set	push
; MIPSELR6-NEXT: .set	at
; MIPSELR6-NEXT: .set	macro
; MIPSELR6-NEXT: .set	reorder
; MIPSELR6:      .set	noreorder
; MIPSELR6-NEXT: beqzc	$9, End
; MIPSELR6-NEXT: nop
; MIPSELR6-NEXT: .set	reorder
; MIPSELR6:      addiu	$9, $9, 1
entry:
  %0 = tail call i32 asm "beqzc $$t1, End", "=r"() nounwind
  %1 = tail call i32 asm "addiu $$t1, $$t1, 1", "=r"() nounwind
  %2 = add nsw i32 %1, %0
  ret i32 %2
}

define i32 @foo3() nounwind {
; MIPSELR6:      addiu	$2, $2, 1
; MIPSELR6-NEXT: .set	noreorder
; MIPSELR6-NEXT: beqzc	$2, $tmp1
; MIPSELR6-NEXT: nop
; MIPSELR6-NEXT: .set	noreorder
; MIPSELR6-NEXT: j	End
; MIPSELR6-NEXT: nop
; MIPSELR6-NEXT: .set	reorder
entry:
  %0 = tail call i32 asm "1: addiu $0, $0, 1; beqzc $0, 1b; j End", "=r"() nounwind
  ret i32 %0
}

define i32 @foo4() nounwind {
; MIPSELR6:      addiu	$2, $2, 1
; MIPSELR6-NEXT: .set	noreorder
; MIPSELR6-NEXT: beqzc  $2, $tmp2
; MIPSELR6-NEXT: addiu	$2, $2, 1
; MIPSELR6-NEXT: .set	reorder
entry:
  %0 = tail call i32 asm "1: addiu $0, $0, 1; beqzc $0, 1b; addiu $0, $0, 1", "=r"() nounwind
  ret i32 %0
}

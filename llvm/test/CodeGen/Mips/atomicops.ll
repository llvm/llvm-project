; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@.str = private unnamed_addr constant [8 x i8] c"%d, %d\0A\00", align 1

define i32 @foo(i32* %mem, i32 %val, i32 %c) nounwind {
entry:
  %0 = atomicrmw add i32* %mem, i32 %val seq_cst
  %add = add nsw i32 %0, %c
  ret i32 %add
; 16: foo:
; 16:	lw	${{[0-9]+}}, %call16(__sync_synchronize)(${{[0-9]+}})
; 16: 	lw	${{[0-9]+}}, %call16(__sync_fetch_and_add_4)(${{[0-9]+}})
}

define i32 @main() nounwind {
entry:
  %x = alloca i32, align 4
  store volatile i32 0, i32* %x, align 4
  %0 = atomicrmw add i32* %x, i32 1 seq_cst
  %add.i = add nsw i32 %0, 2
  %1 = load volatile i32* %x, align 4
  %call1 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([8 x i8]* @.str, i32 0, i32 0), i32 %add.i, i32 %1) nounwind
  %2 = cmpxchg i32* %x, i32 1, i32 2 seq_cst
  %3 = load volatile i32* %x, align 4
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([8 x i8]* @.str, i32 0, i32 0), i32 %2, i32 %3) nounwind
  %4 = atomicrmw xchg i32* %x, i32 1 seq_cst
  %5 = load volatile i32* %x, align 4
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([8 x i8]* @.str, i32 0, i32 0), i32 %4, i32 %5) nounwind
; 16: main:
; 16:	lw	${{[0-9]+}}, %call16(__sync_synchronize)(${{[0-9]+}})
; 16: 	lw	${{[0-9]+}}, %call16(__sync_fetch_and_add_4)(${{[0-9]+}})
; 16:	lw	${{[0-9]+}}, %call16(__sync_val_compare_and_swap_4)(${{[0-9]+}})
; 16:	lw	${{[0-9]+}}, %call16(__sync_lock_test_and_set_4)(${{[0-9]+}})

  ret i32 0
}

declare i32 @printf(i8* nocapture, ...) nounwind



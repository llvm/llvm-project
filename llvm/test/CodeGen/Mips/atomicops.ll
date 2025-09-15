; RUN: llc  -mtriple=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@.str = private unnamed_addr constant [8 x i8] c"%d, %d\0A\00", align 1

define i32 @foo(ptr %mem, i32 %val, i32 %c) nounwind {
entry:
  %0 = atomicrmw add ptr %mem, i32 %val seq_cst
  %add = add nsw i32 %0, %c
  ret i32 %add
; 16-LABEL: foo:
; 16:	lw	${{[0-9]+}}, %call16(__sync_synchronize)(${{[0-9]+}})
; 16: 	lw	${{[0-9]+}}, %call16(__sync_fetch_and_add_4)(${{[0-9]+}})
}

define i32 @atomic_load_sub(ptr %mem, i32 %val, i32 %c) nounwind {
; 16-LABEL: atomic_load_sub:
; 16:	lw	${{[0-9]+}}, %call16(__sync_synchronize)(${{[0-9]+}})
; 16: 	lw	${{[0-9]+}}, %call16(__sync_fetch_and_sub_4)(${{[0-9]+}})
entry:
  %0 = atomicrmw sub ptr %mem, i32 %val seq_cst
  ret i32 %0
}

define i32 @main() nounwind {
entry:
  %x = alloca i32, align 4
  store volatile i32 0, ptr %x, align 4
  %0 = atomicrmw add ptr %x, i32 1 seq_cst
  %add.i = add nsw i32 %0, 2
  %1 = load volatile i32, ptr %x, align 4
  %call1 = call i32 (ptr, ...) @printf(ptr @.str, i32 %add.i, i32 %1) nounwind
  %pair = cmpxchg ptr %x, i32 1, i32 2 seq_cst seq_cst
  %2 = extractvalue { i32, i1 } %pair, 0
  %3 = load volatile i32, ptr %x, align 4
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, i32 %2, i32 %3) nounwind
  %4 = atomicrmw xchg ptr %x, i32 1 seq_cst
  %5 = load volatile i32, ptr %x, align 4
  %call3 = call i32 (ptr, ...) @printf(ptr @.str, i32 %4, i32 %5) nounwind
; 16-LABEL: main:
; 16:	lw	${{[0-9]+}}, %call16(__sync_synchronize)(${{[0-9]+}})
; 16: 	lw	${{[0-9]+}}, %call16(__sync_fetch_and_add_4)(${{[0-9]+}})
; 16:	lw	${{[0-9]+}}, %call16(__sync_val_compare_and_swap_4)(${{[0-9]+}})
; 16:	lw	${{[0-9]+}}, %call16(__sync_lock_test_and_set_4)(${{[0-9]+}})

  ret i32 0
}

declare i32 @printf(ptr nocapture, ...) nounwind

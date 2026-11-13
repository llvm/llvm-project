; RUN: not llubi --verbose --entry-function=noundef_poison < %s 2>&1 | FileCheck %s --check-prefix=NOUNDEF-POISON
; RUN: not llubi --verbose --entry-function=dereferenceable_null < %s 2>&1 | FileCheck %s --check-prefix=DEREF-NULL
; RUN: not llubi --verbose --entry-function=dereferenceable_or_null_oob < %s 2>&1 | FileCheck %s --check-prefix=DEREF-OR-NULL-OOB

define void @noundef_poison() {
entry:
  br label %bb

bb:
  %res = phi i32 [ poison, %entry ], !noundef !{}
  ret void
}
; NOUNDEF-POISON: Entering function: noundef_poison
; NOUNDEF-POISON-NEXT:   br label %bb jump to %bb
; NOUNDEF-POISON-NEXT: Stacktrace:
; NOUNDEF-POISON-NEXT: #0   %res = phi i32 [ poison, %entry ], !noundef !{{[0-9]+}} at @noundef_poison <stdin>:10
; NOUNDEF-POISON-NEXT: Immediate UB detected: The value poison violates !noundef metadata.
; NOUNDEF-POISON-NEXT: error: Execution of function 'noundef_poison' failed.

define void @dereferenceable_null() {
entry:
  br label %bb

bb:
  %res = phi ptr [ null, %entry ], !dereferenceable !{i64 4}
  ret void
}
; DEREF-NULL: Entering function: dereferenceable_null
; DEREF-NULL-NEXT:   br label %bb jump to %bb
; DEREF-NULL-NEXT: Stacktrace:
; DEREF-NULL-NEXT: #0   %res = phi ptr [ null, %entry ], !dereferenceable !{{[0-9]+}} at @dereferenceable_null <stdin>:25
; DEREF-NULL-NEXT: Immediate UB detected: The value ptr 0x0 [nullary] violates !dereferenceable !{i64 4} metadata.
; DEREF-NULL-NEXT: error: Execution of function 'dereferenceable_null' failed.

define void @dereferenceable_or_null_oob() {
entry:
  %alloc = alloca i32
  br label %bb

bb:
  %res = phi ptr [ %alloc, %entry ], !dereferenceable_or_null !{i64 8}
  ret void
}
; DEREF-OR-NULL-OOB: Entering function: dereferenceable_or_null_oob
; DEREF-OR-NULL-OOB-NEXT:   %alloc = alloca i32, align 4 => ptr 0x8 [alloc]
; DEREF-OR-NULL-OOB-NEXT:   br label %bb jump to %bb
; DEREF-OR-NULL-OOB-NEXT: Stacktrace:
; DEREF-OR-NULL-OOB-NEXT: #0   %res = phi ptr [ %alloc, %entry ], !dereferenceable_or_null !{{[0-9]+}} at @dereferenceable_or_null_oob <stdin>:41
; DEREF-OR-NULL-OOB-NEXT: Immediate UB detected: The value ptr 0x8 [alloc] violates !dereferenceable_or_null!{i64 8} metadata.
; DEREF-OR-NULL-OOB-NEXT: error: Execution of function 'dereferenceable_or_null_oob' failed.

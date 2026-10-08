; RUN: not llubi --verbose --entry-function=oracle_out_of_bounds < %s 2>&1 | FileCheck %s --check-prefix=ORACLE-OOB
; RUN: not llubi --verbose --entry-function=oracle_poison_noundef_ret < %s 2>&1 | FileCheck %s --check-prefix=NOUNDEF-RET
; RUN: not llubi --verbose --entry-function=oracle_poison_noundef_arg < %s 2>&1 | FileCheck %s --check-prefix=NOUNDEF-ARG

@a = global [6 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5]

; Returns one element more than the number of bytes from %p to %end.
define i64 @oracle_off_by_one(ptr %p, ptr %end) memory(none) nounwind nosync willreturn {
  %p.int = ptrtoaddr ptr %p to i64
  %end.int = ptrtoaddr ptr %end to i64
  %diff = sub i64 %end.int, %p.int
  %n = add i64 %diff, 4
  ret i64 %n
}

define noundef i64 @oracle_noundef_ret(i64 %n) memory(none) nounwind nosync willreturn {
  ret i64 poison
}

define i64 @oracle_noundef_arg(i64 noundef %n) memory(none) nounwind nosync willreturn {
  ret i64 %n
}

define void @oracle_out_of_bounds() {
; ORACLE-OOB: Entering function: oracle_out_of_bounds
; ORACLE-OOB-NEXT:   %p = getelementptr i32, ptr @a, i64 4 => ptr 0x20 [@a + 16]
; ORACLE-OOB-NEXT:   %end = getelementptr i32, ptr @a, i64 6 => ptr 0x28 [@a + 24]
; ORACLE-OOB-NEXT: Entering function: oracle_off_by_one
; ORACLE-OOB-NEXT:   ptr %p = ptr 0x20 [@a + 16]
; ORACLE-OOB-NEXT:   ptr %end = ptr 0x28 [@a + 24]
; ORACLE-OOB-NEXT:   %p.int = ptrtoaddr ptr %p to i64 => i64 32
; ORACLE-OOB-NEXT:   %end.int = ptrtoaddr ptr %end to i64 => i64 40
; ORACLE-OOB-NEXT:   %diff = sub i64 %end.int, %p.int => i64 8
; ORACLE-OOB-NEXT:   %n = add i64 %diff, 4 => i64 12
; ORACLE-OOB-NEXT:   ret i64 %n
; ORACLE-OOB-NEXT: Exiting function: oracle_off_by_one
; ORACLE-OOB-NEXT: Stacktrace:
; ORACLE-OOB-NEXT: #0   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %p, i1 false, ptr @oracle_off_by_one, ptr %p, ptr %end) at @oracle_out_of_bounds <stdin>:{{[0-9]+}}
; ORACLE-OOB-NEXT: Immediate UB detected: Memory access is out of bounds. Accessed size: 12, Address: 0x20, Object base: 0x10, Object size: 24.
; ORACLE-OOB-NEXT: error: Execution of function 'oracle_out_of_bounds' failed.
  %p = getelementptr i32, ptr @a, i64 4
  %end = getelementptr i32, ptr @a, i64 6
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %p, i1 false, ptr @oracle_off_by_one, ptr %p, ptr %end)
  ret void
}

define void @oracle_poison_noundef_ret() {
; NOUNDEF-RET: Entering function: oracle_poison_noundef_ret
; NOUNDEF-RET-NEXT: Entering function: oracle_noundef_ret
; NOUNDEF-RET-NEXT:   i64 %n = i64 4
; NOUNDEF-RET-NEXT:   ret i64 poison
; NOUNDEF-RET-NEXT: Exiting function: oracle_noundef_ret
; NOUNDEF-RET-NEXT: Stacktrace:
; NOUNDEF-RET-NEXT: #0   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, ptr @oracle_noundef_ret, i64 4) at @oracle_poison_noundef_ret <stdin>:{{[0-9]+}}
; NOUNDEF-RET-NEXT: Immediate UB detected: The value poison violates noundef attribute.
; NOUNDEF-RET-NEXT: error: Execution of function 'oracle_poison_noundef_ret' failed.
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, ptr @oracle_noundef_ret, i64 4)
  ret void
}

define void @oracle_poison_noundef_arg() {
; NOUNDEF-ARG: Entering function: oracle_poison_noundef_arg
; NOUNDEF-ARG-NEXT: Stacktrace:
; NOUNDEF-ARG-NEXT: #0   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, ptr @oracle_noundef_arg, i64 poison) at @oracle_poison_noundef_arg <stdin>:{{[0-9]+}}
; NOUNDEF-ARG-NEXT: Immediate UB detected: The value poison violates noundef attribute.
; NOUNDEF-ARG-NEXT: error: Execution of function 'oracle_poison_noundef_arg' failed.
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, ptr @oracle_noundef_arg, i64 poison)
  ret void
}

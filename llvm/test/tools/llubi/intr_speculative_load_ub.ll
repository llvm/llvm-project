; RUN: not llubi --verbose --entry-function=out_of_bounds < %s 2>&1 | FileCheck %s --check-prefix=OOB
; RUN: not llubi --verbose --entry-function=from_end_out_of_bounds < %s 2>&1 | FileCheck %s --check-prefix=FROM-END-OOB
; RUN: not llubi --verbose --entry-function=exceeds_size < %s 2>&1 | FileCheck %s --check-prefix=EXCEEDS-SIZE
; RUN: not llubi --verbose --entry-function=poison_num_bytes < %s 2>&1 | FileCheck %s --check-prefix=POISON-N
; RUN: not llubi --verbose --entry-function=poison_pointer < %s 2>&1 | FileCheck %s --check-prefix=POISON-PTR
; RUN: not llubi --verbose --entry-function=oracle_out_of_bounds < %s 2>&1 | FileCheck %s --check-prefix=ORACLE-OOB
; RUN: not llubi --verbose --entry-function=oracle_declaration < %s 2>&1 | FileCheck %s --check-prefix=ORACLE-DECL

@a = global [2 x i32] [i32 0, i32 1]

; Returns one element more than the number of bytes from %p to %end.
define i64 @oracle_off_by_one(ptr %p, ptr %end) memory(none) nounwind nosync willreturn {
  %p.int = ptrtoaddr ptr %p to i64
  %end.int = ptrtoaddr ptr %end to i64
  %diff = sub i64 %end.int, %p.int
  %n = add i64 %diff, 4
  ret i64 %n
}

declare i64 @oracle_decl(i64) memory(none) nounwind nosync willreturn

define void @out_of_bounds() {
; OOB: Entering function: out_of_bounds
; OOB-NEXT: Unrecognized instruction:   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, i64 12)
; OOB-NEXT: error: Execution of function 'out_of_bounds' failed.
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, i64 12)
  ret void
}

define void @from_end_out_of_bounds() {
; FROM-END-OOB: Entering function: from_end_out_of_bounds
; FROM-END-OOB-NEXT: Unrecognized instruction:   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr getelementptr (i8, ptr @a, i64 -8), i1 true, i64 12)
; FROM-END-OOB-NEXT: error: Execution of function 'from_end_out_of_bounds' failed.
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr getelementptr (i8, ptr @a, i64 -8), i1 true, i64 12)
  ret void
}

define void @exceeds_size() {
; EXCEEDS-SIZE: Entering function: exceeds_size
; EXCEEDS-SIZE-NEXT: Unrecognized instruction:   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, i64 17)
; EXCEEDS-SIZE-NEXT: error: Execution of function 'exceeds_size' failed.
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, i64 17)
  ret void
}

define void @poison_num_bytes() {
; POISON-N: Entering function: poison_num_bytes
; POISON-N-NEXT: Unrecognized instruction:   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, i64 poison)
; POISON-N-NEXT: error: Execution of function 'poison_num_bytes' failed.
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, i64 poison)
  ret void
}

define void @poison_pointer() {
; POISON-PTR: Entering function: poison_pointer
; POISON-PTR-NEXT: Unrecognized instruction:   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr poison, i1 false, i64 0)
; POISON-PTR-NEXT: error: Execution of function 'poison_pointer' failed.
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr poison, i1 false, i64 0)
  ret void
}

define void @oracle_out_of_bounds() {
; ORACLE-OOB: Entering function: oracle_out_of_bounds
; ORACLE-OOB-NEXT: Unrecognized instruction:   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, ptr @oracle_off_by_one, ptr @a, ptr getelementptr (i8, ptr @a, i64 8))
; ORACLE-OOB-NEXT: error: Execution of function 'oracle_out_of_bounds' failed.
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, ptr @oracle_off_by_one, ptr @a, ptr getelementptr (i8, ptr @a, i64 8))
  ret void
}

define void @oracle_declaration() {
; ORACLE-DECL: Entering function: oracle_declaration
; ORACLE-DECL-NEXT: Unrecognized instruction:   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, ptr @oracle_decl, i64 4)
; ORACLE-DECL-NEXT: error: Execution of function 'oracle_declaration' failed.
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, ptr @oracle_decl, i64 4)
  ret void
}

; RUN: not llubi --verbose --entry-function=out_of_bounds < %s 2>&1 | FileCheck %s --check-prefix=OOB
; RUN: not llubi --verbose --entry-function=from_end_out_of_bounds < %s 2>&1 | FileCheck %s --check-prefix=FROM-END-OOB
; RUN: not llubi --verbose --entry-function=exceeds_size < %s 2>&1 | FileCheck %s --check-prefix=EXCEEDS-SIZE
; RUN: not llubi --verbose --entry-function=poison_num_bytes < %s 2>&1 | FileCheck %s --check-prefix=POISON-N
; RUN: not llubi --verbose --entry-function=poison_pointer < %s 2>&1 | FileCheck %s --check-prefix=POISON-PTR

@a = global [2 x i32] [i32 0, i32 1]

define void @out_of_bounds() {
; OOB: Entering function: out_of_bounds
; OOB-NEXT: Stacktrace:
; OOB-NEXT: #0   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, i64 12) at @out_of_bounds <stdin>:{{[0-9]+}}
; OOB-NEXT: Immediate UB detected: Memory access is out of bounds. Accessed size: 12, Address: 0x8, Object base: 0x8, Object size: 8.
; OOB-NEXT: error: Execution of function 'out_of_bounds' failed.
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, i64 12)
  ret void
}

define void @from_end_out_of_bounds() {
; FROM-END-OOB: Entering function: from_end_out_of_bounds
; FROM-END-OOB-NEXT: Stacktrace:
; FROM-END-OOB-NEXT: #0   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr getelementptr (i8, ptr @a, i64 -8), i1 true, i64 12) at @from_end_out_of_bounds <stdin>:{{[0-9]+}}
; FROM-END-OOB-NEXT: Immediate UB detected: Memory access is out of bounds. Accessed size: 12, Address: 0x4, Object base: 0x8, Object size: 8.
; FROM-END-OOB-NEXT: error: Execution of function 'from_end_out_of_bounds' failed.
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr getelementptr (i8, ptr @a, i64 -8), i1 true, i64 12)
  ret void
}

define void @exceeds_size() {
; EXCEEDS-SIZE: Entering function: exceeds_size
; EXCEEDS-SIZE-NEXT: Stacktrace:
; EXCEEDS-SIZE-NEXT: #0   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, i64 17) at @exceeds_size <stdin>:{{[0-9]+}}
; EXCEEDS-SIZE-NEXT: Immediate UB detected: llvm.speculative.load number of accessible bytes 17 exceeds the loaded size 16.
; EXCEEDS-SIZE-NEXT: error: Execution of function 'exceeds_size' failed.
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, i64 17)
  ret void
}

define void @poison_num_bytes() {
; POISON-N: Entering function: poison_num_bytes
; POISON-N-NEXT: Stacktrace:
; POISON-N-NEXT: #0   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, i64 poison) at @poison_num_bytes <stdin>:{{[0-9]+}}
; POISON-N-NEXT: Immediate UB detected: llvm.speculative.load with poison number of accessible bytes.
; POISON-N-NEXT: error: Execution of function 'poison_num_bytes' failed.
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, i64 poison)
  ret void
}

define void @poison_pointer() {
; POISON-PTR: Entering function: poison_pointer
; POISON-PTR-NEXT: Stacktrace:
; POISON-PTR-NEXT: #0   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr poison, i1 false, i64 0) at @poison_pointer <stdin>:{{[0-9]+}}
; POISON-PTR-NEXT: Immediate UB detected: llvm.speculative.load with poison pointer.
; POISON-PTR-NEXT: error: Execution of function 'poison_pointer' failed.
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr poison, i1 false, i64 0)
  ret void
}

; RUN: llubi --verbose --entry-function=first_bytes < %s 2>&1 | FileCheck %s --check-prefix=FIRST
; RUN: llubi --verbose --entry-function=last_bytes < %s 2>&1 | FileCheck %s --check-prefix=LAST
; RUN: llubi --verbose --entry-function=past_end < %s 2>&1 | FileCheck %s --check-prefix=PAST-END
; RUN: not llubi --verbose --entry-function=oracle_load < %s 2>&1 | FileCheck %s --check-prefix=ORACLE

@a = global [6 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5]

; Returns the number of bytes from %p to %end, clamped to 16.
define internal i64 @oracle(ptr %p, ptr %end) memory(none) nounwind nosync willreturn "speculative-load-oracle" {
  %p.int = ptrtoaddr ptr %p to i64
  %end.int = ptrtoaddr ptr %end to i64
  %diff = sub i64 %end.int, %p.int
  %n = call i64 @llvm.umin.i64(i64 %diff, i64 16)
  ret i64 %n
}

define void @first_bytes() {
; FIRST: Entering function: first_bytes
; FIRST-NEXT:   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, i64 8) => { i32 0, i32 1, poison, poison }
; FIRST-NEXT:   ret void
; FIRST-NEXT: Exiting function: first_bytes
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 false, i64 8)
  ret void
}

define void @last_bytes() {
; LAST: Entering function: last_bytes
; LAST-NEXT:   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 true, i64 8) => { poison, poison, i32 2, i32 3 }
; LAST-NEXT:   ret void
; LAST-NEXT: Exiting function: last_bytes
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 true, i64 8)
  ret void
}

define void @past_end() {
; PAST-END: Entering function: past_end
; PAST-END-NEXT:   %p = getelementptr i32, ptr @a, i64 4 => ptr 0x20 [@a + 16]
; PAST-END-NEXT:   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %p, i1 false, i64 8) => { i32 4, i32 5, poison, poison }
; PAST-END-NEXT:   ret void
; PAST-END-NEXT: Exiting function: past_end
  %p = getelementptr i32, ptr @a, i64 4
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %p, i1 false, i64 8)
  ret void
}

define void @oracle_load() {
; ORACLE: Entering function: oracle_load
; ORACLE-NEXT:   %p = getelementptr i32, ptr @a, i64 4 => ptr 0x20 [@a + 16]
; ORACLE-NEXT:   %end = getelementptr i32, ptr @a, i64 6 => ptr 0x28 [@a + 24]
; ORACLE-NEXT: Unrecognized instruction:   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %p, i1 false, ptr @oracle, ptr %p, ptr %end)
; ORACLE-NEXT: error: Execution of function 'oracle_load' failed.
  %p = getelementptr i32, ptr @a, i64 4
  %end = getelementptr i32, ptr @a, i64 6
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %p, i1 false, ptr @oracle, ptr %p, ptr %end)
  ret void
}

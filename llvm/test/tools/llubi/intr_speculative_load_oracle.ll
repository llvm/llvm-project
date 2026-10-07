; RUN: llubi --verbose --entry-function=oracle_load < %s 2>&1 | FileCheck %s --check-prefix=ORACLE
; RUN: llubi --verbose --entry-function=nested_oracle_load < %s 2>&1 | FileCheck %s --check-prefix=NESTED

@a = global [6 x i32] [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5]

; Returns the number of bytes from %p to %end, clamped to 16.
define i64 @oracle(ptr %p, ptr %end) memory(none) nounwind nosync willreturn {
  %p.int = ptrtoaddr ptr %p to i64
  %end.int = ptrtoaddr ptr %end to i64
  %diff = sub i64 %end.int, %p.int
  %n = call i64 @llvm.umin.i64(i64 %diff, i64 16)
  ret i64 %n
}

; Returns the second element at %p plus 3, loaded via an oracle-form
; llvm.speculative.load.
define i64 @oracle_nested(ptr %p) memory(argmem: read) nounwind nosync willreturn {
  %v = call <2 x i32> (ptr, i1, ...) @llvm.speculative.load.v2i32.p0(ptr %p, i1 false, ptr @oracle_const, i64 8)
  %e = extractelement <2 x i32> %v, i64 1
  %z = zext i32 %e to i64
  %n = add i64 %z, 3
  ret i64 %n
}

define i64 @oracle_const(i64 %n) memory(none) nounwind nosync willreturn {
  ret i64 %n
}

define void @oracle_load() {
; ORACLE: Entering function: oracle_load
; ORACLE-NEXT:   %p = getelementptr i32, ptr @a, i64 4 => ptr 0x20 [@a + 16]
; ORACLE-NEXT:   %end = getelementptr i32, ptr @a, i64 6 => ptr 0x28 [@a + 24]
; ORACLE-NEXT: Entering function: oracle
; ORACLE-NEXT:   ptr %p = ptr 0x20 [@a + 16]
; ORACLE-NEXT:   ptr %end = ptr 0x28 [@a + 24]
; ORACLE-NEXT:   %p.int = ptrtoaddr ptr %p to i64 => i64 32
; ORACLE-NEXT:   %end.int = ptrtoaddr ptr %end to i64 => i64 40
; ORACLE-NEXT:   %diff = sub i64 %end.int, %p.int => i64 8
; ORACLE-NEXT:   %n = call i64 @llvm.umin.i64(i64 %diff, i64 16) => i64 8
; ORACLE-NEXT:   ret i64 %n
; ORACLE-NEXT: Exiting function: oracle
; ORACLE-NEXT:   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %p, i1 false, ptr @oracle, ptr %p, ptr %end) => { i32 4, i32 5, poison, poison }
; ORACLE-NEXT:   ret void
; ORACLE-NEXT: Exiting function: oracle_load
  %p = getelementptr i32, ptr @a, i64 4
  %end = getelementptr i32, ptr @a, i64 6
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %p, i1 false, ptr @oracle, ptr %p, ptr %end)
  ret void
}

define void @nested_oracle_load() {
; NESTED: Entering function: nested_oracle_load
; NESTED-NEXT: Entering function: oracle_nested
; NESTED-NEXT:   ptr %p = ptr 0x10 [@a]
; NESTED-NEXT: Entering function: oracle_const
; NESTED-NEXT:   i64 %n = i64 8
; NESTED-NEXT:   ret i64 %n
; NESTED-NEXT: Exiting function: oracle_const
; NESTED-NEXT:   %v = call <2 x i32> (ptr, i1, ...) @llvm.speculative.load.v2i32.p0(ptr %p, i1 false, ptr @oracle_const, i64 8) => { i32 0, i32 1 }
; NESTED-NEXT:   %e = extractelement <2 x i32> %v, i64 1 => i32 1
; NESTED-NEXT:   %z = zext i32 %e to i64 => i64 1
; NESTED-NEXT:   %n = add i64 %z, 3 => i64 4
; NESTED-NEXT:   ret i64 %n
; NESTED-NEXT: Exiting function: oracle_nested
; NESTED-NEXT:   %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 true, ptr @oracle_nested, ptr @a) => { poison, poison, poison, i32 3 }
; NESTED-NEXT:   ret void
; NESTED-NEXT: Exiting function: nested_oracle_load
  %r = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr @a, i1 true, ptr @oracle_nested, ptr @a)
  ret void
}

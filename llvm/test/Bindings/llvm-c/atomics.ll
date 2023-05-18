; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo


define void @fence_instrs() {
  fence acquire
  fence release
  fence acq_rel
  fence seq_cst

  fence syncscope("singlethread") acquire
  fence syncscope("singlethread") release
  fence syncscope("singlethread") acq_rel
  fence syncscope("singlethread") seq_cst

  ret void
}

define void @atomic_load_store(ptr %word) {
  ; Test different atomic loads
  %ld.1 = load atomic i32, ptr %word monotonic, align 4
  %ld.2 = load atomic volatile i32, ptr %word acquire, align 4
  %ld.3 = load atomic volatile i32, ptr %word seq_cst, align 4
  %ld.4 = load atomic volatile i32, ptr %word syncscope("singlethread") acquire, align 4
  %ld.5 = load atomic volatile i32, ptr %word syncscope("singlethread") seq_cst, align 4
  %ld.6 = load atomic i32, ptr %word syncscope("singlethread") seq_cst, align 4

  ; Test different atomic stores
  store atomic i32 1, ptr %word monotonic, align 4
  store atomic volatile i32 2, ptr %word release, align 4
  store atomic volatile i32 3, ptr %word seq_cst, align 4
  store atomic volatile i32 4, ptr %word syncscope("singlethread") release, align 4
  store atomic volatile i32 5, ptr %word syncscope("singlethread") seq_cst, align 4
  store atomic i32 6, ptr %word syncscope("singlethread") seq_cst, align 4
  ret void
}

define i32 @main() {
  %1 = alloca i32, align 4
  %2 = cmpxchg ptr %1, i32 2, i32 3 seq_cst acquire
  %3 = extractvalue { i32, i1 } %2, 0
  ret i32 %3
}

; RUN: llvm-as < %s -o - | llvm-dis - | FileCheck %s

;CHECK-LABEL: main
;CHECK: store ppc_fp128 0xM0000000000000000FFFFFFFFFFFFFFFF

define i32 @main() local_unnamed_addr {
_main_entry:
  %e3 = alloca ppc_fp128, align 16
  store ppc_fp128 0xM0000000000000000FFFFFFFFFFFFFFFF, ppc_fp128* %e3, align 16
  %0 = call i64 @foo( ppc_fp128* nonnull %e3)
  ret i32 undef
}

declare i64 @foo(ppc_fp128 *)


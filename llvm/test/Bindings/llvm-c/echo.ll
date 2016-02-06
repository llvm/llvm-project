; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo

declare void @decl()

; TODO: label, struct and metadata types
define void @types() {
  %1 = alloca half
  %2 = alloca float
  %3 = alloca double
  %4 = alloca x86_fp80
  %5 = alloca fp128
  %6 = alloca ppc_fp128
  %7 = alloca i7
  %8 = alloca void (i1)*
  %9 = alloca [3 x i22]
  %10 = alloca i328 addrspace(5)*
  %11 = alloca <5 x i23*>
  %12 = alloca x86_mmx
  ret void
}

define i32 @iops(i32 %a, i32 %b) {
  %1 = add i32 %a, %b
  %2 = mul i32 %a, %1
  %3 = sub i32 %2, %1
  %4 = udiv i32 %3, %b
  %5 = sdiv i32 %2, %4
  %6 = urem i32 %3, %5
  %7 = srem i32 %2, %6
  %8 = shl i32 %1, %b
  %9 = lshr i32 %a, %7
  %10 = ashr i32 %b, %8
  %11 = and i32 %9, %10
  %12 = or i32 %2, %11
  %13 = xor i32 %12, %4
  ret i32 %13
}

define i32 @call() {
  %1 = call i32 @iops(i32 23, i32 19)
  ret i32 %1
}

; RUN: llc < %s -march=x86-64 | grep movd | count 4

define i64 @foo(x86_mmx* %p) {
  %t = load x86_mmx* %p
  %u = tail call x86_mmx @llvm.x86.mmx.padd.q(x86_mmx %t, x86_mmx %t)
  %s = bitcast x86_mmx %u to i64
  ret i64 %s
}
define i64 @goo(x86_mmx* %p) {
  %t = load x86_mmx* %p
  %u = tail call x86_mmx @llvm.x86.mmx.padd.d(x86_mmx %t, x86_mmx %t)
  %s = bitcast x86_mmx %u to i64
  ret i64 %s
}
define i64 @hoo(x86_mmx* %p) {
  %t = load x86_mmx* %p
  %u = tail call x86_mmx @llvm.x86.mmx.padd.w(x86_mmx %t, x86_mmx %t)
  %s = bitcast x86_mmx %u to i64
  ret i64 %s
}
define i64 @ioo(x86_mmx* %p) {
  %t = load x86_mmx* %p
  %u = tail call x86_mmx @llvm.x86.mmx.padd.b(x86_mmx %t, x86_mmx %t)
  %s = bitcast x86_mmx %u to i64
  ret i64 %s
}

declare x86_mmx @llvm.x86.mmx.padd.b(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.padd.w(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.padd.d(x86_mmx, x86_mmx)
declare x86_mmx @llvm.x86.mmx.padd.q(x86_mmx, x86_mmx)

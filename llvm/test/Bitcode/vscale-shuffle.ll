; RUN: llvm-as < %s | llvm-dis -disable-output
; RUN: verify-uselistorder < %s

define void @f() {
  %l = call <vscale x 16 x i8> @l(<vscale x 16 x i1> splat (i1 true))
  %i = add <vscale x 2 x i64> undef, splat (i64 1)
  unreachable
}

declare <vscale x 16 x i8> @l(<vscale x 16 x i1>)

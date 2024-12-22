; RUN: llvm-as < %s | llvm-dis -disable-output
; RUN: verify-uselistorder < %s

define void @f() {
  %l = call <vscale x 16 x i8> @l(<vscale x 16 x i1> shufflevector (<vscale x 16 x i1> insertelement (<vscale x 16 x i1> undef, i1 true, i32 0), <vscale x 16 x i1> undef, <vscale x 16 x i32> zeroinitializer))
  %i = add <vscale x 2 x i64> undef, shufflevector (<vscale x 2 x i64> insertelement (<vscale x 2 x i64> undef, i64 1, i32 0), <vscale x 2 x i64> undef, <vscale x 2 x i32> zeroinitializer)
  unreachable
}

declare <vscale x 16 x i8> @l(<vscale x 16 x i1>)

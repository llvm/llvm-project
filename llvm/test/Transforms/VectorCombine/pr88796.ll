; REQUIRES: asserts
; RUN: not --crash opt -passes=vector-combine -disable-output %s

define i32 @test() {
entry:
  %0 = tail call i16 @llvm.vector.reduce.and.nxv8i16(<vscale x 8 x i16> trunc (<vscale x 8 x i32> shufflevector (<vscale x 8 x i32> insertelement (<vscale x 8 x i32> poison, i32 268435456, i64 0), <vscale x 8 x i32> poison, <vscale x 8 x i32> zeroinitializer) to <vscale x 8 x i16>))
  ret i32 0
}

declare i16 @llvm.vector.reduce.and.nxv8i16(<vscale x 8 x i16>)


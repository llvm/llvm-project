target triple = "x86_64-unknown-linux-gnu"

%struct.rtx_def = type { i16, i16 }

define void @bar(%struct.rtx_def* %a, i8 %b, i32 %c) {
  call void  @llvm.memset.p0struct.rtx_def.i32(%struct.rtx_def* align 4 %a, i8 %b, i32 %c, i1 true)
  ret void
}

declare void @llvm.memset.p0struct.rtx_def.i32(%struct.rtx_def*, i8, i32, i1)

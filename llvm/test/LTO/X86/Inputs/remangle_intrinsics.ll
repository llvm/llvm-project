target triple = "x86_64-unknown-linux-gnu"

%struct.rtx_def = type { i16, i16 }

define void @bar(ptr %a, i8 %b, i32 %c) {
  call void  @llvm.memset.p0.rtx_def.i32(ptr align 4 %a, i8 %b, i32 %c, i1 true)
  ret void
}

declare void @llvm.memset.p0.rtx_def.i32(ptr, i8, i32, i1)

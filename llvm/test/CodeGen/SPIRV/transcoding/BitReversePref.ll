; RUN: llc -O0 -mtriple=spirv64-unknown-linux %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK:     OpDecorate %[[#FUNC_NAME:]] LinkageAttributes "_Z10BitReversei"
; CHECK-NOT: OpBitReverse
; CHECK:     %[[#]] = OpFunctionCall %[[#]] %[[#FUNC_NAME]]

%"class._ZTSZ4mainE3$_0.anon" = type { i8 }

$_Z10BitReversei = comdat any

define dso_local spir_kernel void @_ZTSZ4mainE15kernel_function() {
entry:
  %call = call spir_func i32 @_Z10BitReversei(i32 1)
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

define linkonce_odr dso_local spir_func i32 @_Z10BitReversei(i32 %value) comdat {
entry:
  %value.addr = alloca i32, align 4
  %reversed = alloca i32, align 4
  store i32 %value, i32* %value.addr, align 4
  %0 = bitcast i32* %reversed to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0)
  store i32 0, i32* %reversed, align 4
  %1 = load i32, i32* %reversed, align 4
  %2 = bitcast i32* %reversed to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %2)
  ret i32 %1
}

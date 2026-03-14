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

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

define linkonce_odr dso_local spir_func i32 @_Z10BitReversei(i32 %value) comdat {
entry:
  %value.addr = alloca i32, align 4
  %reversed = alloca i32, align 4
  store i32 %value, ptr %value.addr, align 4
  %0 = bitcast ptr %reversed to ptr
  call void @llvm.lifetime.start.p0(i64 4, ptr %0)
  store i32 0, ptr %reversed, align 4
  %1 = load i32, ptr %reversed, align 4
  %2 = bitcast ptr %reversed to ptr
  call void @llvm.lifetime.end.p0(i64 4, ptr %2)
  ret i32 %1
}

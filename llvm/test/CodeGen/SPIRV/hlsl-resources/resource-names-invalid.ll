; RUN: not --crash opt -passes=spirv-lower-resource-names -disable-output %s 2>&1 | FileCheck %s

target triple = "spirv1.6-unknown-vulkan1.3-library"

; CHECK: LLVM ERROR: unresolvable SPIR-V resource name

@.str = private unnamed_addr constant [4 x i8] c"buf\00"
@name.ptr = internal global ptr @.str

define void @test() {
  %name = load ptr, ptr @name.ptr
  %handle = call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr %name)
  ret void
}

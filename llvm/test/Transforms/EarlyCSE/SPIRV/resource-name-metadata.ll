; RUN: opt -S -passes=early-cse %s | FileCheck %s
; RUN: opt -S -passes=gvn %s | FileCheck %s

; check that handle intrinsics are deduplicated (or not) according to whether their
; metadata name is unique along with other operands

; CHECK-LABEL: define void @same_name(
; CHECK: [[H:%.*]] = call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.mdname
; CHECK-NOT: call target("spirv.Sampler")
; CHECK: call void @consume(target("spirv.Sampler") [[H]], target("spirv.Sampler") [[H]])
define void @same_name() {
  %a = call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.mdname(i32 0, i32 1, i32 1, i32 0, metadata !{!"a"})
  %b = call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.mdname(i32 0, i32 1, i32 1, i32 0, metadata !{!"a"})
  call void @consume(target("spirv.Sampler") %a, target("spirv.Sampler") %b)
  ret void
}

; CHECK-LABEL: define void @different_names(
; CHECK: [[A:%.*]] = call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.mdname
; CHECK: [[B:%.*]] = call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.mdname
; CHECK: call void @consume(target("spirv.Sampler") [[A]], target("spirv.Sampler") [[B]])
define void @different_names() {
  %a = call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.mdname(i32 0, i32 1, i32 1, i32 0, metadata !{!"a"})
  %b = call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.mdname(i32 0, i32 1, i32 1, i32 0, metadata !{!"b"})
  call void @consume(target("spirv.Sampler") %a, target("spirv.Sampler") %b)
  ret void
}

; CHECK-LABEL: define void @implicit_same_name(
; CHECK: [[H:%.*]] = call target("spirv.Sampler") @llvm.spv.resource.handlefromimplicitbinding.mdname
; CHECK-NOT: call target("spirv.Sampler")
; CHECK: call void @consume(target("spirv.Sampler") [[H]], target("spirv.Sampler") [[H]])
define void @implicit_same_name() {
  %a = call target("spirv.Sampler") @llvm.spv.resource.handlefromimplicitbinding.mdname(i32 0, i32 0, i32 1, i32 0, metadata !{!"a"})
  %b = call target("spirv.Sampler") @llvm.spv.resource.handlefromimplicitbinding.mdname(i32 0, i32 0, i32 1, i32 0, metadata !{!"a"})
  call void @consume(target("spirv.Sampler") %a, target("spirv.Sampler") %b)
  ret void
}

; CHECK-LABEL: define void @implicit_different_names(
; CHECK: [[A:%.*]] = call target("spirv.Sampler") @llvm.spv.resource.handlefromimplicitbinding.mdname
; CHECK: [[B:%.*]] = call target("spirv.Sampler") @llvm.spv.resource.handlefromimplicitbinding.mdname
; CHECK: call void @consume(target("spirv.Sampler") [[A]], target("spirv.Sampler") [[B]])
define void @implicit_different_names() {
  %a = call target("spirv.Sampler") @llvm.spv.resource.handlefromimplicitbinding.mdname(i32 0, i32 0, i32 1, i32 0, metadata !{!"a"})
  %b = call target("spirv.Sampler") @llvm.spv.resource.handlefromimplicitbinding.mdname(i32 0, i32 0, i32 1, i32 0, metadata !{!"b"})
  call void @consume(target("spirv.Sampler") %a, target("spirv.Sampler") %b)
  ret void
}

declare target("spirv.Sampler") @llvm.spv.resource.handlefrombinding.mdname(i32, i32, i32, i32, metadata)
declare target("spirv.Sampler") @llvm.spv.resource.handlefromimplicitbinding.mdname(i32, i32, i32, i32, metadata)
declare void @consume(target("spirv.Sampler"), target("spirv.Sampler"))

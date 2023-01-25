; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; Check support for basic target extension type usage

@global = global target("spirv.DeviceEvent") zeroinitializer

define target("spirv.Sampler") @foo(target("spirv.Sampler") %a) {
  ret target("spirv.Sampler") %a
}

define target("spirv.Event") @func2() {
  %mem = alloca target("spirv.Event")
  %val = load target("spirv.Event"), ptr %mem
  ret target("spirv.Event") poison
}

; CHECK: @global = global target("spirv.DeviceEvent") zeroinitializer
; CHECK: define target("spirv.Sampler") @foo(target("spirv.Sampler") %a) {
; CHECK:   ret target("spirv.Sampler") %a
; CHECK: }
; CHECK: define target("spirv.Event") @func2() {
; CHECK:   %mem = alloca target("spirv.Event")
; CHECK:   %val = load target("spirv.Event"), ptr %mem
; CHECK:   ret target("spirv.Event") poison
; CHECK: }

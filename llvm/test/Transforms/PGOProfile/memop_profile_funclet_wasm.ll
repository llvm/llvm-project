; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s --check-prefixes=CHECK,GEN
; RUN: opt < %s -passes=pgo-instr-gen,instrprof -S | FileCheck %s --check-prefixes=CHECK,LOWER

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define void @wasm_funclet_op_bundle(ptr %p, ptr %dst, ptr %src) personality ptr @__gxx_wasm_personality_v0 {
entry:
  invoke void @foo()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr null]
; CHECK: %[[CATCHPAD:.*]] = catchpad
  %2 = call ptr @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call ptr @__cxa_begin_catch(ptr %2) #3 [ "funclet"(token %1) ]
  %tmp = load i32, ptr %p, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr %dst, ptr %src, i32 %tmp, i1 false)
; GEN: call void @llvm.instrprof.value.profile({{.*}}) [ "funclet"(token %[[CATCHPAD]]) ]
; LOWER: call void @__llvm_profile_instrument_memop({{.*}}) [ "funclet"(token %[[CATCHPAD]]) ]
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %catch.start, %entry
  ret void
}

declare void @foo()
declare i32 @__gxx_wasm_personality_v0(...)
; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.wasm.get.exception(token) #0
; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i32 @llvm.wasm.get.ehselector(token) #0
; Function Attrs: nounwind memory(none)
declare i32 @llvm.eh.typeid.for(ptr) #1
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg) #2

attributes #0 = { nocallback nofree nosync nounwind willreturn }
attributes #1 = { nounwind memory(none) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nounwind }

; REQUIRES: aarch64-registered-target
; RUN: split-file %s %t
; RUN: opt -verify-each -passes='thinlto-pre-link<O2>' -module-summary %t/producer.ll -o %t/producer.bc
; RUN: opt -verify-each -passes='thinlto-pre-link<O2>' -module-summary %t/consumer.ll -o %t/consumer.bc
; RUN: llvm-lto2 run -save-temps -import-instr-limit=1000 -o %t/out %t/producer.bc %t/consumer.bc \
; RUN:   -r=%t/producer.bc,prototype, -r=%t/producer.bc,malloc, -r=%t/producer.bc,free, \
; RUN:   -r=%t/producer.bc,yield,plx -r=%t/consumer.bc,yield, -r=%t/consumer.bc,entry,plx
; RUN: llvm-dis %t/out.2.4.opt.bc -o - | FileCheck %s --implicit-check-not='call ptr @llvm.coro.prepare.retcon'

; Importing the public forwarding entry need not import its anonymous body.
; The backend must expose a call to the promoted ramp declaration, leaving
; neither an unlowered intrinsic nor an incorrect undef result at codegen.
; CHECK-LABEL: define{{.*}} i32 @entry(
; CHECK: [[PAIR:%.*]] = {{.*}}call { ptr, i32 } @yield.coro.body.llvm.{{[0-9]+}}(ptr
; CHECK: [[VALUE:%.*]] = extractvalue { ptr, i32 } [[PAIR]], 1
; CHECK: ret i32 [[VALUE]]

;--- producer.ll
target triple = "aarch64-unknown-linux-gnu"
declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.end(ptr, i1, token)
declare void @llvm.trap()
declare void @prototype(ptr, i1)
declare ptr @malloc(i64)
declare void @free(ptr)

define { ptr, i32 } @yield(ptr %buffer) presplitcoroutine {
entry:
  %slot = alloca { ptr, i32 }, align 8
  %id = call token @llvm.coro.id.retcon.once(i32 16, i32 8, ptr %buffer,
      ptr @prototype, ptr @malloc, ptr @free, ptr %slot)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 7)
  call void @llvm.trap()
  unreachable

cleanup:
  call void @llvm.coro.end(ptr %hdl, i1 false, token none)
  %result = load { ptr, i32 }, ptr %slot, align 8
  ret { ptr, i32 } %result
}

;--- consumer.ll
target triple = "aarch64-unknown-linux-gnu"
declare { ptr, i32 } @yield(ptr)
define i32 @entry(ptr %buffer) {
  %pair = call { ptr, i32 } @yield(ptr %buffer)
  %value = extractvalue { ptr, i32 } %pair, 1
  ret i32 %value
}

; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that the wasm-store-results pass makes users of stored values use the
; result of store expressions to reduce get_local/set_local traffic.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: single_block:
; CHECK-NOT: .local
; CHECK: i32.const $push{{[0-9]+}}=, 0{{$}}
; CHECK: i32.store $push[[STORE:[0-9]+]]=, 0($0), $pop{{[0-9]+}}{{$}}
; CHECK: return $pop[[STORE]]{{$}}
define i32 @single_block(i32* %p) {
entry:
  store i32 0, i32* %p
  ret i32 0
}

; Test interesting corner cases for wasm-store-results, in which the operand of
; a store ends up getting used by a phi, which needs special handling in the
; dominance test, since phis use their operands on their incoming edges.

%class.Vec3 = type { float, float, float }

@pos = global %class.Vec3 zeroinitializer, align 4

; CHECK-LABEL: foo:
; CHECK: i32.store $discard=, pos($0), $0{{$}}
define void @foo() {
for.body.i:
  br label %for.body5.i

for.body5.i:
  %i.0168.i = phi i32 [ 0, %for.body.i ], [ %inc.i, %for.body5.i ]
  %conv6.i = sitofp i32 %i.0168.i to float
  store volatile float 0.0, float* getelementptr inbounds (%class.Vec3, %class.Vec3* @pos, i32 0, i32 0)
  %inc.i = add nuw nsw i32 %i.0168.i, 1
  %exitcond.i = icmp eq i32 %inc.i, 256
  br i1 %exitcond.i, label %for.cond.cleanup4.i, label %for.body5.i

for.cond.cleanup4.i:
  ret void
}

; CHECK-LABEL: bar:
; CHECK: i32.store $discard=, pos($0), $0{{$}}
define void @bar() {
for.body.i:
  br label %for.body5.i

for.body5.i:
  %i.0168.i = phi float [ 0.0, %for.body.i ], [ %inc.i, %for.body5.i ]
  store volatile float 0.0, float* getelementptr inbounds (%class.Vec3, %class.Vec3* @pos, i32 0, i32 0)
  %inc.i = fadd float %i.0168.i, 1.0
  %exitcond.i = fcmp oeq float %inc.i, 256.0
  br i1 %exitcond.i, label %for.cond.cleanup4.i, label %for.body5.i

for.cond.cleanup4.i:
  ret void
}

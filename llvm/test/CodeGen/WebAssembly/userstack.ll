; RUN: llc < %s -asm-verbose=false | FileCheck %s
; RUN: llc < %s -asm-verbose=false -fast-isel | FileCheck %s


target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: alloca32:
; Check that there is an extra local for the stack pointer.
; CHECK: .local i32, i32, i32, i32{{$}}
define void @alloca32() {
 ; CHECK: i32.const [[L1:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.load [[L1]]=, 0([[L1]])
 ; CHECK-NEXT: i32.const [[L2:.+]]=, 16
 ; CHECK-NEXT: i32.sub [[SP:.+]]=, [[L1]], [[L2]]
 %retval = alloca i32
 ; CHECK: i32.const $push[[L3:.+]]=, 0
 ; CHECK: i32.store {{.*}}=, 12([[SP]]), $pop[[L3]]
 store i32 0, i32* %retval
 ; CHECK: i32.const [[L4:.+]]=, 16
 ; CHECK-NEXT: i32.add [[SP]]=, [[SP]], [[L4]]
 ; CHECK-NEXT: i32.const [[L5:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.store [[SP]]=, 0([[L5]]), [[SP]]
 ret void
}

; CHECK-LABEL: alloca3264:
; CHECK: .local i32, i32, i32, i32{{$}}
define void @alloca3264() {
 ; CHECK: i32.const [[L1:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.load [[L1]]=, 0([[L1]])
 ; CHECK-NEXT: i32.const [[L2:.+]]=, 16
 ; CHECK-NEXT: i32.sub [[SP:.+]]=, [[L1]], [[L2]]
 %r1 = alloca i32
 %r2 = alloca double
 ; CHECK: i32.const $push[[L3:.+]]=, 0
 ; CHECK: i32.store {{.*}}=, 12([[SP]]), $pop[[L3]]
 store i32 0, i32* %r1
 ; CHECK: i64.const $push[[L4:.+]]=, 0
 ; CHECK: i64.store {{.*}}=, 0([[SP]]), $pop[[L4]]
 store double 0.0, double* %r2
 ; CHECK: i32.const [[L4:.+]]=, 16
 ; CHECK-NEXT: i32.add [[SP]]=, [[SP]], [[L4]]
 ; CHECK-NEXT: i32.const [[L5:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.store [[SP]]=, 0([[L5]]), [[SP]]
 ret void
}

; CHECK-LABEL: allocarray:
; CHECK: .local i32, i32, i32, i32, i32{{$}}
define void @allocarray() {
 ; CHECK-NEXT: i32.const [[L1:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.load [[L1]]=, 0([[L1]])
 ; CHECK-NEXT: i32.const [[L2:.+]]=, 32{{$}}
 ; CHECK-NEXT: i32.sub [[SP:.+]]=, [[L1]], [[L2]]
 ; CHECK-NEXT: i32.const [[L2]]=, __stack_pointer{{$}}
 ; CHECK-NEXT: i32.store [[SP]]=, 0([[L2]]), [[SP]]
 %r = alloca [5 x i32]

 ; CHECK-NEXT: i32.const $push[[L4:.+]]=, 12
 ; CHECK-NEXT: i32.const [[L5:.+]]=, 12
 ; CHECK-NEXT: i32.add [[L5]]=, [[SP]], [[L5]]
 ; CHECK-NEXT: i32.add $push[[L6:.+]]=, [[L5]], $pop[[L4]]
 ; CHECK-NEXT: i32.const $push[[L9:.+]]=, 1{{$}}
 ; CHECK-NEXT: i32.store $push[[L10:.+]]=, 12([[SP]]), $pop[[L9]]{{$}}
 ; CHECK-NEXT: i32.store $discard=, 0($pop3), $pop[[L10]]{{$}}
 %p = getelementptr [5 x i32], [5 x i32]* %r, i32 0, i32 0
 store i32 1, i32* %p
 %p2 = getelementptr [5 x i32], [5 x i32]* %r, i32 0, i32 3
 store i32 1, i32* %p2

 ; CHECK-NEXT: i32.const [[L7:.+]]=, 32
 ; CHECK-NEXT: i32.add [[SP]]=, [[SP]], [[L7]]
 ; CHECK-NEXT: i32.const [[L8:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.store [[SP]]=, 0([[L8]]), [[SP]]
 ret void
}

; CHECK-LABEL: allocarray_inbounds:
; CHECK: .local i32, i32, i32, i32{{$}}
define void @allocarray_inbounds() {
 ; CHECK: i32.const [[L1:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.load [[L1]]=, 0([[L1]])
 ; CHECK-NEXT: i32.const [[L2:.+]]=, 32
 ; CHECK-NEXT: i32.sub [[SP:.+]]=, [[L1]], [[L2]]
 %r = alloca [5 x i32]
 ; CHECK: i32.const $push[[L3:.+]]=, 1
 ; CHECK: i32.store {{.*}}=, 12([[SP]]), $pop[[L3]]
 %p = getelementptr inbounds [5 x i32], [5 x i32]* %r, i32 0, i32 0
 store i32 1, i32* %p
 ; This store should have both the GEP and the FI folded into it.
 ; CHECK-NEXT: i32.store {{.*}}=, 24([[SP]]), $pop
 %p2 = getelementptr inbounds [5 x i32], [5 x i32]* %r, i32 0, i32 3
 store i32 1, i32* %p2
 ; CHECK: i32.const [[L7:.+]]=, 32
 ; CHECK-NEXT: i32.add [[SP]]=, [[SP]], [[L7]]
 ; CHECK-NEXT: i32.const [[L8:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.store [[SP]]=, 0([[L7]]), [[SP]]
 ret void
}

; CHECK-LABEL: dynamic_alloca:
define void @dynamic_alloca(i32 %alloc) {
 ; TODO: Support frame pointers
 ;%r = alloca i32, i32 %alloc
 ;store i32 0, i32* %r
 ret void
}
; TODO: test aligned alloc

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

declare void @ext_func(i64* %ptr)
; CHECK-LABEL: non_mem_use
define void @non_mem_use(i8** %addr) {
 ; CHECK: i32.const [[L2:.+]]=, 48
 ; CHECK-NEXT: i32.sub [[SP:.+]]=, {{.+}}, [[L2]]
 %buf = alloca [27 x i8], align 16
 %r = alloca i64
 %r2 = alloca i64
 ; %r is at SP+8
 ; CHECK: i32.const [[OFF:.+]]=, 8
 ; CHECK-NEXT: i32.add [[ARG1:.+]]=, [[SP]], [[OFF]]
 ; CHECK-NEXT: call ext_func@FUNCTION, [[ARG1]]
 call void @ext_func(i64* %r)
 ; %r2 is at SP+0, no add needed
 ; CHECK-NEXT: call ext_func@FUNCTION, [[SP]]
 call void @ext_func(i64* %r2)
 ; Use as a value, but in a store
 ; %buf is at SP+16
 ; CHECK: i32.const [[OFF:.+]]=, 16
 ; CHECK-NEXT: i32.add [[VAL:.+]]=, [[SP]], [[OFF]]
 ; CHECK-NEXT: i32.store {{.*}}=, 0($0), [[VAL]]
 %gep = getelementptr inbounds [27 x i8], [27 x i8]* %buf, i32 0, i32 0
 store i8* %gep, i8** %addr
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
 ; CHECK: i32.const [[L0:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.load [[SP:.+]]=, 0([[L0]])
 ; CHECK-NEXT: copy_local [[FP:.+]]=, [[SP]]
 ; Target independent codegen bumps the stack pointer
 ; FIXME: we need to write the value back to memory
 %r = alloca i32, i32 %alloc
 ; Target-independent codegen also calculates the store addr
 store i32 0, i32* %r
 ; CHECK: i32.const [[L3:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.store [[SP]]=, 0([[L3]]), [[FP]]
 ret void
}


; CHECK-LABEL: dynamic_static_alloca:
define void @dynamic_static_alloca(i32 %alloc) {
 ; CHECK: i32.const [[L0:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.load [[L0]]=, 0([[L0]])
 ; CHECK-NEXT: i32.const [[L2:.+]]=, 16
 ; CHECK-NEXT: i32.sub [[SP:.+]]=, [[L0]], [[L2]]
 ; CHECK-NEXT: copy_local [[FP:.+]]=, [[SP]]
 ; CHECK-NEXT: i32.const [[L3:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.store {{.*}}=, 0([[L3]]), [[SP]]
 %r1 = alloca i32
 %r = alloca i32, i32 %alloc
 store i32 0, i32* %r
 ; CHECK: i32.const [[L3:.+]]=, 16
 ; CHECK: i32.add [[SP]]=, [[FP]], [[L3]]
 ; CHECK: i32.const [[L4:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.store [[SP]]=, 0([[L4]]), [[SP]]
 ret void
}

; TODO: test over-aligned alloca

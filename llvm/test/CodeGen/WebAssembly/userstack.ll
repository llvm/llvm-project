; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

declare void @ext_func(i64* %ptr)
declare void @ext_func_i32(i32* %ptr)

; CHECK-LABEL: alloca32:
; Check that there is an extra local for the stack pointer.
; CHECK: .local i32{{$}}
define void @alloca32() noredzone {
 ; CHECK-NEXT: get_global $push[[L2:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.const $push[[L3:.+]]=, 16
 ; CHECK-NEXT: i32.sub $push[[L9:.+]]=, $pop[[L2]], $pop[[L3]]
 ; CHECK-NEXT: tee_local $push[[L8:.+]]=, [[SP:.+]], $pop[[L9]]{{$}}
 ; CHECK-NEXT: set_global 0, $pop[[L8]]{{$}}
 %retval = alloca i32
 ; CHECK: get_local $push[[L4:.+]]=, [[SP]]{{$}}
 ; CHECK: i32.const $push[[L0:.+]]=, 0
 ; CHECK: i32.store 12($pop[[L4]]), $pop[[L0]]
 store i32 0, i32* %retval
 ; CHECK: get_local $push[[L6:.+]]=, [[SP]]{{$}}
 ; CHECK-NEXT: i32.const $push[[L5:.+]]=, 16
 ; CHECK-NEXT: i32.add $push[[L7:.+]]=, $pop[[L6]], $pop[[L5]]
 ; CHECK-NEXT: set_global 0, $pop[[L7]]
 ret void
}

; CHECK-LABEL: alloca3264:
; CHECK: .local i32{{$}}
define void @alloca3264() {
 ; CHECK: get_global $push[[L3:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.const $push[[L4:.+]]=, 16
 ; CHECK-NEXT: i32.sub $push[[L6:.+]]=, $pop[[L3]], $pop[[L4]]
 ; CHECK-NEXT: tee_local $push[[L5:.+]]=, [[SP:.+]], $pop[[L6]]
 %r1 = alloca i32
 %r2 = alloca double
 ; CHECK-NEXT: i32.const $push[[L0:.+]]=, 0
 ; CHECK-NEXT: i32.store 12($pop[[L5]]), $pop[[L0]]
 store i32 0, i32* %r1
 ; CHECK-NEXT: get_local $push[[L2:.+]]=, [[SP]]{{$}}
 ; CHECK-NEXT: i64.const $push[[L1:.+]]=, 0
 ; CHECK-NEXT: i64.store 0($pop[[L2]]), $pop[[L1]]
 store double 0.0, double* %r2
 ; CHECK-NEXT: return
 ret void
}

; CHECK-LABEL: allocarray:
; CHECK: .local i32{{$}}
define void @allocarray() {
 ; CHECK-NEXT: get_global $push[[L4:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.const $push[[L5:.+]]=, 144{{$}}
 ; CHECK-NEXT: i32.sub $push[[L12:.+]]=, $pop[[L4]], $pop[[L5]]
 ; CHECK-NEXT: tee_local $push[[L11:.+]]=, 0, $pop[[L12]]
 ; CHECK-NEXT: set_global 0, $pop[[L11]]
 %r = alloca [33 x i32]

 ; CHECK:      i32.const $push{{.+}}=, 24
 ; CHECK-NEXT: i32.add $push[[L3:.+]]=, $pop{{.+}}, $pop{{.+}}
 ; CHECK-NEXT: i32.const $push[[L1:.+]]=, 1{{$}}
 ; CHECK-NEXT: i32.store 0($pop[[L3]]), $pop[[L1]]{{$}}
 ; CHECK-NEXT: get_local $push[[L4:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.const $push[[L10:.+]]=, 1{{$}}
 ; CHECK-NEXT: i32.store 12($pop[[L4]]), $pop[[L10]]{{$}}
 %p = getelementptr [33 x i32], [33 x i32]* %r, i32 0, i32 0
 store i32 1, i32* %p
 %p2 = getelementptr [33 x i32], [33 x i32]* %r, i32 0, i32 3
 store i32 1, i32* %p2

 ; CHECK-NEXT: get_local $push[[L2:.+]]=, [[SP]]{{$}}
 ; CHECK-NEXT: i32.const $push[[L7:.+]]=, 144
 ; CHECK-NEXT: i32.add $push[[L8:.+]]=, $pop[[L2]], $pop[[L7]]
 ; CHECK-NEXT: set_global 0, $pop[[L8]]
 ret void
}

; CHECK-LABEL: non_mem_use
define void @non_mem_use(i8** %addr) {
 ; CHECK: i32.const $push[[L2:.+]]=, 48
 ; CHECK-NEXT: i32.sub $push[[L12:.+]]=, {{.+}}, $pop[[L2]]
 ; CHECK-NEXT: tee_local $push[[L11:.+]]=, [[SP:.+]], $pop[[L12]]
 ; CHECK-NEXT: set_global 0, $pop[[L11]]
 %buf = alloca [27 x i8], align 16
 %r = alloca i64
 %r2 = alloca i64
 ; %r is at SP+8
 ; CHECK: get_local $push[[L3:.+]]=, [[SP]]
 ; CHECK: i32.const $push[[OFF:.+]]=, 8
 ; CHECK-NEXT: i32.add $push[[ARG1:.+]]=, $pop[[L3]], $pop[[OFF]]
 ; CHECK-NEXT: call ext_func@FUNCTION, $pop[[ARG1]]
 call void @ext_func(i64* %r)
 ; %r2 is at SP+0, no add needed
 ; CHECK: get_local $push[[L4:.+]]=, [[SP]]
 ; CHECK-NEXT: call ext_func@FUNCTION, $pop[[L4]]
 call void @ext_func(i64* %r2)
 ; Use as a value, but in a store
 ; %buf is at SP+16
 ; CHECK: get_local $push[[L5:.+]]=, [[SP]]
 ; CHECK: i32.const $push[[OFF:.+]]=, 16
 ; CHECK-NEXT: i32.add $push[[VAL:.+]]=, $pop[[L5]], $pop[[OFF]]
 ; CHECK-NEXT: i32.store 0($pop{{.+}}), $pop[[VAL]]
 %gep = getelementptr inbounds [27 x i8], [27 x i8]* %buf, i32 0, i32 0
 store i8* %gep, i8** %addr
 ret void
}

; CHECK-LABEL: allocarray_inbounds:
; CHECK: .local i32{{$}}
define void @allocarray_inbounds() {
 ; CHECK: get_global $push[[L3:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.const $push[[L4:.+]]=, 32{{$}}
 ; CHECK-NEXT: i32.sub $push[[L11:.+]]=, $pop[[L3]], $pop[[L4]]
 ; CHECK-NEXT: tee_local $push[[L10:.+]]=, [[SP:.+]], $pop[[L11]]
 ; CHECK-NEXT: set_global 0, $pop[[L10]]{{$}}
 %r = alloca [5 x i32]
 ; CHECK: i32.const $push[[L3:.+]]=, 1
 ; CHECK-DAG: i32.store 24(${{.+}}), $pop[[L3]]
 %p = getelementptr inbounds [5 x i32], [5 x i32]* %r, i32 0, i32 0
 store i32 1, i32* %p
 ; This store should have both the GEP and the FI folded into it.
 ; CHECK-DAG: i32.store 12(${{.+}}), $pop
 %p2 = getelementptr inbounds [5 x i32], [5 x i32]* %r, i32 0, i32 3
 store i32 1, i32* %p2
 call void @ext_func(i64* null);
 ; CHECK: call ext_func
 ; CHECK: i32.const $push[[L5:.+]]=, 32{{$}}
 ; CHECK-NEXT: i32.add $push[[L7:.+]]=, ${{.+}}, $pop[[L5]]
 ; CHECK-NEXT: set_global 0, $pop[[L7]]
 ret void
}

; CHECK-LABEL: dynamic_alloca:
define void @dynamic_alloca(i32 %alloc) {
 ; CHECK: get_global $push[[L13:.+]]=, 0{{$}}
 ; CHECK-NEXT: tee_local $push[[L12:.+]]=, [[SP:.+]], $pop[[L13]]{{$}}
 ; Target independent codegen bumps the stack pointer.
 ; CHECK: i32.sub
 ; Check that SP is written back to memory after decrement
 ; CHECK: set_global 0, 
 %r = alloca i32, i32 %alloc
 ; Target-independent codegen also calculates the store addr
 ; CHECK: call ext_func_i32@FUNCTION
 call void @ext_func_i32(i32* %r)
 ; CHECK: set_global 0, $pop{{.+}}
 ret void
}

; CHECK-LABEL: dynamic_alloca_redzone:
define void @dynamic_alloca_redzone(i32 %alloc) {
 ; CHECK: get_global $push[[L13:.+]]=, 0{{$}}
 ; CHECK-NEXT: tee_local $push[[L12:.+]]=, [[SP:.+]], $pop[[L13]]{{$}}
 ; Target independent codegen bumps the stack pointer
 ; CHECK: i32.sub
 %r = alloca i32, i32 %alloc
 ; CHECK-NEXT: tee_local       $push[[L8:.+]]=, {{.+}}, $pop
 ; CHECK: get_local $push[[L7:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.const       $push[[L6:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.store       0($pop[[L7]]), $pop[[L6]]{{$}}
 store i32 0, i32* %r
 ; CHECK-NEXT: return
 ret void
}

; CHECK-LABEL: dynamic_static_alloca:
define void @dynamic_static_alloca(i32 %alloc) noredzone {
 ; Decrement SP in the prolog by the static amount and writeback to memory.
 ; CHECK: get_global $push[[L11:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.const $push[[L12:.+]]=, 16
 ; CHECK-NEXT: i32.sub $push[[L23:.+]]=, $pop[[L11]], $pop[[L12]]
 ; CHECK-NEXT: tee_local $push[[L22:.+]]=, [[SP:.+]], $pop[[L23]]
 ; CHECK-NEXT: set_global 0, $pop[[L22]]

 ; Alloc and write to a static alloca
 ; CHECK: get_local $push[[L21:.+]]=, [[SP:.+]]
 ; CHECK-NEXT: tee_local $push[[pushedFP:.+]]=, [[FP:.+]], $pop[[L21]]
 ; CHECK-NEXT: i32.const $push[[L0:.+]]=, 101
 ; CHECK-NEXT: i32.store [[static_offset:.+]]($pop[[pushedFP]]), $pop[[L0]]
 %static = alloca i32
 store volatile i32 101, i32* %static

 ; Decrement SP in the body by the dynamic amount.
 ; CHECK: i32.sub
 ; CHECK: tee_local $push[[L16:.+]]=, [[dynamic_local:.+]], $pop{{.+}}
 ; CHECK: tee_local $push[[L15:.+]]=, [[other:.+]], $pop[[L16]]{{$}}
 ; CHECK: set_global 0, $pop[[L15]]{{$}}
 %dynamic = alloca i32, i32 %alloc

 ; Ensure we don't modify the frame pointer after assigning it.
 ; CHECK-NOT: $[[FP]]=

 ; Ensure the static address doesn't change after modifying the stack pointer.
 ; CHECK: get_local $push[[L17:.+]]=, [[FP]]
 ; CHECK: i32.const $push[[L7:.+]]=, 102
 ; CHECK-NEXT: i32.store [[static_offset]]($pop[[L17]]), $pop[[L7]]
 ; CHECK-NEXT: get_local $push[[L9:.+]]=, [[dynamic_local]]{{$}}
 ; CHECK-NEXT: i32.const $push[[L8:.+]]=, 103
 ; CHECK-NEXT: i32.store 0($pop[[L9]]), $pop[[L8]]
 store volatile i32 102, i32* %static
 store volatile i32 103, i32* %dynamic

 ; Decrement SP in the body by the dynamic amount.
 ; CHECK: i32.sub
 ; CHECK: tee_local $push{{.+}}=, [[dynamic2_local:.+]], $pop{{.+}}
 %dynamic.2 = alloca i32, i32 %alloc

 ; CHECK-NOT: $[[FP]]=

 ; Ensure neither the static nor dynamic address changes after the second
 ; modification of the stack pointer.
 ; CHECK: get_local $push[[L22:.+]]=, [[FP]]
 ; CHECK: i32.const $push[[L9:.+]]=, 104
 ; CHECK-NEXT: i32.store [[static_offset]]($pop[[L22]]), $pop[[L9]]
 ; CHECK-NEXT: get_local $push[[L23:.+]]=, [[dynamic_local]]
 ; CHECK-NEXT: i32.const $push[[L10:.+]]=, 105
 ; CHECK-NEXT: i32.store 0($pop[[L23]]), $pop[[L10]]
 ; CHECK-NEXT: get_local $push[[L23:.+]]=, [[dynamic2_local]]
 ; CHECK-NEXT: i32.const $push[[L11:.+]]=, 106
 ; CHECK-NEXT: i32.store 0($pop[[L23]]), $pop[[L11]]
 store volatile i32 104, i32* %static
 store volatile i32 105, i32* %dynamic
 store volatile i32 106, i32* %dynamic.2

 ; Writeback to memory.
 ; CHECK: get_local $push[[L24:.+]]=, [[FP]]{{$}}
 ; CHECK: i32.const $push[[L18:.+]]=, 16
 ; CHECK-NEXT: i32.add $push[[L19:.+]]=, $pop[[L24]], $pop[[L18]]
 ; CHECK-NEXT: set_global 0, $pop[[L19]]
 ret void
}

declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)

; CHECK-LABEL: llvm_stack_builtins:
define void @llvm_stack_builtins(i32 %alloc) noredzone {
 ; CHECK: get_global $push[[L11:.+]]=, 0{{$}}
 ; CHECK-NEXT: tee_local $push[[L10:.+]]=, {{.+}}, $pop[[L11]]
 ; CHECK-NEXT: set_local [[STACK:.+]], $pop[[L10]]
 %stack = call i8* @llvm.stacksave()

 ; Ensure we don't reassign the stacksave local
 ; CHECK-NOT: set_local [[STACK]],
 %dynamic = alloca i32, i32 %alloc

 ; CHECK: get_local $push[[L12:.+]]=, [[STACK]]
 ; CHECK-NEXT: set_global 0, $pop[[L12]]
 call void @llvm.stackrestore(i8* %stack)

 ret void
}

; Not actually using the alloca'd variables exposed an issue with register
; stackification, where copying the stack pointer into the frame pointer was
; moved after the stack pointer was updated for the dynamic alloca.
; CHECK-LABEL: dynamic_alloca_nouse:
define void @dynamic_alloca_nouse(i32 %alloc) noredzone {
 ; CHECK: get_global $push[[L11:.+]]=, 0{{$}}
 ; CHECK-NEXT: tee_local $push[[L10:.+]]=, {{.+}}, $pop[[L11]]
 ; CHECK-NEXT: set_local [[FP:.+]], $pop[[L10]]
 %dynamic = alloca i32, i32 %alloc

 ; CHECK-NOT: set_local [[FP]],

 ; CHECK: get_local $push[[L12:.+]]=, [[FP]]
 ; CHECK-NEXT: set_global 0, $pop[[L12]]
 ret void
}

; The use of the alloca in a phi causes a CopyToReg DAG node to be generated,
; which has to have special handling because CopyToReg can't have a FI operand
; CHECK-LABEL: copytoreg_fi:
define void @copytoreg_fi(i1 %cond, i32* %b) {
entry:
 ; CHECK: i32.const $push[[L1:.+]]=, 16
 ; CHECK-NEXT: i32.sub $push[[L3:.+]]=, {{.+}}, $pop[[L1]]
 %addr = alloca i32
 ; CHECK: i32.const $push[[OFF:.+]]=, 12
 ; CHECK-NEXT: i32.add $push[[ADDR:.+]]=, $pop[[L3]], $pop[[OFF]]
 ; CHECK-NEXT: set_local [[COPY:.+]], $pop[[ADDR]]
 br label %body
body:
 %a = phi i32* [%addr, %entry], [%b, %body]
 store i32 1, i32* %a
 ; CHECK: get_local $push[[L12:.+]]=, [[COPY]]
 ; CHECK: i32.store 0($pop[[L12]]),
 br i1 %cond, label %body, label %exit
exit:
 ret void
}

declare void @use_i8_star(i8*)
declare i8* @llvm.frameaddress(i32)

; Test __builtin_frame_address(0).
; CHECK-LABEL: frameaddress_0:
; CHECK: get_global $push[[L3:.+]]=, 0{{$}}
; CHECK-NEXT: tee_local $push[[L2:.+]]=, [[FP:.+]], $pop[[L3]]{{$}}
; CHECK-NEXT: call use_i8_star@FUNCTION, $pop[[L2]]
; CHECK-NEXT: get_local $push[[L5:.+]]=, [[FP]]
; CHECK-NEXT: set_global 0, $pop[[L5]]
define void @frameaddress_0() {
  %t = call i8* @llvm.frameaddress(i32 0)
  call void @use_i8_star(i8* %t)
  ret void
}

; Test __builtin_frame_address(1).

; CHECK-LABEL: frameaddress_1:
; CHECK-NEXT: i32.const $push0=, 0{{$}}
; CHECK-NEXT: call use_i8_star@FUNCTION, $pop0{{$}}
; CHECK-NEXT: return{{$}}
define void @frameaddress_1() {
  %t = call i8* @llvm.frameaddress(i32 1)
  call void @use_i8_star(i8* %t)
  ret void
}

; Test a stack address passed to an inline asm.
; CHECK-LABEL: inline_asm:
; CHECK:       get_global {{.+}}, 0{{$}}
; CHECK:       #APP
; CHECK-NEXT:  # %{{[0-9]+}}{{$}}
; CHECK-NEXT:  #NO_APP
define void @inline_asm() {
  %tmp = alloca i8
  call void asm sideeffect "# %0", "r"(i8* %tmp)
  ret void
}

; TODO: test over-aligned alloca

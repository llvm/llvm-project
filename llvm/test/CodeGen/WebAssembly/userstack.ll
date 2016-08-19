; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare void @ext_func(i64* %ptr)
declare void @ext_func_i32(i32* %ptr)

; CHECK-LABEL: alloca32:
; Check that there is an extra local for the stack pointer.
; CHECK: .local i32{{$}}
define void @alloca32() noredzone {
 ; CHECK: i32.const $push[[L4:.+]]=, 0{{$}}
 ; CHECK: i32.const $push[[L1:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.load $push[[L2:.+]]=, __stack_pointer($pop[[L1]])
 ; CHECK-NEXT: i32.const $push[[L3:.+]]=, 16
 ; CHECK-NEXT: i32.sub $push[[L9:.+]]=, $pop[[L2]], $pop[[L3]]
 ; CHECK-NEXT: tee_local $push[[L8:.+]]=, $[[SP:.+]]=, $pop[[L9]]{{$}}
 ; CHECK-NEXT: i32.store $drop=, __stack_pointer($pop[[L4]]), $pop[[L8]]{{$}}
 %retval = alloca i32
 ; CHECK: i32.const $push[[L0:.+]]=, 0
 ; CHECK: i32.store $drop=, 12($[[SP]]), $pop[[L0]]
 store i32 0, i32* %retval
 ; CHECK: i32.const $push[[L6:.+]]=, 0
 ; CHECK-NEXT: i32.const $push[[L5:.+]]=, 16
 ; CHECK-NEXT: i32.add $push[[L7:.+]]=, $[[SP]], $pop[[L5]]
 ; CHECK-NEXT: i32.store $drop=, __stack_pointer($pop[[L6]]), $pop[[L7]]
 ret void
}

; CHECK-LABEL: alloca3264:
; CHECK: .local i32{{$}}
define void @alloca3264() {
 ; CHECK: i32.const $push[[L2:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.load $push[[L3:.+]]=, __stack_pointer($pop[[L2]])
 ; CHECK-NEXT: i32.const $push[[L4:.+]]=, 16
 ; CHECK-NEXT: i32.sub $push[[L6:.+]]=, $pop[[L3]], $pop[[L4]]
 ; CHECK-NEXT: tee_local $push[[L5:.+]]=, $[[SP:.+]]=, $pop[[L6]]
 %r1 = alloca i32
 %r2 = alloca double
 ; CHECK-NEXT: i32.const $push[[L0:.+]]=, 0
 ; CHECK-NEXT: i32.store $drop=, 12($pop[[L5]]), $pop[[L0]]
 store i32 0, i32* %r1
 ; CHECK-NEXT: i64.const $push[[L1:.+]]=, 0
 ; CHECK-NEXT: i64.store $drop=, 0($[[SP]]), $pop[[L1]]
 store double 0.0, double* %r2
 ; CHECK-NEXT: return
 ret void
}

; CHECK-LABEL: allocarray:
; CHECK: .local i32{{$}}
define void @allocarray() {
 ; CHECK: i32.const $push[[L6:.+]]=, 0{{$}}
 ; CHECK: i32.const $push[[L3:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.load $push[[L4:.+]]=, __stack_pointer($pop[[L3]])
 ; CHECK-NEXT: i32.const $push[[L5:.+]]=, 144{{$}}
 ; CHECK-NEXT: i32.sub $push[[L12:.+]]=, $pop[[L4]], $pop[[L5]]
 ; CHECK-NEXT: tee_local $push[[L11:.+]]=, $0=, $pop[[L12]]
 ; CHECK-NEXT: i32.store $drop=, __stack_pointer($pop[[L6]]), $pop[[L11]]
 %r = alloca [33 x i32]

 ; CHECK:      i32.const $push{{.+}}=, 24
 ; CHECK-NEXT: i32.add $push[[L3:.+]]=, $[[SP]], $pop{{.+}}
 ; CHECK-NEXT: i32.const $push[[L1:.+]]=, 1{{$}}
 ; CHECK-NEXT: i32.store $drop=, 0($pop[[L3]]), $pop[[L1]]{{$}}
 ; CHECK-NEXT: i32.const $push[[L10:.+]]=, 1{{$}}
 ; CHECK-NEXT: i32.store $drop=, 12(${{.+}}), $pop[[L10]]{{$}}
 %p = getelementptr [33 x i32], [33 x i32]* %r, i32 0, i32 0
 store i32 1, i32* %p
 %p2 = getelementptr [33 x i32], [33 x i32]* %r, i32 0, i32 3
 store i32 1, i32* %p2

 ; CHECK: i32.const $push[[L9:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.const $push[[L7:.+]]=, 144
 ; CHECK-NEXT: i32.add $push[[L8:.+]]=, $[[SP]], $pop[[L7]]
 ; CHECK-NEXT: i32.store $drop=, __stack_pointer($pop[[L9]]), $pop[[L8]]
 ret void
}

; CHECK-LABEL: non_mem_use
define void @non_mem_use(i8** %addr) {
 ; CHECK: i32.const $push[[L2:.+]]=, 48
 ; CHECK-NEXT: i32.sub $push[[L12:.+]]=, {{.+}}, $pop[[L2]]
 ; CHECK-NEXT: tee_local $push[[L11:.+]]=, $[[SP:.+]]=, $pop[[L12]]
 ; CHECK-NEXT: i32.store $drop=, {{.+}}, $pop[[L11]]
 %buf = alloca [27 x i8], align 16
 %r = alloca i64
 %r2 = alloca i64
 ; %r is at SP+8
 ; CHECK: i32.const $push[[OFF:.+]]=, 8
 ; CHECK-NEXT: i32.add $push[[ARG1:.+]]=, $[[SP]], $pop[[OFF]]
 ; CHECK-NEXT: call ext_func@FUNCTION, $pop[[ARG1]]
 call void @ext_func(i64* %r)
 ; %r2 is at SP+0, no add needed
 ; CHECK-NEXT: call ext_func@FUNCTION, $[[SP]]
 call void @ext_func(i64* %r2)
 ; Use as a value, but in a store
 ; %buf is at SP+16
 ; CHECK: i32.const $push[[OFF:.+]]=, 16
 ; CHECK-NEXT: i32.add $push[[VAL:.+]]=, $[[SP]], $pop[[OFF]]
 ; CHECK-NEXT: i32.store $drop=, 0($0), $pop[[VAL]]
 %gep = getelementptr inbounds [27 x i8], [27 x i8]* %buf, i32 0, i32 0
 store i8* %gep, i8** %addr
 ret void
}

; CHECK-LABEL: allocarray_inbounds:
; CHECK: .local i32{{$}}
define void @allocarray_inbounds() {
 ; CHECK: i32.const $push[[L5:.+]]=, 0{{$}}
 ; CHECK: i32.const $push[[L2:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.load $push[[L3:.+]]=, __stack_pointer($pop[[L2]])
 ; CHECK-NEXT: i32.const $push[[L4:.+]]=, 32{{$}}
 ; CHECK-NEXT: i32.sub $push[[L11:.+]]=, $pop[[L3]], $pop[[L4]]
 ; CHECK-NEXT: tee_local $push[[L10:.+]]=, $[[SP:.+]]=, $pop[[L11]]
 ; CHECK-NEXT: i32.store $drop=, __stack_pointer($pop[[L5]]), $pop[[L10]]{{$}}
 %r = alloca [5 x i32]
 ; CHECK: i32.const $push[[L3:.+]]=, 1
 ; CHECK-DAG: i32.store $drop=, 24(${{.+}}), $pop[[L3]]
 %p = getelementptr inbounds [5 x i32], [5 x i32]* %r, i32 0, i32 0
 store i32 1, i32* %p
 ; This store should have both the GEP and the FI folded into it.
 ; CHECK-DAG: i32.store {{.*}}=, 12(${{.+}}), $pop
 %p2 = getelementptr inbounds [5 x i32], [5 x i32]* %r, i32 0, i32 3
 store i32 1, i32* %p2
 call void @ext_func(i64* null);
 ; CHECK: call ext_func
 ; CHECK: i32.const $push[[L6:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.const $push[[L5:.+]]=, 32{{$}}
 ; CHECK-NEXT: i32.add $push[[L7:.+]]=, ${{.+}}, $pop[[L5]]
 ; CHECK-NEXT: i32.store $drop=, __stack_pointer($pop[[L6]]), $pop[[L7]]
 ret void
}

; CHECK-LABEL: dynamic_alloca:
define void @dynamic_alloca(i32 %alloc) {
 ; CHECK: i32.const $push[[L7:.+]]=, 0{{$}}
 ; CHECK: i32.const $push[[L1:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.load $push[[L13:.+]]=, __stack_pointer($pop[[L1]])
 ; CHECK-NEXT: tee_local $push[[L12:.+]]=, [[SP:.+]], $pop[[L13]]{{$}}
 ; Target independent codegen bumps the stack pointer.
 ; CHECK: i32.sub
 ; Check that SP is written back to memory after decrement
 ; CHECK: i32.store $drop=, __stack_pointer($pop{{.+}}), 
 %r = alloca i32, i32 %alloc
 ; Target-independent codegen also calculates the store addr
 ; CHECK: call ext_func_i32@FUNCTION
 call void @ext_func_i32(i32* %r)
 ; CHECK: i32.const $push[[L3:.+]]=, 0{{$}}
 ; CHECK: i32.store $drop=, __stack_pointer($pop[[L3]]), $pop{{.+}}
 ret void
}

; CHECK-LABEL: dynamic_alloca_redzone:
define void @dynamic_alloca_redzone(i32 %alloc) {
 ; CHECK: i32.const $push[[L8:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.load $push[[L13:.+]]=, __stack_pointer($pop[[L1]])
 ; CHECK-NEXT: tee_local $push[[L12:.+]]=, [[SP:.+]], $pop[[L13]]{{$}}
 ; CHECK-NEXT: copy_local [[FP:.+]]=, $pop[[L12]]{{$}}
 ; Target independent codegen bumps the stack pointer
 ; CHECK: i32.sub
 %r = alloca i32, i32 %alloc
 ; CHECK-NEXT: tee_local       $push[[L8:.+]]=, $0=, $pop
 ; CHECK-NEXT: copy_local      $drop=, $pop[[L8]]{{$}}
 ; CHECK-NEXT: i32.const       $push[[L6:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.store       $drop=, 0($0), $pop[[L6]]{{$}}
 store i32 0, i32* %r
 ; CHECK-NEXT: return
 ret void
}

; CHECK-LABEL: dynamic_static_alloca:
define void @dynamic_static_alloca(i32 %alloc) noredzone {
 ; Decrement SP in the prolog by the static amount and writeback to memory.
 ; CHECK: i32.const $push[[L11:.+]]=, 0{{$}}
 ; CHECK: i32.const $push[[L8:.+]]=, 0{{$}}
 ; CHECK-NEXT: i32.load $push[[L9:.+]]=, __stack_pointer($pop[[L8]])
 ; CHECK-NEXT: i32.const $push[[L10:.+]]=, 16
 ; CHECK-NEXT: i32.sub $push[[L20:.+]]=, $pop[[L9]], $pop[[L10]]
 ; CHECK-NEXT: tee_local $push[[L19:.+]]=, $[[FP:.+]]=, $pop[[L20]]
 ; CHECK:      i32.store $drop=, __stack_pointer($pop{{.+}}), $pop{{.+}}
 ; Decrement SP in the body by the dynamic amount.
 ; CHECK: i32.sub
 ; Writeback to memory.
 ; CHECK: i32.store $drop=, __stack_pointer($pop{{.+}}), $pop{{.+}}
 %r1 = alloca i32
 %r = alloca i32, i32 %alloc
 store i32 0, i32* %r
 ; CHEC: i32.store $drop=, 0($pop{{.+}}), $pop{{.+}}
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
 ; CHECK-NEXT: copy_local [[COPY:.+]]=, $pop[[ADDR]]
 br label %body
body:
 %a = phi i32* [%addr, %entry], [%b, %body]
 store i32 1, i32* %a
 ; CHECK: i32.store {{.*}}, 0([[COPY]]),
 br i1 %cond, label %body, label %exit
exit:
 ret void
}

declare void @use_i8_star(i8*)
declare i8* @llvm.frameaddress(i32)

; Test __builtin_frame_address(0).
; CHECK-LABEL: frameaddress_0:
; CHECK: i32.const $push[[L0:.+]]=, 0{{$}}
; CHECK-NEXT: i32.load $push[[L3:.+]]=, __stack_pointer($pop[[L0]])
; CHECK-NEXT: copy_local $push[[L4:.+]]=, $pop[[L3]]{{$}}
; CHECK-NEXT: tee_local $push[[L2:.+]]=, $[[FP:.+]]=, $pop[[L4]]{{$}}
; CHECK-NEXT: call use_i8_star@FUNCTION, $pop[[L2]]
; CHECK-NEXT: i32.const $push[[L1:.+]]=, 0{{$}}
; CHECK-NEXT: i32.store $drop=, __stack_pointer($pop[[L1]]), $[[FP]]
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
; CHECK:       __stack_pointer
; CHECK:       #APP
; CHECK-NEXT:  # %{{[0-9]+}}{{$}}
; CHECK-NEXT:  #NO_APP
define void @inline_asm() {
  %tmp = alloca i8
  call void asm sideeffect "# %0", "r"(i8* %tmp)
  ret void
}

; TODO: test over-aligned alloca

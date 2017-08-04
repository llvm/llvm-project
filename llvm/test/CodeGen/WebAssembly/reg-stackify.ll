; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -verify-machineinstrs | FileCheck %s

; Test the register stackifier pass.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

; No because of pointer aliasing.

; CHECK-LABEL: no0:
; CHECK: return $1{{$}}
define i32 @no0(i32* %p, i32* %q) {
  %t = load i32, i32* %q
  store i32 0, i32* %p
  ret i32 %t
}

; No because of side effects.

; CHECK-LABEL: no1:
; CHECK: return $1{{$}}
define i32 @no1(i32* %p, i32* dereferenceable(4) %q) {
  %t = load volatile i32, i32* %q, !invariant.load !0
  store volatile i32 0, i32* %p
  ret i32 %t
}

; Yes because of invariant load and no side effects.

; CHECK-LABEL: yes0:
; CHECK: return $pop{{[0-9]+}}{{$}}
define i32 @yes0(i32* %p, i32* dereferenceable(4) %q) {
  %t = load i32, i32* %q, !invariant.load !0
  store i32 0, i32* %p
  ret i32 %t
}

; Yes because of no intervening side effects.

; CHECK-LABEL: yes1:
; CHECK: return $pop0{{$}}
define i32 @yes1(i32* %q) {
  %t = load volatile i32, i32* %q
  ret i32 %t
}

; Yes because undefined behavior can be sunk past a store.

; CHECK-LABEL: sink_trap:
; CHECK: return $pop{{[0-9]+}}{{$}}
define i32 @sink_trap(i32 %x, i32 %y, i32* %p) {
  %t = sdiv i32 %x, %y
  store volatile i32 0, i32* %p
  ret i32 %t
}

; Yes because the call is readnone.

; CHECK-LABEL: sink_readnone_call:
; CHECK: return $pop0{{$}}
declare i32 @readnone_callee() readnone nounwind
define i32 @sink_readnone_call(i32 %x, i32 %y, i32* %p) {
  %t = call i32 @readnone_callee()
  store volatile i32 0, i32* %p
  ret i32 %t
}

; No because the call is readonly and there's an intervening store.

; CHECK-LABEL: no_sink_readonly_call:
; CHECK: return ${{[0-9]+}}{{$}}
declare i32 @readonly_callee() readonly nounwind
define i32 @no_sink_readonly_call(i32 %x, i32 %y, i32* %p) {
  %t = call i32 @readonly_callee()
  store i32 0, i32* %p
  ret i32 %t
}

; Don't schedule stack uses into the stack. To reduce register pressure, the
; scheduler might be tempted to move the definition of $2 down. However, this
; would risk getting incorrect liveness if the instructions are later
; rearranged to make the stack contiguous.

; CHECK-LABEL: stack_uses:
; CHECK: .param i32, i32, i32, i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: block   {{$}}
; CHECK-NEXT: i32.const   $push[[L13:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.lt_s    $push[[L0:[0-9]+]]=, $0, $pop[[L13]]{{$}}
; CHECK-NEXT: i32.const   $push[[L1:[0-9]+]]=, 2{{$}}
; CHECK-NEXT: i32.lt_s    $push[[L2:[0-9]+]]=, $1, $pop[[L1]]{{$}}
; CHECK-NEXT: i32.xor     $push[[L5:[0-9]+]]=, $pop[[L0]], $pop[[L2]]{{$}}
; CHECK-NEXT: i32.const   $push[[L12:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.lt_s    $push[[L3:[0-9]+]]=, $2, $pop[[L12]]{{$}}
; CHECK-NEXT: i32.const   $push[[L11:[0-9]+]]=, 2{{$}}
; CHECK-NEXT: i32.lt_s    $push[[L4:[0-9]+]]=, $3, $pop[[L11]]{{$}}
; CHECK-NEXT: i32.xor     $push[[L6:[0-9]+]]=, $pop[[L3]], $pop[[L4]]{{$}}
; CHECK-NEXT: i32.xor     $push[[L7:[0-9]+]]=, $pop[[L5]], $pop[[L6]]{{$}}
; CHECK-NEXT: i32.const   $push10=, 1{{$}}
; CHECK-NEXT: i32.ne      $push8=, $pop7, $pop10{{$}}
; CHECK-NEXT: br_if       0, $pop8{{$}}
; CHECK-NEXT: i32.const   $push9=, 0{{$}}
; CHECK-NEXT: return      $pop9{{$}}
; CHECK-NEXT: .LBB7_2:
; CHECK-NEXT: end_block{{$}}
; CHECK-NEXT: i32.const   $push14=, 1{{$}}
; CHECK-NEXT: return      $pop14{{$}}
define i32 @stack_uses(i32 %x, i32 %y, i32 %z, i32 %w) {
entry:
  %c = icmp sle i32 %x, 0
  %d = icmp sle i32 %y, 1
  %e = icmp sle i32 %z, 0
  %f = icmp sle i32 %w, 1
  %g = xor i1 %c, %d
  %h = xor i1 %e, %f
  %i = xor i1 %g, %h
  br i1 %i, label %true, label %false
true:
  ret i32 0
false:
  ret i32 1
}

; Test an interesting case where the load has multiple uses and cannot
; be trivially stackified. However, it can be stackified with a tee_local.

; CHECK-LABEL: multiple_uses:
; CHECK: .param       i32, i32, i32{{$}}
; CHECK-NEXT: block   {{$}}
; CHECK-NEXT: i32.load    $push[[NUM0:[0-9]+]]=, 0($2){{$}}
; CHECK-NEXT: tee_local   $push[[NUM1:[0-9]+]]=, $3=, $pop[[NUM0]]{{$}}
; CHECK-NEXT: i32.ge_u    $push[[NUM2:[0-9]+]]=, $pop[[NUM1]], $1{{$}}
; CHECK-NEXT: br_if       0, $pop[[NUM2]]{{$}}
; CHECK-NEXT: i32.lt_u    $push[[NUM3:[0-9]+]]=, $3, $0{{$}}
; CHECK-NEXT: br_if       0, $pop[[NUM3]]{{$}}
; CHECK-NEXT: i32.store   0($2), $3{{$}}
; CHECK-NEXT: .LBB8_3:
; CHECK-NEXT: end_block{{$}}
; CHECK-NEXT: return{{$}}
define void @multiple_uses(i32* %arg0, i32* %arg1, i32* %arg2) nounwind {
bb:
  br label %loop

loop:
  %tmp7 = load i32, i32* %arg2
  %tmp8 = inttoptr i32 %tmp7 to i32*
  %tmp9 = icmp uge i32* %tmp8, %arg1
  %tmp10 = icmp ult i32* %tmp8, %arg0
  %tmp11 = or i1 %tmp9, %tmp10
  br i1 %tmp11, label %back, label %then

then:
  store i32 %tmp7, i32* %arg2
  br label %back

back:
  br i1 undef, label %return, label %loop

return:
  ret void
}

; Don't stackify stores effects across other instructions with side effects.

; CHECK:      side_effects:
; CHECK:      store
; CHECK-NEXT: call
; CHECK:      store
; CHECK-NEXT: call
declare void @evoke_side_effects()
define hidden void @stackify_store_across_side_effects(double* nocapture %d) {
entry:
  store double 2.0, double* %d
  call void @evoke_side_effects()
  store double 2.0, double* %d
  call void @evoke_side_effects()
  ret void
}

; Div instructions have side effects and can't be reordered, but this entire
; function should still be able to be stackified because it's already in
; tree order.

; CHECK-LABEL: div_tree:
; CHECK: .param i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32{{$}}
; CHECK-NEXT: .result     i32{{$}}
; CHECK-NEXT: i32.div_s   $push[[L0:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: i32.div_s   $push[[L1:[0-9]+]]=, $2, $3{{$}}
; CHECK-NEXT: i32.div_s   $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: i32.div_s   $push[[L3:[0-9]+]]=, $4, $5{{$}}
; CHECK-NEXT: i32.div_s   $push[[L4:[0-9]+]]=, $6, $7{{$}}
; CHECK-NEXT: i32.div_s   $push[[L5:[0-9]+]]=, $pop[[L3]], $pop[[L4]]{{$}}
; CHECK-NEXT: i32.div_s   $push[[L6:[0-9]+]]=, $pop[[L2]], $pop[[L5]]{{$}}
; CHECK-NEXT: i32.div_s   $push[[L7:[0-9]+]]=, $8, $9{{$}}
; CHECK-NEXT: i32.div_s   $push[[L8:[0-9]+]]=, $10, $11{{$}}
; CHECK-NEXT: i32.div_s   $push[[L9:[0-9]+]]=, $pop[[L7]], $pop[[L8]]{{$}}
; CHECK-NEXT: i32.div_s   $push[[L10:[0-9]+]]=, $12, $13{{$}}
; CHECK-NEXT: i32.div_s   $push[[L11:[0-9]+]]=, $14, $15{{$}}
; CHECK-NEXT: i32.div_s   $push[[L12:[0-9]+]]=, $pop[[L10]], $pop[[L11]]{{$}}
; CHECK-NEXT: i32.div_s   $push[[L13:[0-9]+]]=, $pop[[L9]], $pop[[L12]]{{$}}
; CHECK-NEXT: i32.div_s   $push[[L14:[0-9]+]]=, $pop[[L6]], $pop[[L13]]{{$}}
; CHECK-NEXT: return      $pop[[L14]]{{$}}
define i32 @div_tree(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h, i32 %i, i32 %j, i32 %k, i32 %l, i32 %m, i32 %n, i32 %o, i32 %p) {
entry:
  %div = sdiv i32 %a, %b
  %div1 = sdiv i32 %c, %d
  %div2 = sdiv i32 %div, %div1
  %div3 = sdiv i32 %e, %f
  %div4 = sdiv i32 %g, %h
  %div5 = sdiv i32 %div3, %div4
  %div6 = sdiv i32 %div2, %div5
  %div7 = sdiv i32 %i, %j
  %div8 = sdiv i32 %k, %l
  %div9 = sdiv i32 %div7, %div8
  %div10 = sdiv i32 %m, %n
  %div11 = sdiv i32 %o, %p
  %div12 = sdiv i32 %div10, %div11
  %div13 = sdiv i32 %div9, %div12
  %div14 = sdiv i32 %div6, %div13
  ret i32 %div14
}

; A simple multiple-use case.

; CHECK-LABEL: simple_multiple_use:
; CHECK:  .param      i32, i32{{$}}
; CHECK-NEXT:  i32.mul     $push[[NUM0:[0-9]+]]=, $1, $0{{$}}
; CHECK-NEXT:  tee_local   $push[[NUM1:[0-9]+]]=, $[[NUM2:[0-9]+]]=, $pop[[NUM0]]{{$}}
; CHECK-NEXT:  call        use_a@FUNCTION, $pop[[NUM1]]{{$}}
; CHECK-NEXT:  call        use_b@FUNCTION, $[[NUM2]]{{$}}
; CHECK-NEXT:  return{{$}}
declare void @use_a(i32)
declare void @use_b(i32)
define void @simple_multiple_use(i32 %x, i32 %y) {
  %mul = mul i32 %y, %x
  call void @use_a(i32 %mul)
  call void @use_b(i32 %mul)
  ret void
}

; Multiple uses of the same value in one instruction.

; CHECK-LABEL: multiple_uses_in_same_insn:
; CHECK:  .param      i32, i32{{$}}
; CHECK-NEXT:  i32.mul     $push[[NUM0:[0-9]+]]=, $1, $0{{$}}
; CHECK-NEXT:  tee_local   $push[[NUM1:[0-9]+]]=, $[[NUM2:[0-9]+]]=, $pop[[NUM0]]{{$}}
; CHECK-NEXT:  call        use_2@FUNCTION, $pop[[NUM1]], $[[NUM2]]{{$}}
; CHECK-NEXT:  return{{$}}
declare void @use_2(i32, i32)
define void @multiple_uses_in_same_insn(i32 %x, i32 %y) {
  %mul = mul i32 %y, %x
  call void @use_2(i32 %mul, i32 %mul)
  ret void
}

; Commute operands to achieve better stackifying.

; CHECK-LABEL: commute:
; CHECK-NOT: param
; CHECK:  .result     i32{{$}}
; CHECK-NEXT:  i32.call    $push0=, red@FUNCTION{{$}}
; CHECK-NEXT:  i32.call    $push1=, green@FUNCTION{{$}}
; CHECK-NEXT:  i32.add     $push2=, $pop0, $pop1{{$}}
; CHECK-NEXT:  i32.call    $push3=, blue@FUNCTION{{$}}
; CHECK-NEXT:  i32.add     $push4=, $pop2, $pop3{{$}}
; CHECK-NEXT:  return      $pop4{{$}}
declare i32 @red()
declare i32 @green()
declare i32 @blue()
define i32 @commute() {
  %call = call i32 @red()
  %call1 = call i32 @green()
  %add = add i32 %call1, %call
  %call2 = call i32 @blue()
  %add3 = add i32 %add, %call2
  ret i32 %add3
}

; Don't stackify a register when it would move a the def of the register past
; an implicit get_local for the register.

; CHECK-LABEL: no_stackify_past_use:
; CHECK:      i32.call        $1=, callee@FUNCTION, $0
; CHECK-NEXT: i32.const       $push0=, 1
; CHECK-NEXT: i32.add         $push1=, $0, $pop0
; CHECK-NEXT: i32.call        $push2=, callee@FUNCTION, $pop1
; CHECK-NEXT: i32.sub         $push3=, $pop2, $1
; CHECK-NEXT: i32.div_s       $push4=, $pop3, $1
; CHECK-NEXT: return          $pop4
declare i32 @callee(i32)
define i32 @no_stackify_past_use(i32 %arg) {
  %tmp1 = call i32 @callee(i32 %arg)
  %tmp2 = add i32 %arg, 1
  %tmp3 = call i32 @callee(i32 %tmp2)
  %tmp5 = sub i32 %tmp3, %tmp1
  %tmp6 = sdiv i32 %tmp5, %tmp1
  ret i32 %tmp6
}

; This is the same as no_stackify_past_use, except using a commutative operator,
; so we can reorder the operands and stackify.

; CHECK-LABEL: commute_to_fix_ordering:
; CHECK: i32.call        $push[[L0:.+]]=, callee@FUNCTION, $0
; CHECK: tee_local       $push[[L1:.+]]=, $1=, $pop[[L0]]
; CHECK: i32.const       $push0=, 1
; CHECK: i32.add         $push1=, $0, $pop0
; CHECK: i32.call        $push2=, callee@FUNCTION, $pop1
; CHECK: i32.add         $push3=, $1, $pop2
; CHECK: i32.mul         $push4=, $pop[[L1]], $pop3
; CHECK: return          $pop4
define i32 @commute_to_fix_ordering(i32 %arg) {
  %tmp1 = call i32 @callee(i32 %arg)
  %tmp2 = add i32 %arg, 1
  %tmp3 = call i32 @callee(i32 %tmp2)
  %tmp5 = add i32 %tmp3, %tmp1
  %tmp6 = mul i32 %tmp5, %tmp1
  ret i32 %tmp6
}

; Stackify individual defs of virtual registers with multiple defs.

; CHECK-LABEL: multiple_defs:
; CHECK:        f64.add         $push[[NUM0:[0-9]+]]=, ${{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT:   tee_local       $push[[NUM1:[0-9]+]]=, $[[NUM2:[0-9]+]]=, $pop[[NUM0]]{{$}}
; CHECK-NEXT:   f64.select      $push{{[0-9]+}}=, $pop{{[0-9]+}}, $pop[[NUM1]], ${{[0-9]+}}{{$}}
; CHECK:        $[[NUM2]]=,
define void @multiple_defs(i32 %arg, i32 %arg1, i1 %arg2, i1 %arg3, i1 %arg4) {
bb:
  br label %bb5

bb5:                                              ; preds = %bb21, %bb
  %tmp = phi double [ 0.000000e+00, %bb ], [ %tmp22, %bb21 ]
  %tmp6 = phi double [ 0.000000e+00, %bb ], [ %tmp23, %bb21 ]
  %tmp7 = fcmp olt double %tmp6, 2.323450e+01
  br i1 %tmp7, label %bb8, label %bb21

bb8:                                              ; preds = %bb17, %bb5
  %tmp9 = phi double [ %tmp19, %bb17 ], [ %tmp, %bb5 ]
  %tmp10 = fadd double %tmp6, -1.000000e+00
  %tmp11 = select i1 %arg2, double -1.135357e+04, double %tmp10
  %tmp12 = fadd double %tmp11, %tmp9
  br i1 %arg3, label %bb17, label %bb13

bb13:                                             ; preds = %bb8
  %tmp14 = or i32 %arg1, 2
  %tmp15 = icmp eq i32 %tmp14, 14
  %tmp16 = select i1 %tmp15, double -1.135357e+04, double 0xBFCE147AE147B000
  br label %bb17

bb17:                                             ; preds = %bb13, %bb8
  %tmp18 = phi double [ %tmp16, %bb13 ], [ %tmp10, %bb8 ]
  %tmp19 = fadd double %tmp18, %tmp12
  %tmp20 = fcmp olt double %tmp6, 2.323450e+01
  br i1 %tmp20, label %bb8, label %bb21

bb21:                                             ; preds = %bb17, %bb5
  %tmp22 = phi double [ %tmp, %bb5 ], [ %tmp9, %bb17 ]
  %tmp23 = fadd double %tmp6, 1.000000e+00
  br label %bb5
}

; Don't move calls past loads
; CHECK-LABEL: no_stackify_call_past_load:
; CHECK: i32.call $0=, red
; CHECK: i32.const $push0=, 0
; CHECK: i32.load $1=, count($pop0)
@count = hidden global i32 0, align 4
define i32 @no_stackify_call_past_load() {
  %a = call i32 @red()
  %b = load i32, i32* @count, align 4
  call i32 @callee(i32 %a)
  ret i32 %b
  ; use of a
}

; Don't move stores past loads if there may be aliasing
; CHECK-LABEL: no_stackify_store_past_load
; CHECK: i32.store 0($1), $0
; CHECK: i32.load {{.*}}, 0($2)
; CHECK: i32.call {{.*}}, callee@FUNCTION, $0{{$}}
define i32 @no_stackify_store_past_load(i32 %a, i32* %p1, i32* %p2) {
  store i32 %a, i32* %p1
  %b = load i32, i32* %p2, align 4
  call i32 @callee(i32 %a)
  ret i32 %b
}

; Can still stackify past invariant loads.
; CHECK-LABEL: store_past_invar_load
; CHECK: i32.store 0($1), $0
; CHECK: i32.call {{.*}}, callee@FUNCTION, $0
; CHECK: i32.load $push{{.*}}, 0($2)
; CHECK: return $pop
define i32 @store_past_invar_load(i32 %a, i32* %p1, i32* dereferenceable(4) %p2) {
  store i32 %a, i32* %p1
  %b = load i32, i32* %p2, !invariant.load !0
  call i32 @callee(i32 %a)
  ret i32 %b
}

; CHECK-LABEL: ignore_dbg_value:
; CHECK-NEXT: .Lfunc_begin
; CHECK-NEXT: unreachable
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)
define void @ignore_dbg_value() {
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !7, metadata !9), !dbg !10
  unreachable
}

; Don't stackify an expression that might use the stack into a return, since we
; might insert a prologue before the return.

; CHECK-LABEL: no_stackify_past_epilogue:
; CHECK: return ${{[0-9]+}}{{$}}
declare i32 @use_memory(i32*)
define i32 @no_stackify_past_epilogue() {
  %x = alloca i32
  %call = call i32 @use_memory(i32* %x)
  ret i32 %call
}

; Stackify a loop induction variable into a loop comparison.

; CHECK-LABEL: stackify_indvar:
; CHECK:             i32.const   $push[[L5:.+]]=, 1{{$}}
; CHECK-NEXT:        i32.add     $push[[L4:.+]]=, $[[R0:.+]], $pop[[L5]]{{$}}
; CHECK-NEXT:        tee_local   $push[[L3:.+]]=, $[[R0]]=, $pop[[L4]]{{$}}
; CHECK-NEXT:        i32.ne      $push[[L2:.+]]=, $0, $pop[[L3]]{{$}}
define void @stackify_indvar(i32 %tmp, i32* %v) #0 {
bb:
  br label %bb3

bb3:                                              ; preds = %bb3, %bb2
  %tmp4 = phi i32 [ %tmp7, %bb3 ], [ 0, %bb ]
  %tmp5 = load volatile i32, i32* %v, align 4
  %tmp6 = add nsw i32 %tmp5, %tmp4
  store volatile i32 %tmp6, i32* %v, align 4
  %tmp7 = add nuw nsw i32 %tmp4, 1
  %tmp8 = icmp eq i32 %tmp7, %tmp
  br i1 %tmp8, label %bb10, label %bb3

bb10:                                             ; preds = %bb9, %bb
  ret void
}

; Don't stackify a call past a __stack_pointer store.

; CHECK-LABEL: stackpointer_dependency:
; CHECK:      call {{.+}}, stackpointer_callee@FUNCTION,
; CHECK-NEXT: set_global __stack_pointer,
declare i32 @stackpointer_callee(i8* readnone, i8* readnone)
declare i8* @llvm.frameaddress(i32)
define i32 @stackpointer_dependency(i8* readnone) {
  %2 = tail call i8* @llvm.frameaddress(i32 0)
  %3 = tail call i32 @stackpointer_callee(i8* %0, i8* %2)
  ret i32 %3
}

; Stackify a call_indirect with respect to its ordering

; CHECK-LABEL: call_indirect_stackify:
; CHECK: i32.load  $push[[L4:.+]]=, 0($0)
; CHECK-NEXT: tee_local $push[[L3:.+]]=, $0=, $pop[[L4]]
; CHECK-NEXT: i32.load  $push[[L0:.+]]=, 0($0)
; CHECK-NEXT: i32.load  $push[[L1:.+]]=, 0($pop[[L0]])
; CHECK-NEXT: i32.call_indirect $push{{.+}}=, $pop[[L3]], $1, $pop[[L1]]
%class.call_indirect = type { i32 (...)** }
define i32 @call_indirect_stackify(%class.call_indirect** %objptr, i32 %arg) {
  %obj = load %class.call_indirect*, %class.call_indirect** %objptr
  %addr = bitcast %class.call_indirect* %obj to i32(%class.call_indirect*, i32)***
  %vtable = load i32(%class.call_indirect*, i32)**, i32(%class.call_indirect*, i32)*** %addr
  %vfn = getelementptr inbounds i32(%class.call_indirect*, i32)*, i32(%class.call_indirect*, i32)** %vtable, i32 0
  %f = load i32(%class.call_indirect*, i32)*, i32(%class.call_indirect*, i32)** %vfn
  %ret = call i32 %f(%class.call_indirect* %obj, i32 %arg)
  ret i32 %ret
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 3.9.0 (trunk 266005) (llvm/trunk 266105)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3)
!2 = !DIFile(filename: "test.c", directory: "/")
!3 = !{}
!5 = distinct !DISubprogram(name: "test", scope: !2, file: !2, line: 10, type: !6, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: true, unit: !1, variables: !3)
!6 = !DISubroutineType(types: !3)
!7 = !DILocalVariable(name: "nzcnt", scope: !5, file: !2, line: 15, type: !8)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIExpression()
!10 = !DILocation(line: 15, column: 6, scope: !5)

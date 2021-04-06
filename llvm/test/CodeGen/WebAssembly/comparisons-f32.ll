; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers | FileCheck %s

; Test that basic 32-bit floating-point comparison operations assemble as
; expected.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: ord_f32:
; CHECK-NEXT: .functype ord_f32 (f32, f32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: f32.eq $push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: local.get $push[[L2:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: local.get $push[[L3:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f32.eq $push[[NUM1:[0-9]+]]=, $pop[[L2]], $pop[[L3]]{{$}}
; CHECK-NEXT: i32.and $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[NUM1]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}
define i32 @ord_f32(float %x, float %y) {
  %a = fcmp ord float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uno_f32:
; CHECK-NEXT: .functype uno_f32 (f32, f32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: f32.ne $push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: local.get $push[[L2:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: local.get $push[[L3:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f32.ne $push[[NUM1:[0-9]+]]=, $pop[[L2]], $pop[[L3]]{{$}}
; CHECK-NEXT: i32.or $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[NUM1]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}
define i32 @uno_f32(float %x, float %y) {
  %a = fcmp uno float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: oeq_f32:
; CHECK-NEXT: .functype oeq_f32 (f32, f32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f32.eq $push[[NUM:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @oeq_f32(float %x, float %y) {
  %a = fcmp oeq float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: une_f32:
; CHECK: f32.ne $push[[NUM:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @une_f32(float %x, float %y) {
  %a = fcmp une float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: olt_f32:
; CHECK: f32.lt $push[[NUM:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @olt_f32(float %x, float %y) {
  %a = fcmp olt float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ole_f32:
; CHECK: f32.le $push[[NUM:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ole_f32(float %x, float %y) {
  %a = fcmp ole float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ogt_f32:
; CHECK: f32.gt $push[[NUM:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ogt_f32(float %x, float %y) {
  %a = fcmp ogt float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: oge_f32:
; CHECK: f32.ge $push[[NUM:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @oge_f32(float %x, float %y) {
  %a = fcmp oge float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; Expanded comparisons, which also check for NaN.
; These simply rely on SDAG's Expand cond code action.

; CHECK-LABEL: ueq_f32:
; CHECK-NEXT: .functype ueq_f32 (f32, f32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f32.gt $push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: local.get $push[[L2:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L3:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f32.lt $push[[NUM1:[0-9]+]]=, $pop[[L2]], $pop[[L3]]{{$}}
; CHECK-NEXT: i32.or $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[NUM1]]{{$}}
; CHECK-NEXT: i32.const $push[[C0:[0-9]+]]=, 1
; CHECK-NEXT: i32.xor $push[[NUM3:[0-9]+]]=, $pop[[NUM2]], $pop[[C0]]{{$}}
; CHECK-NEXT: return $pop[[NUM3]]{{$}}
define i32 @ueq_f32(float %x, float %y) {
  %a = fcmp ueq float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: one_f32:
; CHECK-NEXT: .functype one_f32 (f32, f32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f32.gt $push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: local.get $push[[L2:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L3:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f32.lt $push[[NUM1:[0-9]+]]=, $pop[[L2]], $pop[[L3]]{{$}}
; CHECK-NEXT: i32.or $push[[NUM4:[0-9]+]]=, $pop[[NUM0]], $pop[[NUM1]]{{$}}
; CHECK-NEXT: return $pop[[NUM4]]
define i32 @one_f32(float %x, float %y) {
  %a = fcmp one float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ult_f32:
; CHECK-NEXT: .functype ult_f32 (f32, f32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f32.ge $push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: i32.const $push[[C0:[0-9]+]]=, 1
; CHECK-NEXT: i32.xor $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[C0]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}
define i32 @ult_f32(float %x, float %y) {
  %a = fcmp ult float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ule_f32:
; CHECK-NEXT: .functype ule_f32 (f32, f32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f32.gt $push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: i32.const $push[[C0:[0-9]+]]=, 1
; CHECK-NEXT: i32.xor $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[C0]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}
define i32 @ule_f32(float %x, float %y) {
  %a = fcmp ule float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ugt_f32:
; CHECK-NEXT: .functype ugt_f32 (f32, f32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f32.le $push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: i32.const $push[[C0:[0-9]+]]=, 1
; CHECK-NEXT: i32.xor $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[C0]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}
define i32 @ugt_f32(float %x, float %y) {
  %a = fcmp ugt float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uge_f32:
; CHECK-NEXT: .functype uge_f32 (f32, f32) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f32.lt $push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: i32.const $push[[C0:[0-9]+]]=, 1
; CHECK-NEXT: i32.xor $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[C0]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}
define i32 @uge_f32(float %x, float %y) {
  %a = fcmp uge float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: olt_f32_branch
; CHECK:      local.get	$push[[L4:[0-9]+]]=, 0
; CHECK-NEXT: local.get	$push[[L3:[0-9]+]]=, 1
; CHECK-NEXT: f32.lt  	$push[[NUM0:[0-9]+]]=, $pop[[L4]], $pop[[L3]]
; CHECK-NEXT: i32.eqz 	$push[[NUM3:[0-9]+]]=, $pop[[NUM0]]
; CHECK-NEXT: br_if   	0, $pop[[NUM3]]
; CHECK-NEXT: call	call1
define void @olt_f32_branch(float %a, float %b) {
entry:
  %cmp = fcmp olt float %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @call1()
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: ole_f32_branch
; CHECK:      local.get	$push[[L4:[0-9]+]]=, 0
; CHECK-NEXT: local.get	$push[[L3:[0-9]+]]=, 1
; CHECK-NEXT: f32.le  	$push[[NUM0:[0-9]+]]=, $pop[[L4]], $pop[[L3]]
; CHECK-NEXT: i32.eqz 	$push[[NUM3:[0-9]+]]=, $pop[[NUM0]]
; CHECK-NEXT: br_if   	0, $pop[[NUM3]]
; CHECK-NEXT: call	call1
define void @ole_f32_branch(float %a, float %b) {
entry:
  %cmp = fcmp ole float %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @call1()
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: ugt_f32_branch
; CHECK:      local.get	$push[[L4:[0-9]+]]=, 0
; CHECK-NEXT: local.get	$push[[L3:[0-9]+]]=, 1
; CHECK-NEXT: f32.le  	$push[[NUM0:[0-9]+]]=, $pop[[L4]], $pop[[L3]]
; CHECK-NEXT: i32.eqz 	$push[[NUM3:[0-9]+]]=, $pop[[NUM0]]
; CHECK-NEXT: br_if   	0, $pop[[NUM3]]
; CHECK-NEXT: call	call1
define void @ugt_f32_branch(float %a, float %b) {
entry:
  %cmp = fcmp ugt float %a, %b
  br i1 %cmp, label %if.end, label %if.then

if.then:
  tail call void @call1()
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: ogt_f32_branch
; CHECK:      local.get	$push[[L4:[0-9]+]]=, 0
; CHECK-NEXT: local.get	$push[[L3:[0-9]+]]=, 1
; CHECK-NEXT: f32.gt  	$push[[NUM0:[0-9]+]]=, $pop[[L4]], $pop[[L3]]
; CHECK-NEXT: i32.eqz 	$push[[NUM3:[0-9]+]]=, $pop[[NUM0]]
; CHECK-NEXT: br_if   	0, $pop[[NUM3]]
; CHECK-NEXT: call	call1
define void @ogt_f32_branch(float %a, float %b) {
entry:
  %cmp = fcmp ogt float %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @call1()
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: ult_f32_branch
; CHECK:      local.get	$push[[L4:[0-9]+]]=, 0
; CHECK-NEXT: local.get	$push[[L3:[0-9]+]]=, 1
; CHECK-NEXT: f32.ge  	$push[[NUM0:[0-9]+]]=, $pop[[L4]], $pop[[L3]]
; CHECK-NEXT: i32.eqz 	$push[[NUM3:[0-9]+]]=, $pop[[NUM0]]
; CHECK-NEXT: br_if   	0, $pop[[NUM3]]
; CHECK-NEXT: call	call1
define void @ult_f32_branch(float %a, float %b) {
entry:
  %cmp = fcmp ult float %a, %b
  br i1 %cmp, label %if.end, label %if.then

if.then:
  tail call void @call1()
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: ule_f32_branch
; CHECK:      local.get	$push[[L4:[0-9]+]]=, 0
; CHECK-NEXT: local.get	$push[[L3:[0-9]+]]=, 1
; CHECK-NEXT: f32.ge  	$push[[NUM0:[0-9]+]]=, $pop[[L4]], $pop[[L3]]
; CHECK-NEXT: i32.eqz 	$push[[NUM3:[0-9]+]]=, $pop[[NUM0]]
; CHECK-NEXT: br_if   	0, $pop[[NUM3]]
; CHECK-NEXT: call	call1
define void @ule_f32_branch(float %a, float %b) {
entry:
  %cmp = fcmp ult float %a, %b
  br i1 %cmp, label %if.end, label %if.then

if.then:
  tail call void @call1()
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: xor_zext_switch
; CHECK:      i32.const	$push[[L1:[0-9]+]]=, 0
; CHECK-NEXT: br_if   	0, $pop[[L1]]
; CHECK-NEXT: block
; CHECK-NEXT: block
; CHECK-NEXT: local.get	$push[[L3:[0-9]+]]=, 0
; CHECK-NEXT: local.get	$push[[L2:[0-9]+]]=, 1
; CHECK-NEXT: f32.ge  	$push[[L0:[0-9]+]]=, $pop[[L3]], $pop[[L2]]
; CHECK-NEXT: br_table 	$pop[[L0]], 0, 1, 0
define void @xor_zext_switch(float %a, float %b) {
entry:
  %cmp = fcmp ult float %a, %b
  %zext = zext i1 %cmp to i32
  %xor = xor i32 %zext, 1
  switch i32 %xor, label %exit [
    i32 0, label %sw.bb.1
    i32 1, label %sw.bb.2
  ]

sw.bb.1:
  tail call void @foo1()
  br label %exit

sw.bb.2:
  tail call void @foo2()
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: xor_add_switch
; CHECK:      local.get	$push[[L8:[0-9]+]]=, 0
; CHECK-NEXT: local.get	$push[[L7:[0-9]+]]=, 1
; CHECK-NEXT: f32.ge  	$push[[L1:[0-9]+]]=, $pop[[L8]], $pop[[L7]]
; CHECK-NEXT: i32.const	$push[[L2:[0-9]+]]=, 1
; CHECK-NEXT: i32.xor 	$push[[L3:[0-9]+]]=, $pop[[L1]], $pop[[L2]]
; CHECK-NEXT: i32.const	$push[[L6:[0-9]+]]=, 1
; CHECK-NEXT: i32.add 	$push[[L4:[0-9]+]]=, $pop[[L3]], $pop[[L6]]
; CHECK-NEXT: i32.const	$push[[L5:[0-9]+]]=, 1
; CHECK-NEXT: i32.xor 	$push[[L0:[0-9]+]]=, $pop[[L4]], $pop[[L5]]
; CHECK-NEXT: br_table 	$pop[[L0]], 0, 1, 2, 3
define void @xor_add_switch(float %a, float %b) {
entry:
  %cmp = fcmp ult float %a, %b
  %zext = zext i1 %cmp to i32
  %add = add nsw nuw i32 %zext, 1
  %xor = xor i32 %add, 1
  switch i32 %xor, label %exit [
    i32 0, label %sw.bb.1
    i32 1, label %sw.bb.2
    i32 2, label %sw.bb.3
  ]

sw.bb.1:
  tail call void @foo1()
  br label %exit

sw.bb.2:
  tail call void @foo2()
  br label %exit

sw.bb.3:
  tail call void @foo3()
  br label %exit

exit:
  ret void
}

declare void @foo1()
declare void @foo2()
declare void @foo3()
declare void @call1()

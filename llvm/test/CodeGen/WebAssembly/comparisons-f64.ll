; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers | FileCheck %s

; Test that basic 64-bit floating-point comparison operations assemble as
; expected.

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: ord_f64:
; CHECK-NEXT: .functype ord_f64 (f64, f64) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: f64.eq $push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: local.get $push[[L2:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: local.get $push[[L3:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f64.eq $push[[NUM1:[0-9]+]]=, $pop[[L2]], $pop[[L3]]{{$}}
; CHECK-NEXT: i32.and $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[NUM1]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}
define i32 @ord_f64(double %x, double %y) {
  %a = fcmp ord double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uno_f64:
; CHECK-NEXT: .functype uno_f64 (f64, f64) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: f64.ne $push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: local.get $push[[L2:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: local.get $push[[L3:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f64.ne $push[[NUM1:[0-9]+]]=, $pop[[L2]], $pop[[L3]]{{$}}
; CHECK-NEXT: i32.or $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[NUM1]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}
define i32 @uno_f64(double %x, double %y) {
  %a = fcmp uno double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: oeq_f64:
; CHECK-NEXT: .functype oeq_f64 (f64, f64) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f64.eq $push[[NUM:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @oeq_f64(double %x, double %y) {
  %a = fcmp oeq double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: une_f64:
; CHECK: f64.ne $push[[NUM:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @une_f64(double %x, double %y) {
  %a = fcmp une double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: olt_f64:
; CHECK: f64.lt $push[[NUM:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @olt_f64(double %x, double %y) {
  %a = fcmp olt double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ole_f64:
; CHECK: f64.le $push[[NUM:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ole_f64(double %x, double %y) {
  %a = fcmp ole double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ogt_f64:
; CHECK: f64.gt $push[[NUM:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @ogt_f64(double %x, double %y) {
  %a = fcmp ogt double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: oge_f64:
; CHECK: f64.ge $push[[NUM:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop[[NUM]]{{$}}
define i32 @oge_f64(double %x, double %y) {
  %a = fcmp oge double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; Expanded comparisons, which also check for NaN.

; CHECK-LABEL: ueq_f64:
; CHECK-NEXT: .functype ueq_f64 (f64, f64) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f64.gt $push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: local.get $push[[L2:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L3:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f64.lt $push[[NUM1:[0-9]+]]=, $pop[[L2]], $pop[[L3]]{{$}}
; CHECK-NEXT: i32.or $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[NUM1]]{{$}}
; CHECK-NEXT: i32.const $push[[C0:[0-9]+]]=, 1
; CHECK-NEXT: i32.xor $push[[NUM3:[0-9]+]]=, $pop[[NUM2]], $pop[[C0]]{{$}}
; CHECK-NEXT: return $pop[[NUM3]]{{$}}
define i32 @ueq_f64(double %x, double %y) {
  %a = fcmp ueq double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: one_f64:
; CHECK-NEXT: .functype one_f64 (f64, f64) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f64.gt $push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: local.get $push[[L2:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L3:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f64.lt $push[[NUM1:[0-9]+]]=, $pop[[L2]], $pop[[L3]]{{$}}
; CHECK-NEXT: i32.or $push[[NUM4:[0-9]+]]=, $pop[[NUM0]], $pop[[NUM1]]{{$}}
; CHECK-NEXT: return $pop[[NUM4]]
define i32 @one_f64(double %x, double %y) {
  %a = fcmp one double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ult_f64:
; CHECK-NEXT: .functype ult_f64 (f64, f64) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f64.ge $push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: i32.const $push[[C0:[0-9]+]]=, 1
; CHECK-NEXT: i32.xor $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[C0]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}
define i32 @ult_f64(double %x, double %y) {
  %a = fcmp ult double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ule_f64:
; CHECK-NEXT: .functype ule_f64 (f64, f64) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f64.gt $push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: i32.const $push[[C0:[0-9]+]]=, 1
; CHECK-NEXT: i32.xor $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[C0]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}
define i32 @ule_f64(double %x, double %y) {
  %a = fcmp ule double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ugt_f64:
; CHECK-NEXT: .functype ugt_f64 (f64, f64) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f64.le $push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: i32.const $push[[C0:[0-9]+]]=, 1
; CHECK-NEXT: i32.xor $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[C0]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}
define i32 @ugt_f64(double %x, double %y) {
  %a = fcmp ugt double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uge_f64:
; CHECK-NEXT: .functype uge_f64 (f64, f64) -> (i32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f64.lt $push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: i32.const $push[[C0:[0-9]+]]=, 1
; CHECK-NEXT: i32.xor $push[[NUM2:[0-9]+]]=, $pop[[NUM0]], $pop[[C0]]{{$}}
; CHECK-NEXT: return $pop[[NUM2]]{{$}}
define i32 @uge_f64(double %x, double %y) {
  %a = fcmp uge double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: olt_f64_branch:
; CHECK:      local.get	$push[[L0:[0-9]+]]=, 0
; CHECK-NEXT: local.get	$push[[L1:[0-9]+]]=, 1
; CHECK-NEXT: f64.lt  	$push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]
; CHECK-NEXT: i32.eqz   $push[[NUM3:[0-9]+]]=, $pop[[NUM0]]
; CHECK-NEXT: br_if   	0, $pop[[NUM3]]
; CHECK-NEXT: call	call1
define void @olt_f64_branch(double %a, double %b) {
entry:
  %cmp = fcmp olt double %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @call1()
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: ole_f64_branch:
; CHECK:      local.get	$push[[L0:[0-9]+]]=, 0
; CHECK-NEXT: local.get	$push[[L1:[0-9]+]]=, 1
; CHECK-NEXT: f64.le  	$push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]
; CHECK-NEXT: i32.eqz   $push[[NUM3:[0-9]+]]=, $pop[[NUM0]]
; CHECK-NEXT: br_if   	0, $pop[[NUM3]]
; CHECK-NEXT: call	call1
define void @ole_f64_branch(double %a, double %b) {
entry:
  %cmp = fcmp ole double %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @call1()
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: ugt_f64_branch:
; CHECK:      local.get	$push[[L0:[0-9]+]]=, 0
; CHECK-NEXT: local.get	$push[[L1:[0-9]+]]=, 1
; CHECK-NEXT: f64.le  	$push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]
; CHECK-NEXT: i32.eqz   $push[[NUM3:[0-9]+]]=, $pop[[NUM0]]
; CHECK-NEXT: br_if   	0, $pop[[NUM3]]
; CHECK-NEXT: call	call1
define void @ugt_f64_branch(double %a, double %b) {
entry:
  %cmp = fcmp ugt double %a, %b
  br i1 %cmp, label %if.end, label %if.then

if.then:
  tail call void @call1()
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: ogt_f64_branch:
; CHECK:      local.get	$push[[L0:[0-9]+]]=, 0
; CHECK-NEXT: local.get	$push[[L1:[0-9]+]]=, 1
; CHECK-NEXT: f64.gt  	$push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]
; CHECK-NEXT: i32.eqz   $push[[NUM3:[0-9]+]]=, $pop[[NUM0]]
; CHECK-NEXT: br_if   	0, $pop[[NUM3]]
; CHECK-NEXT: call	call1
define void @ogt_f64_branch(double %a, double %b) {
entry:
  %cmp = fcmp ogt double %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @call1()
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: ult_f64_branch:
; CHECK:      local.get	$push[[L0:[0-9]+]]=, 0
; CHECK-NEXT: local.get	$push[[L1:[0-9]+]]=, 1
; CHECK-NEXT: f64.ge  	$push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]
; CHECK-NEXT: i32.eqz   $push[[NUM3:[0-9]+]]=, $pop[[NUM0]]
; CHECK-NEXT: br_if   	0, $pop[[NUM3]]
; CHECK-NEXT: call	call1
define void @ult_f64_branch(double %a, double %b) {
entry:
  %cmp = fcmp ult double %a, %b
  br i1 %cmp, label %if.end, label %if.then

if.then:
  tail call void @call1()
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: ule_f64_branch:
; CHECK:      local.get	$push[[L0:[0-9]+]]=, 0
; CHECK-NEXT: local.get	$push[[L1:[0-9]+]]=, 1
; CHECK-NEXT: f64.gt  	$push[[NUM0:[0-9]+]]=, $pop[[L0]], $pop[[L1]]
; CHECK-NEXT: i32.eqz   $push[[NUM3:[0-9]+]]=, $pop[[NUM0]]
; CHECK-NEXT: br_if   	0, $pop[[NUM3]]
; CHECK-NEXT: call	call1
define void @ule_f64_branch(double %a, double %b) {
entry:
  %cmp = fcmp ule double %a, %b
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
; CHECK-NEXT: f64.ge  	$push[[L0:[0-9]+]]=, $pop[[L3]], $pop[[L2]]
; CHECK-NEXT: br_table 	$pop[[L0]], 0, 1, 0
define void @xor_zext_switch(double %a, double %b) {
entry:
  %cmp = fcmp ult double %a, %b
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
; CHECK-NEXT: f64.ge  	$push[[L1:[0-9]+]]=, $pop[[L8]], $pop[[L7]]
; CHECK-NEXT: i32.const	$push[[L2:[0-9]+]]=, 1
; CHECK-NEXT: i32.xor 	$push[[L3:[0-9]+]]=, $pop[[L1]], $pop[[L2]]
; CHECK-NEXT: i32.const	$push[[L6:[0-9]+]]=, 1
; CHECK-NEXT: i32.add 	$push[[L4:[0-9]+]]=, $pop[[L3]], $pop[[L6]]
; CHECK-NEXT: i32.const	$push[[L5:[0-9]+]]=, 1
; CHECK-NEXT: i32.xor 	$push[[L0:[0-9]+]]=, $pop[[L4]], $pop[[L5]]
; CHECK-NEXT: br_table 	$pop[[L0]], 0, 1, 2, 3
define void @xor_add_switch(double %a, double %b) {
entry:
  %cmp = fcmp ult double %a, %b
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

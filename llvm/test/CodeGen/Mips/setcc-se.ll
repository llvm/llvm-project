; RUN: llc -mtriple=mipsel < %s | FileCheck %s
; RUN: llc  -mtriple=mips -mcpu=mips32r6 -mattr=micromips -relocation-model=pic < %s -asm-show-inst | FileCheck %s -check-prefix=MMR6

@g1 = external global i32

; CHECK-LABEL: seteq0:
; CHECK:  sltiu ${{[0-9]+}}, $4, 1
; MMR6:   sltiu ${{[0-9]+}}, $4, 1
; MMR6:   <MCInst #{{[0-9]+}} SLTiu_MM

define i32 @seteq0(i32 %a) {
entry:
  %cmp = icmp eq i32 %a, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: setne0:
; CHECK:  sltu ${{[0-9]+}}, $zero, $4
; MMR6:   sltu ${{[0-9]+}}, $zero, $4
; MMR6:   <MCInst #{{[0-9]+}} SLTu_MM

define i32 @setne0(i32 %a) {
entry:
  %cmp = icmp ne i32 %a, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: slti_beq0:
; CHECK:  slti $[[R0:[0-9]+]], $4, -32768
; MMR6:   slti $[[R0:[0-9]+]], $4, -32768
; MMR6:   <MCInst #{{[0-9]+}} SLTi_MM
; CHECK:  beqz $[[R0]]

define void @slti_beq0(i32 %a) {
entry:
  %cmp = icmp slt i32 %a, -32768
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %a, ptr @g1, align 4
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: slti_beq1:
; CHECK:  slt ${{[0-9]+}}
; MMR6:   slt ${{[0-9]+}}
; MMR6:   <MCInst #{{[0-9]+}} SLT_MM

define void @slti_beq1(i32 %a) {
entry:
  %cmp = icmp slt i32 %a, -32769
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %a, ptr @g1, align 4
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: slti_beq2:
; CHECK:  slti $[[R0:[0-9]+]], $4, 32767
; MMR6:   slti $[[R0:[0-9]+]], $4, 32767
; MMR6:   <MCInst #{{[0-9]+}} SLTi_MM
; CHECK:  beqz $[[R0]]

define void @slti_beq2(i32 %a) {
entry:
  %cmp = icmp slt i32 %a, 32767
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %a, ptr @g1, align 4
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: slti_beq3:
; CHECK:  slt ${{[0-9]+}}
; MMR6:   slt ${{[0-9]+}}
; MMR6:   <MCInst #{{[0-9]+}} SLT_MM

define void @slti_beq3(i32 %a) {
entry:
  %cmp = icmp slt i32 %a, 32768
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %a, ptr @g1, align 4
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: sltiu_beq0:
; CHECK:  sltiu $[[R0:[0-9]+]], $4, 32767
; MMR6:   sltiu $[[R0:[0-9]+]], $4, 32767
; MMR6:   <MCInst #{{[0-9]+}} SLTiu_MM
; CHECK:  beqz $[[R0]]

define void @sltiu_beq0(i32 %a) {
entry:
  %cmp = icmp ult i32 %a, 32767
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %a, ptr @g1, align 4
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: sltiu_beq1:
; CHECK:  sltu ${{[0-9]+}}
; MMR6:   sltu ${{[0-9]+}}
; MMR6:   <MCInst #{{[0-9]+}} SLTu_MM

define void @sltiu_beq1(i32 %a) {
entry:
  %cmp = icmp ult i32 %a, 32768
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %a, ptr @g1, align 4
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: sltiu_beq2:
; CHECK:  sltiu $[[R0:[0-9]+]], $4, -32768
; MMR6:   sltiu $[[R0:[0-9]+]], $4, -32768
; MMR6:   <MCInst #{{[0-9]+}} SLTiu_MM
; CHECK:  beqz $[[R0]]

define void @sltiu_beq2(i32 %a) {
entry:
  %cmp = icmp ult i32 %a, -32768
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %a, ptr @g1, align 4
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: sltiu_beq3:
; CHECK:  sltu ${{[0-9]+}}
; MMR6:   sltu ${{[0-9]+}}
; MMR6:   <MCInst #{{[0-9]+}} SLTu_MM

define void @sltiu_beq3(i32 %a) {
entry:
  %cmp = icmp ult i32 %a, -32769
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 %a, ptr @g1, align 4
  br label %if.end

if.end:
  ret void
}

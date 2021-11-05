; RUN: llc -mtriple=aarch64-none-eabi %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-none-eabi -global-isel=true -global-isel-abort=2 %s -o - | FileCheck %s


; Div whose result is unused should be removed unless we have strict exceptions

; CHECK-LABEL: unused_div:
; CHECK-NOT: fdiv
; CHECK: ret
define void @unused_div(float %x, float %y) #0 {
entry:
  %add = fdiv float %x, %y
  ret void
}

; CHECK-LABEL: unused_div_fpexcept_strict:
; CHECK: fdiv s0, s0, s1
; CHECK-NEXT: ret
define void @unused_div_fpexcept_strict(float %x, float %y) #0 {
entry:
  %add = call float @llvm.experimental.constrained.fdiv.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret void
}

; CHECK-LABEL: unused_div_round_dynamic:
; CHECK-NOT: fdiv
; CHECK: ret
define void @unused_div_round_dynamic(float %x, float %y) #0 {
entry:
  %add = call float @llvm.experimental.constrained.fdiv.f32(float %x, float %y, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  ret void
}


; Machine CSE should eliminate the second add unless we have strict exceptions

; CHECK-LABEL: add_twice:
; CHECK: fadd [[ADD:s[0-9]+]], s0, s1
; CHECK-NEXT: cmp w0, #0
; CHECK-NEXT: fmul [[MUL:s[0-9]+]], [[ADD]], [[ADD]]
; CHECK-NEXT: fcsel s0, [[ADD]], [[MUL]], eq
; CHECK-NEXT: ret
define float @add_twice(float %x, float %y, i32 %n) #0 {
entry:
  %add = fadd float %x, %y
  %tobool.not = icmp eq i32 %n, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:
  %add1 = fadd float %x, %y
  %mul = fmul float %add, %add1
  br label %if.end

if.end:
  %a.0 = phi float [ %mul, %if.then ], [ %add, %entry ]
  ret float %a.0
}

; CHECK-LABEL: add_twice_fpexcept_strict:
; CHECK: fmov [[X:s[0-9]+]], s0
; CHECK-NEXT: fadd s0, s0, s1
; CHECK-NEXT: cbz w0, [[LABEL:.LBB[0-9_]+]]
; CHECK: fadd [[ADD:s[0-9]+]], [[X]], s1
; CHECK-NEXT: fmul s0, s0, [[ADD]]
; CHECK: [[LABEL]]:
; CHECK-NEXT: ret
define float @add_twice_fpexcept_strict(float %x, float %y, i32 %n) #0 {
entry:
  %add = call float @llvm.experimental.constrained.fadd.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  %tobool.not = icmp eq i32 %n, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:
  %add1 = call float @llvm.experimental.constrained.fadd.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  %mul = call float @llvm.experimental.constrained.fmul.f32(float %add, float %add1, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  br label %if.end

if.end:
  %a.0 = phi float [ %mul, %if.then ], [ %add, %entry ]
  ret float %a.0
}

; CHECK-LABEL: add_twice_round_dynamic:
; CHECK: fadd s0, s0, s1
; CHECK-NEXT: cbz w0, [[LABEL:.LBB[0-9_]+]]
; CHECK-NOT: fadd
; CHECK: fmul s0, s0, s0
; CHECK: [[LABEL]]:
; CHECK-NEXT: ret
define float @add_twice_round_dynamic(float %x, float %y, i32 %n) #0 {
entry:
  %add = call float @llvm.experimental.constrained.fadd.f32(float %x, float %y, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  %tobool.not = icmp eq i32 %n, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:
  %add1 = call float @llvm.experimental.constrained.fadd.f32(float %x, float %y, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  %mul = call float @llvm.experimental.constrained.fmul.f32(float %add, float %add1, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  br label %if.end

if.end:
  %a.0 = phi float [ %mul, %if.then ], [ %add, %entry ]
  ret float %a.0
}


; Two adds separated by llvm.set.rounding should be preserved when rounding is
; dynamic (as they may give different results) or when we have strict exceptions
; (the llvm.set.rounding is irrelevant, but both could trap).

; CHECK-LABEL: set_rounding:
; CHECK-DAG: fadd [[SREG:s[0-9]+]], s0, s1
; CHECK-DAG: mrs [[XREG1:x[0-9]+]], FPCR
; CHECK-DAG: orr [[XREG2:x[0-9]+]], [[XREG1]], #0xc00000
; CHECK: msr FPCR, [[XREG2]]
; CHECK-NEXT: mrs [[XREG3:x[0-9]+]], FPCR
; CHECK-NEXT: and [[XREG4:x[0-9]+]], [[XREG3]], #0xffffffffff3fffff
; CHECK-NEXT: msr FPCR, [[XREG4]]
; CHECK-NEXT: fsub s0, [[SREG]], [[SREG]]
; CHECK-NEXT: ret
define float @set_rounding(float %x, float %y) #0 {
entry:
  %add1 = fadd float %x, %y
  call void @llvm.set.rounding(i32 0)
  %add2 = fadd float %x, %y
  call void @llvm.set.rounding(i32 1)
  %sub = fsub float %add1, %add2
  ret float %sub
}

; CHECK-LABEL: set_rounding_fpexcept_strict:
; CHECK-DAG: fadd [[SREG1:s[0-9]+]], s0, s1
; CHECK-DAG: mrs [[XREG1:x[0-9]+]], FPCR
; CHECK-DAG: orr [[XREG2:x[0-9]+]], [[XREG1]], #0xc00000
; CHECK: msr FPCR, [[XREG2]]
; CHECK-DAG: fadd [[SREG2:s[0-9]+]], s0, s1
; CHECK-DAG: mrs [[XREG3:x[0-9]+]], FPCR
; CHECK-DAG: and [[XREG4:x[0-9]+]], [[XREG3]], #0xffffffffff3fffff
; CHECK-NEXT: msr FPCR, [[XREG4]]
; CHECK-NEXT: fsub s0, [[SREG1]], [[SREG2]]
; CHECK-NEXT: ret
define float @set_rounding_fpexcept_strict(float %x, float %y) #0 {
entry:
  %add1 = call float @llvm.experimental.constrained.fadd.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  call void @llvm.set.rounding(i32 0)
  %add2 = call float @llvm.experimental.constrained.fadd.f32(float %x, float %y, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  call void @llvm.set.rounding(i32 1)
  %sub = call float @llvm.experimental.constrained.fsub.f32(float %add1, float %add2, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  ret float %sub
}

; CHECK-LABEL: set_rounding_round_dynamic:
; CHECK-DAG: fadd [[SREG1:s[0-9]+]], s0, s1
; CHECK-DAG: mrs [[XREG1:x[0-9]+]], FPCR
; CHECK-DAG: orr [[XREG2:x[0-9]+]], [[XREG1]], #0xc00000
; CHECK: msr FPCR, [[XREG2]]
; CHECK-DAG: fadd [[SREG2:s[0-9]+]], s0, s1
; CHECK-DAG: mrs [[XREG3:x[0-9]+]], FPCR
; CHECK-DAG: and [[XREG4:x[0-9]+]], [[XREG3]], #0xffffffffff3fffff
; CHECK-NEXT: msr FPCR, [[XREG4]]
; CHECK-NEXT: fsub s0, [[SREG1]], [[SREG2]]
; CHECK-NEXT: ret
define float @set_rounding_round_dynamic(float %x, float %y) #0 {
entry:
  %add1 = call float @llvm.experimental.constrained.fadd.f32(float %x, float %y, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  call void @llvm.set.rounding(i32 0)
  %add2 = call float @llvm.experimental.constrained.fadd.f32(float %x, float %y, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  call void @llvm.set.rounding(i32 1)
  %sub = call float @llvm.experimental.constrained.fsub.f32(float %add1, float %add2, metadata !"round.dynamic", metadata !"fpexcept.ignore") #0
  ret float %sub
}

declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata) #0
declare float @llvm.experimental.constrained.fsub.f32(float, float, metadata, metadata) #0
declare float @llvm.experimental.constrained.fmul.f32(float, float, metadata, metadata) #0
declare float @llvm.experimental.constrained.fdiv.f32(float, float, metadata, metadata) #0
declare i32 @llvm.flt.rounds()
declare void @llvm.set.rounding(i32)

attributes #0 = { "strictfp" }

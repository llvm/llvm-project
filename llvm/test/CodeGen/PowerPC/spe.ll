; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu \
; RUN:          -mattr=+spe |  FileCheck %s

declare float @llvm.fabs.float(float)
define float @test_float_abs(float %a) #0 {
  entry:
    %0 = tail call float @llvm.fabs.float(float %a)
    ret float %0
; CHECK-LABEL: test_float_abs
; CHECK: efsabs 3, 3
; CHECK: blr
}

define float @test_fnabs(float %a) #0 {
  entry:
    %0 = tail call float @llvm.fabs.float(float %a)
    %sub = fsub float -0.000000e+00, %0
    ret float %sub
; CHECK-LABEL: @test_fnabs
; CHECK: efsnabs
; CHECK: blr
}

define float @test_fdiv(float %a, float %b) {
entry:
  %v = fdiv float %a, %b
  ret float %v

; CHECK-LABEL: test_fdiv
; CHECK: efsdiv
; CHECK: blr
}

define float @test_fmul(float %a, float %b) {
  entry:
  %v = fmul float %a, %b
  ret float %v
; CHECK-LABEL @test_fmul
; CHECK: efsmul
; CHECK: blr
}

define float @test_fadd(float %a, float %b) {
  entry:
  %v = fadd float %a, %b
  ret float %v
; CHECK-LABEL @test_fadd
; CHECK: efsadd
; CHECK: blr
}

define float @test_fsub(float %a, float %b) {
  entry:
  %v = fsub float %a, %b
  ret float %v
; CHECK-LABEL @test_fsub
; CHECK: efssub
; CHECK: blr
}

define float @test_fneg(float %a) {
  entry:
  %v = fsub float -0.0, %a
  ret float %v

; CHECK-LABEL @test_fneg
; CHECK: efsneg
; CHECK: blr
}

define float @test_dtos(double %a) {
  entry:
  %v = fptrunc double %a to float
  ret float %v
; CHECK-LABEL: test_dtos
; CHECK: efscfd
; CHECK: blr
}

define i1 @test_fcmpgt(float %a, float %b) {
  entry:
  %r = fcmp ogt float %a, %b
  ret i1 %r
; CHECK-LABEL: test_fcmpgt
; CHECK: efscmpgt
; CHECK: blr
}

define i1 @test_fcmpugt(float %a, float %b) {
  entry:
  %r = fcmp ugt float %a, %b
  ret i1 %r
; CHECK-LABEL: test_fcmpugt
; CHECK: efscmpgt
; CHECK: blr
}

define i1 @test_fcmple(float %a, float %b) {
  entry:
  %r = fcmp ole float %a, %b
  ret i1 %r
; CHECK-LABEL: test_fcmple
; CHECK: efscmpgt
; CHECK: blr
}

define i1 @test_fcmpule(float %a, float %b) {
  entry:
  %r = fcmp ule float %a, %b
  ret i1 %r
; CHECK-LABEL: test_fcmpule
; CHECK: efscmpgt
; CHECK: blr
}

define i1 @test_fcmpeq(float %a, float %b) {
  entry:
  %r = fcmp oeq float %a, %b
  ret i1 %r
; CHECK-LABEL: test_fcmpeq
; CHECK: efscmpeq
; CHECK: blr
}

; (un)ordered tests are expanded to une and oeq so verify
define i1 @test_fcmpuno(float %a, float %b) {
  entry:
  %r = fcmp uno float %a, %b
  ret i1 %r
; CHECK-LABEL: test_fcmpuno
; CHECK: efscmpeq
; CHECK: efscmpeq
; CHECK: crand
; CHECK: blr
}

define i1 @test_fcmpord(float %a, float %b) {
  entry:
  %r = fcmp ord float %a, %b
  ret i1 %r
; CHECK-LABEL: test_fcmpord
; CHECK: efscmpeq
; CHECK: efscmpeq
; CHECK: crnand
; CHECK: blr
}

define i1 @test_fcmpueq(float %a, float %b) {
  entry:
  %r = fcmp ueq float %a, %b
  ret i1 %r
; CHECK-LABEL: test_fcmpueq
; CHECK: efscmpeq
; CHECK: blr
}

define i1 @test_fcmpne(float %a, float %b) {
  entry:
  %r = fcmp one float %a, %b
  ret i1 %r
; CHECK-LABEL: test_fcmpne
; CHECK: efscmpeq
; CHECK: blr
}

define i1 @test_fcmpune(float %a, float %b) {
  entry:
  %r = fcmp une float %a, %b
  ret i1 %r
; CHECK-LABEL: test_fcmpune
; CHECK: efscmpeq
; CHECK: blr
}

define i1 @test_fcmplt(float %a, float %b) {
  entry:
  %r = fcmp olt float %a, %b
  ret i1 %r
; CHECK-LABEL: test_fcmplt
; CHECK: efscmplt
; CHECK: blr
}

define i1 @test_fcmpult(float %a, float %b) {
  entry:
  %r = fcmp ult float %a, %b
  ret i1 %r
; CHECK-LABEL: test_fcmpult
; CHECK: efscmplt
; CHECK: blr
}

define i1 @test_fcmpge(float %a, float %b) {
  entry:
  %r = fcmp oge float %a, %b
  ret i1 %r
; CHECK-LABEL: test_fcmpge
; CHECK: efscmplt
; CHECK: blr
}

define i1 @test_fcmpuge(float %a, float %b) {
  entry:
  %r = fcmp uge float %a, %b
  ret i1 %r
; CHECK-LABEL: test_fcmpuge
; CHECK: efscmplt
; CHECK: blr
}

define i32 @test_ftoui(float %a) {
  %v = fptoui float %a to i32
  ret i32 %v
; CHECK-LABEL: test_ftoui
; CHECK: efsctuiz
}

define i32 @test_ftosi(float %a) {
  %v = fptosi float %a to i32
  ret i32 %v
; CHECK-LABEL: test_ftosi
; CHECK: efsctsiz
}

define float @test_ffromui(i32 %a) {
  %v = uitofp i32 %a to float
  ret float %v
; CHECK-LABEL: test_ffromui
; CHECK: efscfui
}

define float @test_ffromsi(i32 %a) {
  %v = sitofp i32 %a to float
  ret float %v
; CHECK-LABEL: test_ffromsi
; CHECK: efscfsi
}

define i32 @test_fasmconst(float %x) {
entry:
  %x.addr = alloca float, align 8
  store float %x, float* %x.addr, align 8
  %0 = load float, float* %x.addr, align 8
  %1 = call i32 asm sideeffect "efsctsi $0, $1", "=f,f"(float %0)
  ret i32 %1
; CHECK-LABEL: test_fasmconst
; Check that it's not loading a double
; CHECK-NOT: evldd
; CHECK: #APP
; CHECK: efsctsi
; CHECK: #NO_APP
}

; Double tests

define void @test_double_abs(double * %aa) #0 {
  entry:
    %0 = load double, double * %aa
    %1 = tail call double @llvm.fabs.f64(double %0) #2
    store double %1, double * %aa
    ret void
; CHECK-LABEL: test_double_abs
; CHECK: efdabs
; CHECK: blr
}

; Function Attrs: nounwind readnone
declare double @llvm.fabs.f64(double) #1

define void @test_dnabs(double * %aa) #0 {
  entry:
    %0 = load double, double * %aa
    %1 = tail call double @llvm.fabs.f64(double %0) #2
    %sub = fsub double -0.000000e+00, %1
    store double %sub, double * %aa
    ret void
}
; CHECK-LABEL: @test_dnabs
; CHECK: efdnabs
; CHECK: blr

define double @test_ddiv(double %a, double %b) {
entry:
  %v = fdiv double %a, %b
  ret double %v

; CHECK-LABEL: test_ddiv
; CHECK: efddiv
; CHECK: blr
}

define double @test_dmul(double %a, double %b) {
  entry:
  %v = fmul double %a, %b
  ret double %v
; CHECK-LABEL @test_dmul
; CHECK: efdmul
; CHECK: blr
}

define double @test_dadd(double %a, double %b) {
  entry:
  %v = fadd double %a, %b
  ret double %v
; CHECK-LABEL @test_dadd
; CHECK: efdadd
; CHECK: blr
}

define double @test_dsub(double %a, double %b) {
  entry:
  %v = fsub double %a, %b
  ret double %v
; CHECK-LABEL @test_dsub
; CHECK: efdsub
; CHECK: blr
}

define double @test_dneg(double %a) {
  entry:
  %v = fsub double -0.0, %a
  ret double %v

; CHECK-LABEL @test_dneg
; CHECK: blr
}

define double @test_stod(float %a) {
  entry:
  %v = fpext float %a to double
  ret double %v
; CHECK-LABEL: test_stod
; CHECK: efdcfs
; CHECK: blr
}

; (un)ordered tests are expanded to une and oeq so verify
define i1 @test_dcmpuno(double %a, double %b) {
  entry:
  %r = fcmp uno double %a, %b
  ret i1 %r
; CHECK-LABEL: test_dcmpuno
; CHECK: efdcmpeq
; CHECK: efdcmpeq
; CHECK: crand
; CHECK: blr
}

define i1 @test_dcmpord(double %a, double %b) {
  entry:
  %r = fcmp ord double %a, %b
  ret i1 %r
; CHECK-LABEL: test_dcmpord
; CHECK: efdcmpeq
; CHECK: efdcmpeq
; CHECK: crnand
; CHECK: blr
}

define i1 @test_dcmpgt(double %a, double %b) {
  entry:
  %r = fcmp ogt double %a, %b
  ret i1 %r
; CHECK-LABEL: test_dcmpgt
; CHECK: efdcmpgt
; CHECK: blr
}

define i1 @test_dcmpugt(double %a, double %b) {
  entry:
  %r = fcmp ugt double %a, %b
  ret i1 %r
; CHECK-LABEL: test_dcmpugt
; CHECK: efdcmpgt
; CHECK: blr
}

define i1 @test_dcmple(double %a, double %b) {
  entry:
  %r = fcmp ole double %a, %b
  ret i1 %r
; CHECK-LABEL: test_dcmple
; CHECK: efdcmpgt
; CHECK: blr
}

define i1 @test_dcmpule(double %a, double %b) {
  entry:
  %r = fcmp ule double %a, %b
  ret i1 %r
; CHECK-LABEL: test_dcmpule
; CHECK: efdcmpgt
; CHECK: blr
}

define i1 @test_dcmpeq(double %a, double %b) {
  entry:
  %r = fcmp oeq double %a, %b
  ret i1 %r
; CHECK-LABEL: test_dcmpeq
; CHECK: efdcmpeq
; CHECK: blr
}

define i1 @test_dcmpueq(double %a, double %b) {
  entry:
  %r = fcmp ueq double %a, %b
  ret i1 %r
; CHECK-LABEL: test_dcmpueq
; CHECK: efdcmpeq
; CHECK: blr
}

define i1 @test_dcmpne(double %a, double %b) {
  entry:
  %r = fcmp one double %a, %b
  ret i1 %r
; CHECK-LABEL: test_dcmpne
; CHECK: efdcmpeq
; CHECK: blr
}

define i1 @test_dcmpune(double %a, double %b) {
  entry:
  %r = fcmp une double %a, %b
  ret i1 %r
; CHECK-LABEL: test_dcmpune
; CHECK: efdcmpeq
; CHECK: blr
}

define i1 @test_dcmplt(double %a, double %b) {
  entry:
  %r = fcmp olt double %a, %b
  ret i1 %r
; CHECK-LABEL: test_dcmplt
; CHECK: efdcmplt
; CHECK: blr
}

define i1 @test_dcmpult(double %a, double %b) {
  entry:
  %r = fcmp ult double %a, %b
  ret i1 %r
; CHECK-LABEL: test_dcmpult
; CHECK: efdcmplt
; CHECK: blr
}

define i1 @test_dcmpge(double %a, double %b) {
  entry:
  %r = fcmp oge double %a, %b
  ret i1 %r
; CHECK-LABEL: test_dcmpge
; CHECK: efdcmplt
; CHECK: blr
}

define i1 @test_dcmpuge(double %a, double %b) {
  entry:
  %r = fcmp uge double %a, %b
  ret i1 %r
; CHECK-LABEL: test_dcmpuge
; CHECK: efdcmplt
; CHECK: blr
}

define double @test_dselect(double %a, double %b, i1 %c) {
entry:
  %r = select i1 %c, double %a, double %b
  ret double %r
; CHECK-LABEL: test_dselect
; CHECK: andi.
; CHECK: bc
; CHECK: evldd
; CHECK: b
; CHECK: evldd
; CHECK: evstdd
; CHECK: blr
}

define i32 @test_dtoui(double %a) {
entry:
  %v = fptoui double %a to i32
  ret i32 %v
; CHECK-LABEL: test_dtoui
; CHECK: efdctuiz
}

define i32 @test_dtosi(double %a) {
entry:
  %v = fptosi double %a to i32
  ret i32 %v
; CHECK-LABEL: test_dtosi
; CHECK: efdctsiz
}

define double @test_dfromui(i32 %a) {
entry:
  %v = uitofp i32 %a to double
  ret double %v
; CHECK-LABEL: test_dfromui
; CHECK: efdcfui
}

define double @test_dfromsi(i32 %a) {
entry:
  %v = sitofp i32 %a to double
  ret double %v
; CHECK-LABEL: test_dfromsi
; CHECK: efdcfsi
}

define i32 @test_dasmconst(double %x) {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  %0 = load double, double* %x.addr, align 8
  %1 = call i32 asm sideeffect "efdctsi $0, $1", "=d,d"(double %0)
  ret i32 %1
; CHECK-LABEL: test_dasmconst
; CHECK: evldd
; CHECK: #APP
; CHECK: efdctsi
; CHECK: #NO_APP
}

define double @test_spill(double %a) nounwind {
entry:
  %0 = fadd double %a, %a
  call void asm sideeffect "","~{r0},~{r3},~{s4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31}"() nounwind
  %1 = fadd double %0, 3.14159
  br label %return

return:
  ret double %1

; CHECK-LABEL: test_spill
; CHECK: efdadd
; CHECK: evstdd
; CHECK: evldd
}

; RUN: llc -mtriple=s390x-linux-gnu < %s | FileCheck -check-prefix=LINUX %s
; RUN: llc -mtriple=s390x-ibm-zos < %s | FileCheck -check-prefix=ZOS %s

; FIXME: half cases split out since they are broken on zos

; FIXME: Check ZOS function content

define { float, float } @test_sincos_f32(float %a) #0 {
; LINUX-LABEL: test_sincos_f32:
; LINUX:       # %bb.0:
; LINUX-NEXT:    stmg %r14, %r15, 112(%r15)
; LINUX-NEXT:    aghi %r15, -168
; LINUX-NEXT:    la %r2, 164(%r15)
; LINUX-NEXT:    la %r3, 160(%r15)
; LINUX-NEXT:    brasl %r14, sincosf@PLT
; LINUX-NEXT:    le %f0, 164(%r15)
; LINUX-NEXT:    le %f2, 160(%r15)
; LINUX-NEXT:    lmg %r14, %r15, 280(%r15)
; LINUX-NEXT:    br %r14
  %result = call { float, float } @llvm.sincos.f32(float %a)
  ret { float, float } %result
}

define { <2 x float>, <2 x float> } @test_sincos_v2f32(<2 x float> %a) #0 {
; LINUX-LABEL: test_sincos_v2f32:
; LINUX:       # %bb.0:
; LINUX-NEXT:    stmg %r14, %r15, 112(%r15)
; LINUX-NEXT:    aghi %r15, -184
; LINUX-NEXT:    std %f8, 176(%r15) # 8-byte Spill
; LINUX-NEXT:    la %r2, 164(%r15)
; LINUX-NEXT:    la %r3, 160(%r15)
; LINUX-NEXT:    ler %f8, %f2
; LINUX-NEXT:    brasl %r14, sincosf@PLT
; LINUX-NEXT:    la %r2, 172(%r15)
; LINUX-NEXT:    la %r3, 168(%r15)
; LINUX-NEXT:    ler %f0, %f8
; LINUX-NEXT:    brasl %r14, sincosf@PLT
; LINUX-NEXT:    le %f0, 164(%r15)
; LINUX-NEXT:    le %f2, 172(%r15)
; LINUX-NEXT:    le %f4, 160(%r15)
; LINUX-NEXT:    le %f6, 168(%r15)
; LINUX-NEXT:    ld %f8, 176(%r15) # 8-byte Reload
; LINUX-NEXT:    lmg %r14, %r15, 296(%r15)
; LINUX-NEXT:    br %r14
  %result = call { <2 x float>, <2 x float> } @llvm.sincos.v2f32(<2 x float> %a)
  ret { <2 x float>, <2 x float> } %result
}

define { <3 x float>, <3 x float> } @test_sincos_v3f32(<3 x float> %a) #0 {
; LINUX-LABEL: test_sincos_v3f32:
; LINUX:       # %bb.0:
; LINUX-NEXT:    stmg %r13, %r15, 104(%r15)
; LINUX-NEXT:    aghi %r15, -192
; LINUX-NEXT:    std %f8, 184(%r15) # 8-byte Spill
; LINUX-NEXT:    std %f9, 176(%r15) # 8-byte Spill
; LINUX-NEXT:    lgr %r13, %r2
; LINUX-NEXT:    la %r2, 164(%r15)
; LINUX-NEXT:    la %r3, 160(%r15)
; LINUX-NEXT:    ler %f8, %f4
; LINUX-NEXT:    ler %f9, %f2
; LINUX-NEXT:    brasl %r14, sincosf@PLT
; LINUX-NEXT:    la %r2, 172(%r15)
; LINUX-NEXT:    la %r3, 168(%r15)
; LINUX-NEXT:    ler %f0, %f9
; LINUX-NEXT:    brasl %r14, sincosf@PLT
; LINUX-NEXT:    la %r2, 8(%r13)
; LINUX-NEXT:    la %r3, 24(%r13)
; LINUX-NEXT:    ler %f0, %f8
; LINUX-NEXT:    brasl %r14, sincosf@PLT
; LINUX-NEXT:    le %f0, 164(%r15)
; LINUX-NEXT:    le %f1, 172(%r15)
; LINUX-NEXT:    le %f2, 160(%r15)
; LINUX-NEXT:    lgdr %r0, %f0
; LINUX-NEXT:    lgdr %r1, %f1
; LINUX-NEXT:    lgdr %r2, %f2
; LINUX-NEXT:    le %f0, 168(%r15)
; LINUX-NEXT:    nilf %r0, 0
; LINUX-NEXT:    srlg %r1, %r1, 32
; LINUX-NEXT:    nilf %r2, 0
; LINUX-NEXT:    lgdr %r3, %f0
; LINUX-NEXT:    srlg %r3, %r3, 32
; LINUX-NEXT:    lr %r0, %r1
; LINUX-NEXT:    lr %r2, %r3
; LINUX-NEXT:    stg %r2, 16(%r13)
; LINUX-NEXT:    stg %r0, 0(%r13)
; LINUX-NEXT:    ld %f8, 184(%r15) # 8-byte Reload
; LINUX-NEXT:    ld %f9, 176(%r15) # 8-byte Reload
; LINUX-NEXT:    lmg %r13, %r15, 296(%r15)
; LINUX-NEXT:    br %r14
  %result = call { <3 x float>, <3 x float> } @llvm.sincos.v3f32(<3 x float> %a)
  ret { <3 x float>, <3 x float> } %result
}

define { double, double } @test_sincos_f64(double %a) #0 {
; LINUX-LABEL: test_sincos_f64:
; LINUX:       # %bb.0:
; LINUX-NEXT:    stmg %r14, %r15, 112(%r15)
; LINUX-NEXT:    aghi %r15, -176
; LINUX-NEXT:    la %r2, 168(%r15)
; LINUX-NEXT:    la %r3, 160(%r15)
; LINUX-NEXT:    brasl %r14, sincos@PLT
; LINUX-NEXT:    ld %f0, 168(%r15)
; LINUX-NEXT:    ld %f2, 160(%r15)
; LINUX-NEXT:    lmg %r14, %r15, 288(%r15)
; LINUX-NEXT:    br %r14
  %result = call { double, double } @llvm.sincos.f64(double %a)
  ret { double, double } %result
}

define { <2 x double>, <2 x double> } @test_sincos_v2f64(<2 x double> %a) #0 {
; LINUX-LABEL: test_sincos_v2f64:
; LINUX:       # %bb.0:
; LINUX-NEXT:    stmg %r14, %r15, 112(%r15)
; LINUX-NEXT:    aghi %r15, -200
; LINUX-NEXT:    std %f8, 192(%r15) # 8-byte Spill
; LINUX-NEXT:    la %r2, 168(%r15)
; LINUX-NEXT:    la %r3, 160(%r15)
; LINUX-NEXT:    ldr %f8, %f2
; LINUX-NEXT:    brasl %r14, sincos@PLT
; LINUX-NEXT:    la %r2, 184(%r15)
; LINUX-NEXT:    la %r3, 176(%r15)
; LINUX-NEXT:    ldr %f0, %f8
; LINUX-NEXT:    brasl %r14, sincos@PLT
; LINUX-NEXT:    ld %f0, 168(%r15)
; LINUX-NEXT:    ld %f2, 184(%r15)
; LINUX-NEXT:    ld %f4, 160(%r15)
; LINUX-NEXT:    ld %f6, 176(%r15)
; LINUX-NEXT:    ld %f8, 192(%r15) # 8-byte Reload
; LINUX-NEXT:    lmg %r14, %r15, 312(%r15)
; LINUX-NEXT:    br %r14
  %result = call { <2 x double>, <2 x double> } @llvm.sincos.v2f64(<2 x double> %a)
  ret { <2 x double>, <2 x double> } %result
}

define { fp128, fp128 } @test_sincos_f128(fp128 %a) #0 {
; LINUX-LABEL: test_sincos_f128:
; LINUX:       # %bb.0:
; LINUX-NEXT:    stmg %r14, %r15, 112(%r15)
; LINUX-NEXT:    aghi %r15, -176
; LINUX-NEXT:    ld %f0, 0(%r3)
; LINUX-NEXT:    ld %f2, 8(%r3)
; LINUX-NEXT:    lgr %r3, %r2
; LINUX-NEXT:    la %r4, 16(%r2)
; LINUX-NEXT:    la %r2, 160(%r15)
; LINUX-NEXT:    std %f0, 160(%r15)
; LINUX-NEXT:    std %f2, 168(%r15)
; LINUX-NEXT:    brasl %r14, sincosl@PLT
; LINUX-NEXT:    lmg %r14, %r15, 288(%r15)
; LINUX-NEXT:    br %r14
  %result = call { fp128, fp128 } @llvm.sincos.f128(fp128 %a)
  ret { fp128, fp128 } %result
}

define { <2 x fp128>, <2 x fp128> } @test_sincos_v2f128(<2 x fp128> %a) #0 {
; LINUX-LABEL: test_sincos_v2f128:
; LINUX:       # %bb.0:
; LINUX-NEXT:    stmg %r13, %r15, 104(%r15)
; LINUX-NEXT:    aghi %r15, -208
; LINUX-NEXT:    std %f8, 200(%r15) # 8-byte Spill
; LINUX-NEXT:    std %f10, 192(%r15) # 8-byte Spill
; LINUX-NEXT:    lgr %r13, %r2
; LINUX-NEXT:    ld %f8, 0(%r3)
; LINUX-NEXT:    ld %f10, 8(%r3)
; LINUX-NEXT:    ld %f0, 16(%r3)
; LINUX-NEXT:    ld %f2, 24(%r3)
; LINUX-NEXT:    la %r2, 176(%r15)
; LINUX-NEXT:    la %r3, 16(%r13)
; LINUX-NEXT:    la %r4, 48(%r13)
; LINUX-NEXT:    std %f0, 176(%r15)
; LINUX-NEXT:    std %f2, 184(%r15)
; LINUX-NEXT:    brasl %r14, sincosl@PLT
; LINUX-NEXT:    la %r4, 32(%r13)
; LINUX-NEXT:    la %r2, 160(%r15)
; LINUX-NEXT:    std %f8, 160(%r15)
; LINUX-NEXT:    std %f10, 168(%r15)
; LINUX-NEXT:    lgr %r3, %r13
; LINUX-NEXT:    brasl %r14, sincosl@PLT
; LINUX-NEXT:    ld %f8, 200(%r15) # 8-byte Reload
; LINUX-NEXT:    ld %f10, 192(%r15) # 8-byte Reload
; LINUX-NEXT:    lmg %r13, %r15, 312(%r15)
; LINUX-NEXT:    br %r14
  %result = call { <2 x fp128>, <2 x fp128> } @llvm.sincos.v2f128(<2 x fp128> %a)
  ret { <2 x fp128>, <2 x fp128> } %result
}


; ZOS: .quad	R(@@FSIN@B)                     * Offset 0 function descriptor of @@FSIN@B
; ZOS: .quad	V(@@FSIN@B)
; ZOS: .quad	R(@@FCOS@B)                     * Offset 16 function descriptor of @@FCOS@B
; ZOS: .quad	V(@@FCOS@B)
; ZOS: .quad	R(@@SSIN@B)                     * Offset 32 function descriptor of @@SSIN@B
; ZOS: .quad	V(@@SSIN@B)
; ZOS: .quad	R(@@SCOS@B)                     * Offset 48 function descriptor of @@SCOS@B
; ZOS: .quad	V(@@SCOS@B)
; ZOS: .quad	R(@@LSIN@B)                     * Offset 64 function descriptor of @@LSIN@B
; ZOS: .quad	V(@@LSIN@B)
; ZOS: .quad	R(@@LCOS@B)                     * Offset 80 function descriptor of @@LCOS@B
; ZOS: .quad	V(@@LCOS@B)


attributes #0 = { nounwind }

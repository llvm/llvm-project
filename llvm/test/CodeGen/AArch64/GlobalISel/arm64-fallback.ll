; RUN: llc -O0 -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' -verify-machineinstrs %s -o %t.out 2> %t.err
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-OUT < %t.out
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-ERR < %t.err
; RUN: not --crash llc -global-isel -mtriple aarch64_be %s -o - 2>&1 | FileCheck %s --check-prefix=BIG-ENDIAN
; This file checks that the fallback path to selection dag works.
; The test is fragile in the sense that it must be updated to expose
; something that fails with global-isel.
; When we cannot produce a test case anymore, that means we can remove
; the fallback path.

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--"

; BIG-ENDIAN: unable to translate in big endian mode

; Make sure we don't mess up metadata arguments.
declare void @llvm.write_register.i64(metadata, i64)

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: G_WRITE_REGISTER !0, %0:_(s64) (in function: test_write_register_intrin)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for test_write_register_intrin
; FALLBACK-WITH-REPORT-LABEL: test_write_register_intrin:
define void @test_write_register_intrin() {
  call void @llvm.write_register.i64(metadata !{!"sp"}, i64 0)
  ret void
}

@_ZTIi = external global i8*
declare i32 @__gxx_personality_v0(...)

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: %2:_(<2 x p0>) = G_INSERT_VECTOR_ELT %0:_, %{{[0-9]+}}:_(p0), %{{[0-9]+}}:_(s32) (in function: vector_of_pointers_insertelement)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for vector_of_pointers_insertelement
; FALLBACK-WITH-REPORT-OUT-LABEL: vector_of_pointers_insertelement:
define void @vector_of_pointers_insertelement() {
  br label %end

block:
  %dummy = insertelement <2 x i16*> %vec, i16* null, i32 0
  store <2 x i16*> %dummy, <2 x i16*>* undef
  ret void

end:
  %vec = load <2 x i16*>, <2 x i16*>* undef
  br label %block
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: cannot select: RET_ReallyLR implicit $x0 (in function: strict_align_feature)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for strict_align_feature
; FALLBACK-WITH-REPORT-OUT-LABEL: strict_align_feature
define i64 @strict_align_feature(i64* %p) #0 {
  %x = load i64, i64* %p, align 1
  ret i64 %x
}

attributes #0 = { "target-features"="+strict-align" }

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction: call
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for direct_mem
; FALLBACK-WITH-REPORT-OUT-LABEL: direct_mem
define void @direct_mem(i32 %x, i32 %y) {
entry:
  tail call void asm sideeffect "", "imr,imr,~{memory}"(i32 %x, i32 %y)
  ret void
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to lower function{{.*}}scalable_arg
; FALLBACK-WITH-REPORT-OUT-LABEL: scalable_arg
define <vscale x 16 x i8> @scalable_arg(<vscale x 16 x i1> %pred, i8* %addr) #1 {
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1> %pred, i8* %addr)
  ret <vscale x 16 x i8> %res
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to lower function{{.*}}scalable_ret
; FALLBACK-WITH-REPORT-OUT-LABEL: scalable_ret
define <vscale x 16 x i8> @scalable_ret(i8* %addr) #1 {
  %pred = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 0)
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1> %pred, i8* %addr)
  ret <vscale x 16 x i8> %res
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction{{.*}}scalable_call
; FALLBACK-WITH-REPORT-OUT-LABEL: scalable_call
define i8 @scalable_call(i8* %addr) #1 {
  %pred = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 0)
  %vec = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1> %pred, i8* %addr)
  %res = extractelement <vscale x 16 x i8> %vec, i32 0
  ret i8 %res
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction{{.*}}scalable_alloca
; FALLBACK-WITH-REPORT-OUT-LABEL: scalable_alloca
define void @scalable_alloca() #1 {
  %local0 = alloca <vscale x 16 x i8>
  load volatile <vscale x 16 x i8>, <vscale x 16 x i8>* %local0
  ret void
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction{{.*}}asm_indirect_output
; FALLBACK-WITH-REPORT-OUT-LABEL: asm_indirect_output
define void @asm_indirect_output() {
entry:
  %ap = alloca i8*, align 8
  %0 = load i8*, i8** %ap, align 8
  call void asm sideeffect "", "=*r|m,0,~{memory}"(i8** elementtype(i8*) %ap, i8* %0)
  ret void
}

%struct.foo = type { [8 x i64] }

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction:{{.*}}ld64b{{.*}}asm_output_ls64
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for asm_output_ls64
; FALLBACK-WITH-REPORT-OUT-LABEL: asm_output_ls64
define void @asm_output_ls64(%struct.foo* %output, i8* %addr) #2 {
entry:
  %val = call i512 asm sideeffect "ld64b $0,[$1]", "=r,r,~{memory}"(i8* %addr)
  %outcast = bitcast %struct.foo* %output to i512*
  store i512 %val, i512* %outcast, align 8
  ret void
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction:{{.*}}st64b{{.*}}asm_input_ls64
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for asm_input_ls64
; FALLBACK-WITH-REPORT-OUT-LABEL: asm_input_ls64
define void @asm_input_ls64(%struct.foo* %input, i8* %addr) #2 {
entry:
  %incast = bitcast %struct.foo* %input to i512*
  %val = load i512, i512* %incast, align 8
  call void asm sideeffect "st64b $0,[$1]", "r,r,~{memory}"(i512 %val, i8* %addr)
  ret void
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: %4:_(s128), %5:_(s1) = G_UMULO %0:_, %6:_ (in function: umul_s128)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for umul_s128
; FALLBACK-WITH-REPORT-OUT-LABEL: umul_s128
declare {i128, i1} @llvm.umul.with.overflow.i128(i128, i128) nounwind readnone
define zeroext i1 @umul_s128(i128 %v1, i128* %res) {
entry:
  %t = call {i128, i1} @llvm.umul.with.overflow.i128(i128 %v1, i128 2)
  %val = extractvalue {i128, i1} %t, 0
  %obit = extractvalue {i128, i1} %t, 1
  store i128 %val, i128* %res
  ret i1 %obit
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction: {{.*}}llvm.experimental.gc.statepoint{{.*}} (in function: gc_intr)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for gc_intr
; FALLBACK-WITH-REPORT-OUT-LABEL: gc_intr

declare token @llvm.experimental.gc.statepoint.p0(i64 immarg, i32 immarg, i32()*, i32 immarg, i32 immarg, ...)
declare i32 @llvm.experimental.gc.result(token)

declare i32 @extern_returning_i32()

define i32 @gc_intr() gc "statepoint-example" {
   %statepoint_token = call token (i64, i32, i32()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 2882400000, i32 0, i32()* elementtype(i32 ()) @extern_returning_i32, i32 0, i32 0, i32 0, i32 0) [ "deopt"() ]
   %ret = call i32 (token) @llvm.experimental.gc.result(token %statepoint_token)
   ret i32 %ret
}

attributes #1 = { "target-features"="+sve" }
attributes #2 = { "target-features"="+ls64" }

declare <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 %pattern)
declare <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1>, i8*)

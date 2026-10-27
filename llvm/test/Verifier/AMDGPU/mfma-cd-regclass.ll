; RUN: split-file %s %t
; RUN: not opt -passes=verify -disable-output %t/function.ll 2>&1 | FileCheck %s --check-prefix=SITE
; RUN: not opt -passes=verify -disable-output %t/call.ll 2>&1 | FileCheck %s --check-prefix=SITE
; RUN: not opt -passes=verify -disable-output %t/parameter.ll 2>&1 | FileCheck %s --check-prefix=PARAM
; RUN: not llvm-as %t/unknown.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VALUE
; RUN: not llvm-as %t/integer.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VALUE
;
; SITE: cd_regclass is only valid on MFMA call sites
; PARAM: this attribute does not apply to parameters
; VALUE: expected MFMA C/D register class: auto, agpr, or vgpr

;--- function.ll
define void @bad() cd_regclass(agpr) {
  ret void
}

;--- call.ll
declare float @llvm.sqrt.f32(float)
define float @bad(float %x) {
  %d = call float @llvm.sqrt.f32(float %x) cd_regclass(vgpr)
  ret float %d
}

;--- parameter.ll
declare void @bad(i32 cd_regclass(agpr))

;--- unknown.ll
declare void @bad() cd_regclass(unknown)

;--- integer.ll
declare void @bad() cd_regclass(1)

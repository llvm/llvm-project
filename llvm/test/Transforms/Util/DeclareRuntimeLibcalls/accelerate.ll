; REQUIRES: aarch64-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=arm64-apple-macosx -vector-library=Accelerate < %s | FileCheck %s

; CHECK: declare <4 x float> @vacosf(<4 x float>) [[ATTRS:#[0-9]+]]
; CHECK: declare <4 x float> @vacoshf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vasinf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vasinhf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vatan2f(<4 x float>, <4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vatanf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vatanhf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vceilf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vcosf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vcoshf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vexpf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vexpm1f(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vfabsf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vfloorf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vlog10f(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vlog1pf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vlogbf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vlogf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vsinf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vsinhf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vsqrtf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vtanf(<4 x float>) [[ATTRS]]
; CHECK: declare <4 x float> @vtanhf(<4 x float>) [[ATTRS]]

; CHECK: attributes [[ATTRS]] = { nocallback nofree nosync nounwind willreturn }

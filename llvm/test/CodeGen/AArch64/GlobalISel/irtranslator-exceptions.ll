; RUN: llc -O0 -mtriple=aarch64-apple-ios -global-isel -stop-after=irtranslator %s -o - | FileCheck %s

@_ZTIi = external global i8*

declare i32 @foo(i32)
declare i32 @__gxx_personality_v0(...)
declare i32 @llvm.eh.typeid.for(i8*)

; CHECK: name: bar
; CHECK: body:
; CHECK:   bb.0:
; CHECK:     successors: %bb.2{{.*}}%bb.1
; CHECK:     EH_LABEL
; CHECK:     %w0 = COPY
; CHECK:     BL @foo, csr_aarch64_aapcs, implicit-def %lr, implicit %sp, implicit %w0, implicit-def %w0
; CHECK:     {{%[0-9]+}}(s32) = COPY %w0
; CHECK:     EH_LABEL

; CHECK:   bb.1
; CHECK:     EH_LABEL
; CHECK:     [[PTR:%[0-9]+]](p0) = COPY %x0
; CHECK:     [[SEL:%[0-9]+]](p0) = COPY %x1
; CHECK:     [[PTR_SEL:%[0-9]+]](s128) = G_SEQUENCE [[PTR]](p0), 0, [[SEL]](p0), 64
; CHECK:     [[PTR_RET:%[0-9]+]](s64), [[SEL_RET:%[0-9]+]](s32) = G_EXTRACT [[PTR_SEL]](s128), 0, 64
; CHECK:     %x0 = COPY [[PTR_RET]]
; CHECK:     %w1 = COPY [[SEL_RET]]

; CHECK:   bb.2:
; CHECK:     [[SEL:%[0-9]+]](s32) = G_CONSTANT i32 1
; CHECK:     {{%[0-9]+}}(s128) = G_INSERT {{%[0-9]+}}(s128), [[SEL]](s32), 64

define { i8*, i32 } @bar() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %res32 = invoke i32 @foo(i32 42) to label %continue unwind label %broken


broken:
  %ptr.sel = landingpad { i8*, i32 } catch i8* bitcast(i8** @_ZTIi to i8*)
  ret { i8*, i32 } %ptr.sel

continue:
  %sel.int = tail call i32 @llvm.eh.typeid.for(i8* bitcast(i8** @_ZTIi to i8*))
  %res.good = insertvalue { i8*, i32 } undef, i32 %sel.int, 1
  ret { i8*, i32 } %res.good
}

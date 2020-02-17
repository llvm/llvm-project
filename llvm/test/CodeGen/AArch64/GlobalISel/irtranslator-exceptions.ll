; RUN: llc -O0 -mtriple=aarch64-apple-ios -global-isel -stop-after=irtranslator %s -o - | FileCheck %s

@_ZTIi = external global i8*

declare i32 @foo(i32)
declare i32 @__gxx_personality_v0(...)
declare i32 @llvm.eh.typeid.for(i8*)

; CHECK-LABEL: name: bar
; CHECK: body:
; CHECK-NEXT:   bb.1 (%ir-block.0):
; CHECK:     successors: %[[GOOD:bb.[0-9]+]]{{.*}}%[[BAD:bb.[0-9]+]]
; CHECK:     EH_LABEL
; CHECK:     $w0 = COPY
; CHECK:     BL @foo, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit $w0, implicit-def $w0
; CHECK:     {{%[0-9]+}}:_(s32) = COPY $w0
; CHECK:     EH_LABEL
; CHECK:     G_BR %[[GOOD]]

; CHECK:   [[BAD]].{{[a-z]+}} (landing-pad):
; CHECK:     EH_LABEL
; CHECK:     [[PTR_RET:%[0-9]+]]:_(p0) = COPY $x0
; CHECK:     [[SEL_PTR:%[0-9]+]]:_(p0) = COPY $x1
; CHECK:     [[SEL_RET:%[0-9]+]]:_(s32) = G_PTRTOINT [[SEL_PTR]]
; CHECK:     $x0 = COPY [[PTR_RET]]
; CHECK:     $w1 = COPY [[SEL_RET]]

; CHECK:   [[GOOD]].{{[a-z]+}}:
; CHECK:     [[SEL:%[0-9]+]]:_(s32) = G_CONSTANT i32 1
; CHECK:     $w1 = COPY [[SEL]]

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

; CHECK-LABEL: name: test_invoke_indirect
; CHECK: [[CALLEE:%[0-9]+]]:gpr64(p0) = COPY $x0
; CHECK: BLR [[CALLEE]]
define void @test_invoke_indirect(void()* %callee) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  invoke void %callee() to label %continue unwind label %broken

broken:
  landingpad { i8*, i32 } catch i8* bitcast(i8** @_ZTIi to i8*)
  ret void

continue:
  ret void
}

; CHECK-LABEL: name: test_invoke_varargs

; CHECK: [[NULL:%[0-9]+]]:_(p0) = G_CONSTANT i64 0
; CHECK: [[ANSWER:%[0-9]+]]:_(s32) = G_CONSTANT i32 42
; CHECK: [[ONE:%[0-9]+]]:_(s32) = G_FCONSTANT float 1.0

; CHECK: $x0 = COPY [[NULL]]

; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFFSET:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
; CHECK: [[SLOT:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[OFFSET]](s64)
; CHECK: [[ANSWER_EXT:%[0-9]+]]:_(s64) = G_ANYEXT [[ANSWER]]
; CHECK: G_STORE [[ANSWER_EXT]](s64), [[SLOT]]

; CHECK: [[SP:%[0-9]+]]:_(p0) = COPY $sp
; CHECK: [[OFFSET:%[0-9]+]]:_(s64) = G_CONSTANT i64 8
; CHECK: [[SLOT:%[0-9]+]]:_(p0) = G_PTR_ADD [[SP]], [[OFFSET]](s64)
; CHECK: G_STORE [[ONE]](s32), [[SLOT]]

; CHECK: BL @printf
declare void @printf(i8*, ...)
define void @test_invoke_varargs() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  invoke void(i8*, ...) @printf(i8* null, i32 42, float 1.0) to label %continue unwind label %broken

broken:
  landingpad { i8*, i32 } catch i8* bitcast(i8** @_ZTIi to i8*)
  ret void

continue:
  ret void
}

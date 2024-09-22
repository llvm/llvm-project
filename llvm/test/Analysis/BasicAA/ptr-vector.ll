; RUN: opt -print-all-alias-modref-info -passes=aa-eval -disable-output < %s 2>&1 | FileCheck %s

; CHECK: MayAlias:	i8* %b, i8* %p
; CHECK: Just Ref:  Ptr: i8* %p	<->  %v1p = call <1 x ptr> @llvm.masked.load.v1p0.p0(ptr %a, i32 8, <1 x i1> %c, <1 x ptr> poison)
; CHECK: Just Ref:  Ptr: i8* %b	<->  %v1p = call <1 x ptr> @llvm.masked.load.v1p0.p0(ptr %a, i32 8, <1 x i1> %c, <1 x ptr> poison)
define void @test(ptr %a, ptr %b, <1 x i1> %c) {
  %v1p = call <1 x ptr> @llvm.masked.load.v1p0.p0(ptr %a, i32 8, <1 x i1> %c, <1 x ptr> poison)
  %p = bitcast <1 x ptr> %v1p to ptr
  load i8, ptr %p
  store i8 0, ptr %b
  ret void
}

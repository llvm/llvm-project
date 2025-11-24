; RUN: llc -march=hexagon -mcpu=hexagonv68 -mattr=+hvxv68,+hvx-length128B < %s | FileCheck %s
; REQUIRES: asserts

; Check that the test does not assert when unaligned vector store V6_vS32Ub_npred_ai is generated.
; CHECK: if (!p{{[0-3]}}) vmemu

target triple = "hexagon-unknown-unknown-elf"

define fastcc void @test(i1 %cmp.i.i) {
entry:
  %call.i.i.i172 = load ptr, ptr null, align 4
  %add.ptr = getelementptr i8, ptr %call.i.i.i172, i32 1
  store <32 x i32> zeroinitializer, ptr %add.ptr, align 128
  %add.ptr4.i4 = getelementptr i8, ptr %call.i.i.i172, i32 129
  br i1 %cmp.i.i, label %common.ret, label %if.end.i.i

common.ret:                                       ; preds = %if.end.i.i, %entry
  ret void

if.end.i.i:                                       ; preds = %entry
  store <32 x i32> zeroinitializer, ptr %add.ptr4.i4, align 1
  br label %common.ret
}

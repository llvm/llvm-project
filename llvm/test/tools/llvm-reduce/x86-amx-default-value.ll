; RUN: llvm-reduce --delta-passes=instructions-to-return --test FileCheck --test-arg %s --test-arg --input-file %s

; Check that we do not end up executing an unreachable by trying to create
; a default constant value for an x86_amx value.

; CHECK: x86_amx

target triple = "x86_64-unknown-linux-gnu"

define void @wobble() #0 {
bbl:
  %call = call x86_amx @llvm.x86.tilezero.internal(i16 0, i16 poison)
  ret void
}

define <256 x i32> @ham() #0 {
bbl:
  %call = call x86_amx @llvm.x86.tileloadd64.internal(i16 poison, i16 0, ptr null, i64 0)
  ret <256 x i32> zeroinitializer
}

define <256 x i32> @spam(i64 %arg, i1 %arg1) #0 {
bbl:
  %alloca = alloca i32, i32 4
  br i1 %arg1, label %bbl2, label %bbl3

bbl2:                                             ; preds = %bbl
  call void @baz()
  ret <256 x i32> zeroinitializer

bbl3:                                             ; preds = %bbl
  switch i64 %arg, label %bbl11 [
    i64 2, label %bbl9
    i64 1, label %bbl4
  ]

bbl4:                                             ; preds = %bbl3
  call void @llvm.lifetime.start.p0(ptr %alloca)
  call x86_amx @llvm.x86.tilezero.internal(i16 0, i16 poison)
  call x86_amx @llvm.x86.tilezero.internal(i16 0, i16 poison)
  br label %bbl5

bbl5:                                             ; preds = %bbl5, %bbl4
  %phi = phi ptr [ %alloca, %bbl4 ], [ null, %bbl5 ]
  %phi6 = phi i1 [ false, %bbl4 ], [ true, %bbl5 ]
  store volatile i32 0, ptr %phi, align 4
  br i1 %phi6, label %bbl7, label %bbl5

bbl7:                                             ; preds = %bbl5
  %call = call <256 x i32> @ham()
  %call8 = call <256 x i32> @ham()
  ret <256 x i32> zeroinitializer

bbl9:                                             ; preds = %bbl3
  call x86_amx @llvm.x86.tilezero.internal(i16 0, i16 poison)
  call x86_amx @llvm.x86.tilezero.internal(i16 0, i16 poison)
  %call10 = call x86_amx @llvm.x86.tileloadd64.internal(i16 0, i16 0, ptr null, i64 0)
  unreachable

bbl11:                                            ; preds = %bbl3
  call void @wobble()
  ret <256 x i32> zeroinitializer
}

; Function Attrs: noreturn
define void @baz() #3 {
bbl:
  call void @wobble()
  unreachable
}

attributes #0 = { "target-features"="+amx-tile" }
attributes #3 = { noreturn }

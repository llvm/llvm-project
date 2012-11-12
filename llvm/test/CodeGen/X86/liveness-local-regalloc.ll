; RUN: llc < %s -regalloc=fast -optimize-regalloc=0 -verify-machineinstrs -mtriple=x86_64-apple-darwin10
; <rdar://problem/7755473>
; PR12821

%0 = type { i32, i8*, i8*, %1*, i8*, i64, i64, i32, i32, i32, i32, [1024 x i8] }
%1 = type { i8*, i32, i32, i16, i16, %2, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %2, %3*, i32, [3 x i8], [1 x i8], %2, i32, i64 }
%2 = type { i8*, i32 }
%3 = type opaque

declare fastcc i32 @func(%0*, i32, i32) nounwind ssp

define fastcc void @func2(%0* %arg, i32 %arg1) nounwind ssp {
bb:
  br label %.exit3

.exit3:                                           ; preds = %.exit3, %bb
  switch i32 undef, label %.exit3 [
    i32 -1, label %.loopexit
    i32 37, label %bb2
  ]

bb2:                                              ; preds = %bb5, %bb3, %.exit3
  br i1 undef, label %bb3, label %bb5

bb3:                                              ; preds = %bb2
  switch i32 undef, label %infloop [
    i32 125, label %.loopexit
    i32 -1, label %bb4
    i32 37, label %bb2
  ]

bb4:                                              ; preds = %bb3
  %tmp = add nsw i32 undef, 1                     ; <i32> [#uses=1]
  br label %.loopexit

bb5:                                              ; preds = %bb2
  switch i32 undef, label %infloop1 [
    i32 -1, label %.loopexit
    i32 37, label %bb2
  ]

.loopexit:                                        ; preds = %bb5, %bb4, %bb3, %.exit3
  %.04 = phi i32 [ %tmp, %bb4 ], [ undef, %bb3 ], [ undef, %.exit3 ], [ undef, %bb5 ] ; <i32> [#uses=2]
  br i1 undef, label %bb8, label %bb6

bb6:                                              ; preds = %.loopexit
  %tmp7 = tail call fastcc i32 @func(%0* %arg, i32 %.04, i32 undef) nounwind ssp ; <i32> [#uses=0]
  ret void

bb8:                                              ; preds = %.loopexit
  %tmp9 = sext i32 %.04 to i64                    ; <i64> [#uses=1]
  %tmp10 = getelementptr inbounds %0* %arg, i64 0, i32 11, i64 %tmp9 ; <i8*> [#uses=1]
  store i8 0, i8* %tmp10, align 1
  ret void

infloop:                                          ; preds = %infloop, %bb3
  br label %infloop

infloop1:                                         ; preds = %infloop1, %bb5
  br label %infloop1
}


; RAFast would forget to add a super-register <imp-def> when rewriting:
;  %vreg10:sub_32bit<def,read-undef> = COPY %R9D<kill>
; This trips up the machine code verifier.
define void @autogen_SD24657(i8*, i32*, i64*, i32, i64, i8) {
BB:
  %A4 = alloca <16 x i16>
  %A3 = alloca double
  %A2 = alloca <2 x i8>
  %A1 = alloca i1
  %A = alloca i32
  %L = load i8* %0
  store i8 -37, i8* %0
  %E = extractelement <4 x i64> zeroinitializer, i32 2
  %Shuff = shufflevector <4 x i64> zeroinitializer, <4 x i64> zeroinitializer, <4 x i32> <i32 5, i32 7, i32 1, i32 3>
  %I = insertelement <2 x i8> <i8 -1, i8 -1>, i8 %5, i32 1
  %B = fadd float 0x45CDF5B1C0000000, 0x45CDF5B1C0000000
  %FC = uitofp i32 275048 to double
  %Sl = select i1 true, <2 x i8> %I, <2 x i8> <i8 -1, i8 -1>
  %Cmp = icmp slt i64 0, %E
  br label %CF

CF:                                               ; preds = %BB
  store i8 %5, i8* %0
  store <2 x i8> %I, <2 x i8>* %A2
  store i8 %5, i8* %0
  store i8 %5, i8* %0
  store i8 %5, i8* %0
  ret void
}

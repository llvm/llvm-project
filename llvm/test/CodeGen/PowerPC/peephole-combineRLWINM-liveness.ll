; RUN: llc -verify-machineinstrs -mtriple=powerpc-ibm-aix < %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix < %s

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <8 x i16> @llvm.ppc.altivec.vupkhsb(<16 x i8>) #0

define i32 @main(i32 %argc, ptr %argv, i1 %exitcond.not.i, <16 x i8> %0, i8 %1, i128 %2, i128 %3, i8 %4, i8 %5, i8 %6, i8 %7, i8 %8, i8 %9, i16 %10, i1 %cmp9.not.2.i, i1 %cmp9.not.3.i, i1 %cmp9.not.4.i, i1 %cmp9.not.6.i, i1 %cmp9.not.7.i, <4 x i32> %bc128.i, i32 %11, i32 %12, i1 %cmp.not.i.i, i1 %cmp4.not.i.i, i1 %or.cond.i.i, i32 %13, i32 %14, i32 %15, i1 %cmp8.not.i.i, i1 %cmp12.not.i.i, i1 %or.cond.i, i32 %16, i32 %17, i32 %e.sroa.0.0.copyload.lcssa116.i, i32 %18, i32 %add.i9.i84.i, i32 %add.i9.i84100101.i, i32 %19, i32 %add.i5.i80.i, ptr %20, i32 %21, i1 %cmp38.not.i, i32 %conv36.i, i32 %conv36.1.i, i32 %conv36.2.i, i32 %conv36.3.i, ptr %22, i32 %j.198.lcssa.i, i32 %23, i1 %cmp38.not.1.i, i32 %24, i1 %cmp38.not.3.i, i1 %exitcond.not.i48, <16 x i8> %25, <8 x i16> %26, i8 %27, i1 %cmp10.not.i, i8 %28, i8 %29, i8 %30, i8 %31, i8 %32, i8 %33, i8 %34, ptr %35, ptr %arrayidx8.lcssa.i, i1 %cmp10.not.1.i, i1 %cmp10.not.2.i, i1 %cmp10.not.3.i, i1 %cmp10.not.4.i, i1 %cmp10.not.5.i, i1 %cmp10.not.6.i, i1 %cmp10.not.7.i, <4 x i32> %bc130.i31, <8 x i16> %36, <4 x i32> %37, i1 %cmp40.not.i, ptr %38, ptr %arrayidx39.lcssa.i, i32 %conv36.2.i27) #1 {
entry:
  %call.i = load volatile i32, ptr %arrayidx39.lcssa.i, align 4
  %seed.promoted.i = load i32, ptr %38, align 4
  br label %for.body.i

for.cond.i:                                       ; preds = %lor.lhs.false5.i.i
  br i1 %exitcond.not.i, label %for.body27.i, label %for.body.i

for.body.i:                                       ; preds = %for.cond.i, %entry
  %add.i9.i9496.i = phi i32 [ %seed.promoted.i, %entry ], [ %add.i9.i.i, %for.cond.i ]
  %mul.i.i.i = mul i32 %add.i9.i9496.i, 69607
  %add.i.i.i4 = or i32 %mul.i.i.i, 54329
  store i32 %add.i.i.i4, ptr %arrayidx8.lcssa.i, align 16
  %39 = mul i32 %argc, 1556607735
  %add.i9.i.i = or i32 %39, 840608209
  %40 = load <16 x i8>, ptr %35, align 16
  %41 = tail call <8 x i16> @llvm.ppc.altivec.vupkhsb(<16 x i8> %40)
  %42 = bitcast <8 x i16> %41 to i128
  %cmp9.not.i = icmp eq i8 %1, 0
  %43 = lshr i128 %42, 96
  %44 = lshr i128 %42, 32
  %45 = trunc i128 %44 to i16
  br i1 %cmp9.not.i, label %for.inc.i, label %if.then.i

if.then.i:                                        ; preds = %for.inc.6.i, %for.inc.5.i, %for.inc.4.i, %for.inc.3.i, %for.inc.2.i, %for.inc.1.i, %for.inc.i, %for.body.i
  %j.093.lcssa.i = phi i32 [ 0, %for.body.i ], [ 1, %for.inc.i ], [ 0, %for.inc.1.i ], [ 0, %for.inc.2.i ], [ 0, %for.inc.3.i ], [ 0, %for.inc.4.i ], [ 0, %for.inc.5.i ], [ 0, %for.inc.6.i ]
  %.lcssa.i = phi i8 [ 0, %for.body.i ], [ 0, %for.inc.i ], [ %4, %for.inc.1.i ], [ %5, %for.inc.2.i ], [ %6, %for.inc.3.i ], [ %7, %for.inc.4.i ], [ %8, %for.inc.5.i ], [ %9, %for.inc.6.i ]
  %arrayidx7.lcssa.i = phi ptr [ null, %for.body.i ], [ null, %for.inc.i ], [ null, %for.inc.1.i ], [ null, %for.inc.2.i ], [ null, %for.inc.3.i ], [ null, %for.inc.4.i ], [ %argv, %for.inc.5.i ], [ null, %for.inc.6.i ]
  %conv6.i = sext i8 %.lcssa.i to i32
  %call12.i = tail call i32 (ptr, ...) null(ptr null, i32 %j.093.lcssa.i)
  %46 = load i16, ptr %arrayidx7.lcssa.i, align 2
  %conv14.i = sext i16 %46 to i32
  %call16.i = tail call i32 (ptr, ...) null(ptr null, i32 %conv14.i, i32 %conv6.i)
  unreachable

for.inc.i:                                        ; preds = %for.body.i
  %47 = trunc i128 %43 to i16
  %cmp9.not.1.i = icmp eq i16 %47, 0
  br i1 %cmp9.not.1.i, label %for.inc.1.i, label %if.then.i

for.inc.1.i:                                      ; preds = %for.inc.i
  br i1 %cmp9.not.2.i, label %for.inc.2.i, label %if.then.i

for.inc.2.i:                                      ; preds = %for.inc.1.i
  br i1 %cmp9.not.3.i, label %for.inc.3.i, label %if.then.i

for.inc.3.i:                                      ; preds = %for.inc.2.i
  br i1 %cmp9.not.4.i, label %for.inc.4.i, label %if.then.i

for.inc.4.i:                                      ; preds = %for.inc.3.i
  %cmp9.not.5.i = icmp eq i16 %45, 0
  br i1 %cmp9.not.5.i, label %for.inc.5.i, label %if.then.i

for.inc.5.i:                                      ; preds = %for.inc.4.i
  br i1 %cmp9.not.6.i, label %for.inc.6.i, label %if.then.i

for.inc.6.i:                                      ; preds = %for.inc.5.i
  br i1 %cmp9.not.7.i, label %for.inc.7.i, label %if.then.i

for.inc.7.i:                                      ; preds = %for.inc.6.i
  br i1 %or.cond.i.i, label %lor.lhs.false5.i.i, label %verify_equal.exit.i

lor.lhs.false5.i.i:                               ; preds = %for.inc.7.i
  br i1 %cmp12.not.i.i, label %for.cond.i, label %verify_equal.exit.i

verify_equal.exit.i:                              ; preds = %lor.lhs.false5.i.i, %for.inc.7.i
  %call23.i.i = tail call i32 (ptr, ...) null(ptr null, i32 %e.sroa.0.0.copyload.lcssa116.i, i32 %18, i32 0, i32 0)
  br label %for.body.i3

for.body27.i:                                     ; preds = %for.inc49.2.i, %for.cond.i
  store i32 %add.i5.i80.i, ptr %20, align 4
  br i1 %cmp38.not.i, label %for.inc49.i, label %if.then40.i

if.then40.i:                                      ; preds = %for.inc49.2.i, %for.inc49.1.i, %for.inc49.i, %for.body27.i
  %conv36.lcssa.i = phi i32 [ %conv36.i, %for.body27.i ], [ %conv36.1.i, %for.inc49.i ], [ %conv36.2.i, %for.inc49.1.i ], [ %conv36.3.i, %for.inc49.2.i ]
  %arrayidx37.lcssa.i = phi ptr [ null, %for.body27.i ], [ %22, %for.inc49.i ], [ null, %for.inc49.1.i ], [ null, %for.inc49.2.i ]
  %call42.i = tail call i32 (ptr, ...) null(ptr null, i32 %12)
  %48 = load i32, ptr %arrayidx37.lcssa.i, align 4
  %call44.i = tail call i32 (ptr, ...) null(ptr null, i32 %48, i32 %conv36.lcssa.i)
  unreachable

for.inc49.i:                                      ; preds = %for.body27.i
  br i1 %cmp38.not.1.i, label %for.inc49.1.i, label %if.then40.i

for.inc49.1.i:                                    ; preds = %for.inc49.i
  %cmp38.not.2.i = icmp eq i32 %24, %conv36.2.i27
  br i1 %cmp38.not.2.i, label %for.inc49.2.i, label %if.then40.i

for.inc49.2.i:                                    ; preds = %for.inc49.1.i
  br i1 %cmp.not.i.i, label %for.body27.i, label %if.then40.i

for.cond.i47:                                     ; preds = %lor.lhs.false5.i.i43
  br i1 %cmp8.not.i.i, label %for.body28.i, label %for.body.i3

for.body.i3:                                      ; preds = %for.cond.i47, %verify_equal.exit.i
  store i32 %add.i9.i84100101.i, ptr %argv, align 4
  store i32 %11, ptr null, align 8
  %49 = lshr i128 %3, 96
  %50 = lshr i128 %2, 32
  %51 = trunc i128 %49 to i32
  %52 = trunc i128 %50 to i32
  br i1 %cmp10.not.i, label %for.inc.i22, label %if.then.i19

if.then.i19:                                      ; preds = %for.inc.6.i29, %for.inc.5.i28, %for.inc.4.i27, %for.inc.3.i26, %for.inc.2.i25, %for.inc.1.i24, %for.inc.i22, %for.body.i3
  %.lcssa.i20 = phi i8 [ %27, %for.body.i3 ], [ %28, %for.inc.i22 ], [ %29, %for.inc.1.i24 ], [ %30, %for.inc.2.i25 ], [ %31, %for.inc.3.i26 ], [ %32, %for.inc.4.i27 ], [ %33, %for.inc.5.i28 ], [ %34, %for.inc.6.i29 ]
  %conv7.i = sext i8 %.lcssa.i20 to i32
  %call13.i = tail call i32 (ptr, ...) null(ptr null, i32 %13)
  %call17.i = tail call i32 (ptr, ...) null(ptr null, i32 %15, i32 %conv7.i)
  unreachable

for.inc.i22:                                      ; preds = %for.body.i3
  br i1 %cmp10.not.1.i, label %for.inc.1.i24, label %if.then.i19

for.inc.1.i24:                                    ; preds = %for.inc.i22
  br i1 %cmp10.not.2.i, label %for.inc.2.i25, label %if.then.i19

for.inc.2.i25:                                    ; preds = %for.inc.1.i24
  br i1 %cmp10.not.3.i, label %for.inc.3.i26, label %if.then.i19

for.inc.3.i26:                                    ; preds = %for.inc.2.i25
  br i1 %cmp10.not.4.i, label %for.inc.4.i27, label %if.then.i19

for.inc.4.i27:                                    ; preds = %for.inc.3.i26
  br i1 %cmp10.not.5.i, label %for.inc.5.i28, label %if.then.i19

for.inc.5.i28:                                    ; preds = %for.inc.4.i27
  br i1 %cmp38.not.3.i, label %for.inc.6.i29, label %if.then.i19

for.inc.6.i29:                                    ; preds = %for.inc.5.i28
  br i1 %cmp4.not.i.i, label %for.inc.7.i30, label %if.then.i19

for.inc.7.i30:                                    ; preds = %for.inc.6.i29
  br i1 %cmp40.not.i, label %lor.lhs.false5.i.i43, label %verify_equal.exit.i38

lor.lhs.false5.i.i43:                             ; preds = %for.inc.7.i30
  %cmp8.not.i.i44 = icmp eq i32 0, %52
  %or.cond.i46 = select i1 %cmp8.not.i.i44, i1 %cmp10.not.7.i, i1 false
  br i1 %or.cond.i46, label %for.cond.i47, label %verify_equal.exit.i38

verify_equal.exit.i38:                            ; preds = %lor.lhs.false5.i.i43, %for.inc.7.i30
  %e.sroa.0.0.copyload.lcssa118.i = phi i32 [ %23, %for.inc.7.i30 ], [ %51, %lor.lhs.false5.i.i43 ]
  %call23.i.i42 = tail call i32 (ptr, ...) null(ptr null, i32 %e.sroa.0.0.copyload.lcssa118.i, i32 %19, i32 0, i32 0)
  ret i32 0

for.body28.i:                                     ; preds = %for.inc51.2.i, %for.cond.i47
  %add.i9.i86102103.i = phi i32 [ 0, %for.cond.i47 ], [ 1, %for.inc51.2.i ]
  %53 = mul i32 %add.i9.i86102103.i, 1654633953
  store i32 %53, ptr null, align 4
  br i1 %exitcond.not.i48, label %for.inc51.i, label %if.then42.i

if.then42.i:                                      ; preds = %for.inc51.2.i, %for.inc51.1.i, %for.inc51.i, %for.body28.i
  %conv38.lcssa.i = phi i32 [ %14, %for.body28.i ], [ %16, %for.inc51.i ], [ %j.198.lcssa.i, %for.inc51.1.i ], [ %add.i9.i84.i, %for.inc51.2.i ]
  %call44.i51 = tail call i32 (ptr, ...) null(ptr null, i32 %21)
  %call46.i = tail call i32 (ptr, ...) null(ptr null, i32 %17, i32 %conv38.lcssa.i)
  unreachable

for.inc51.i:                                      ; preds = %for.body28.i
  br i1 %cmp10.not.6.i, label %for.inc51.1.i, label %if.then42.i

for.inc51.1.i:                                    ; preds = %for.inc51.i
  %conv38.2.i58 = sext i16 %10 to i32
  %cmp40.not.2.i = icmp eq i32 0, %conv38.2.i58
  br i1 %cmp40.not.2.i, label %for.inc51.2.i, label %if.then42.i

for.inc51.2.i:                                    ; preds = %for.inc51.1.i
  br i1 %or.cond.i, label %for.body28.i, label %if.then42.i
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #1 = { "target-features"="+altivec,+bpermd,+extdiv,+isa-v206-instructions,+vsx,-aix-shared-lib-tls-model-opt,-aix-small-local-dynamic-tls,-aix-small-local-exec-tls,-crbits,-crypto,-direct-move,-htm,-isa-v207-instructions,-isa-v30-instructions,-power8-vector,-power9-vector,-privileged,-quadword-atomics,-rop-protect,-spe" }

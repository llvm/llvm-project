; RUN: llc -verify-machineinstrs -mtriple=powerpc-ibm-aix < %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix < %s

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <8 x i16> @llvm.ppc.altivec.vupkhsb(<16 x i8>) #0

define i32 @call_tester(i32 %argc, ptr %argv, i1 %exitcond.not.i, <16 x i8> %0, i8 %1, i128 %2, i128 %3, i16 %4, i1 %cmp9.not.6.i) #1 {
entry:
  %seed.promoted.i = load i32, ptr null, align 4
  br label %for.body.i

for.cond.i:                                       ; preds = %for.inc.7.i
  br i1 %exitcond.not.i, label %for.body27.i, label %for.body.i

for.body.i:                                       ; preds = %for.cond.i, %entry
  %add.i9.i9496.i = phi i32 [ %seed.promoted.i, %entry ], [ %5, %for.cond.i ]
  %mul.i.i.i = mul i32 %add.i9.i9496.i, 69607
  %add.i.i.i4 = or i32 %mul.i.i.i, 54329
  store i32 %add.i.i.i4, ptr %argv, align 16
  %5 = mul i32 %argc, 1556607735
  %6 = load <16 x i8>, ptr %argv, align 16
  %7 = tail call <8 x i16> @llvm.ppc.altivec.vupkhsb(<16 x i8> %6)
  %8 = bitcast <8 x i16> %7 to i128
  %9 = lshr i128 %8, 96
  %10 = lshr i128 %3, 32
  %11 = trunc i128 %10 to i16
  br i1 %exitcond.not.i, label %for.inc.i, label %if.then.i

if.then.i:                                        ; preds = %for.inc.5.i, %for.inc.4.i, %for.inc.i, %for.body.i
  %j.093.lcssa.i = phi i32 [ 0, %for.body.i ], [ 1, %for.inc.i ], [ 0, %for.inc.4.i ], [ 0, %for.inc.5.i ]
  %.lcssa.i = phi i8 [ 0, %for.body.i ], [ 0, %for.inc.i ], [ 1, %for.inc.4.i ], [ 0, %for.inc.5.i ]
  %arrayidx7.lcssa.i = phi ptr [ null, %for.body.i ], [ null, %for.inc.i ], [ null, %for.inc.4.i ], [ %argv, %for.inc.5.i ]
  %conv6.i = sext i8 %.lcssa.i to i32
  %call12.i = tail call i32 (ptr, ...) null(ptr null, i32 %j.093.lcssa.i)
  %12 = load i16, ptr %arrayidx7.lcssa.i, align 2
  %conv14.i = sext i16 %12 to i32
  %call16.i = tail call i32 (ptr, ...) null(ptr null, i32 %conv14.i, i32 %conv6.i)
  unreachable

for.inc.i:                                        ; preds = %for.body.i
  %13 = trunc i128 %9 to i16
  %cmp9.not.1.i = icmp eq i16 %13, 0
  br i1 %cmp9.not.1.i, label %for.inc.4.i, label %if.then.i

for.inc.4.i:                                      ; preds = %for.inc.i
  %cmp9.not.5.i = icmp eq i16 %11, 0
  br i1 %cmp9.not.5.i, label %for.inc.5.i, label %if.then.i

for.inc.5.i:                                      ; preds = %for.inc.4.i
  br i1 %exitcond.not.i, label %for.inc.7.i, label %if.then.i

for.inc.7.i:                                      ; preds = %for.inc.5.i
  br i1 %cmp9.not.6.i, label %for.cond.i, label %verify_equal.exit.i

verify_equal.exit.i:                              ; preds = %for.inc.7.i
  %14 = lshr i128 %2, 96
  %15 = lshr i128 %2, 32
  %16 = trunc i128 %14 to i32
  %17 = trunc i128 %15 to i32
  %.mux = select i1 %exitcond.not.i, i8 0, i8 1
  br i1 %exitcond.not.i, label %if.then.i19, label %for.inc.7.i30

for.body27.i:                                     ; preds = %for.cond.i
  br i1 %exitcond.not.i, label %for.inc49.1.i, label %if.then40.i

if.then40.i:                                      ; preds = %for.inc49.1.i, %for.body27.i
  %conv36.lcssa.i = phi i32 [ 1, %for.body27.i ], [ 0, %for.inc49.1.i ]
  %arrayidx37.lcssa.i = phi ptr [ %argv, %for.body27.i ], [ null, %for.inc49.1.i ]
  %18 = load i32, ptr %arrayidx37.lcssa.i, align 4
  %call44.i = tail call i32 (ptr, ...) null(ptr null, i32 %18, i32 %conv36.lcssa.i)
  unreachable

for.inc49.1.i:                                    ; preds = %for.body27.i
  br label %if.then40.i

if.then.i19:                                      ; preds = %verify_equal.exit.i
  %conv7.i = sext i8 %.mux to i32
  %call17.i = tail call i32 (ptr, ...) null(ptr null, i32 0, i32 %conv7.i)
  unreachable

for.inc.7.i30:                                    ; preds = %verify_equal.exit.i
  br i1 %exitcond.not.i, label %lor.lhs.false5.i.i43, label %verify_equal.exit.i38

lor.lhs.false5.i.i43:                             ; preds = %for.inc.7.i30
  %cmp8.not.i.i44 = icmp eq i32 0, %17
  br i1 %cmp8.not.i.i44, label %for.body28.i, label %verify_equal.exit.i38

verify_equal.exit.i38:                            ; preds = %lor.lhs.false5.i.i43, %for.inc.7.i30
  %e.sroa.0.0.copyload.lcssa118.i = phi i32 [ 0, %for.inc.7.i30 ], [ %16, %lor.lhs.false5.i.i43 ]
  %call23.i.i42 = tail call i32 (ptr, ...) null(ptr null, i32 %e.sroa.0.0.copyload.lcssa118.i, i32 0, i32 0, i32 0)
  ret i32 0

for.body28.i:                                     ; preds = %for.inc51.1.i, %lor.lhs.false5.i.i43
  %add.i9.i86102103.i = phi i32 [ 0, %lor.lhs.false5.i.i43 ], [ 1, %for.inc51.1.i ]
  %19 = mul i32 %add.i9.i86102103.i, 1654633953
  store i32 %19, ptr null, align 4
  br i1 %exitcond.not.i, label %for.inc51.1.i, label %if.then42.i

if.then42.i:                                      ; preds = %for.inc51.1.i, %for.body28.i
  %conv38.lcssa.i = phi i32 [ 1, %for.body28.i ], [ 0, %for.inc51.1.i ]
  %call46.i = tail call i32 (ptr, ...) null(ptr null, i32 0, i32 %conv38.lcssa.i)
  unreachable

for.inc51.1.i:                                    ; preds = %for.body28.i
  %conv38.2.i58 = sext i16 %4 to i32
  %cmp40.not.2.i = icmp eq i32 0, %conv38.2.i58
  br i1 %cmp40.not.2.i, label %for.body28.i, label %if.then42.i
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #1 = { "target-features"="+altivec,+bpermd,+extdiv,+isa-v206-instructions,+vsx,-aix-shared-lib-tls-model-opt,-aix-small-local-dynamic-tls,-aix-small-local-exec-tls,-crbits,-crypto,-direct-move,-htm,-isa-v207-instructions,-isa-v30-instructions,-power8-vector,-power9-vector,-privileged,-quadword-atomics,-rop-protect,-spe" }

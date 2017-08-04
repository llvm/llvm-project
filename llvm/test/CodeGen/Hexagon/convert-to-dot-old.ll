; RUN: llc -march=hexagon -mcpu=hexagonv55 -filetype=obj -o /dev/null
; REQUIRES: asserts
; There should be no output (nothing on stderr).

; Due to a bug in converting a dot-new branch into a dot-old one, opcodes
; with branch prediction bits were selected even if the architecture did
; not support them. On V55-, the dot-old branch opcodes are J2_jumpt and
; J2_jumpf (and a pair of J2_jumpr*), whereas J2_jumptpt could have been
; a result of the conversion to dot-old. This would fail a verification
; check in the MC code emitter, so make sure it does not happen.

target triple = "hexagon"

define void @fred(i16* nocapture %a0, i16* nocapture %a1, i16* nocapture %a2, i16 signext %a3, i16* %a4, i16 signext %a5, i16 signext %a6, i16 signext %a7, i32 %a8, i16 signext %a9, i16 signext %a10) local_unnamed_addr #0 {
b11:
  %v12 = sext i16 %a5 to i32
  %v13 = tail call i32 @llvm.hexagon.A2.sxth(i32 %v12)
  %v14 = tail call i32 @llvm.hexagon.A2.sxth(i32 2)
  %v15 = tail call i32 @llvm.hexagon.A2.sxth(i32 undef)
  %v16 = tail call i32 @llvm.hexagon.A2.sath(i32 undef)
  %v17 = tail call i32 @llvm.hexagon.A2.sxth(i32 %v16)
  %v18 = tail call i32 @llvm.hexagon.A2.aslh(i32 undef)
  %v19 = tail call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %v18, i32 %v14)
  %v20 = tail call i32 @llvm.hexagon.A2.asrh(i32 %v19)
  %v21 = tail call i32 @llvm.hexagon.A2.sxth(i32 %v20)
  %v22 = tail call i32 @llvm.hexagon.A2.sub(i32 %v17, i32 %v21)
  %v23 = tail call i32 @llvm.hexagon.A2.sath(i32 %v22)
  %v24 = select i1 undef, i32 undef, i32 %v23
  %v25 = tail call i32 @llvm.hexagon.A2.sxth(i32 %v24)
  %v26 = tail call i32 @llvm.hexagon.A2.sub(i32 %v13, i32 %v25)
  %v27 = tail call i32 @llvm.hexagon.A2.sath(i32 %v26)
  %v28 = tail call i32 @llvm.hexagon.A2.sxth(i32 %v27)
  %v29 = tail call i32 @llvm.hexagon.A2.sub(i32 %v28, i32 %v14)
  %v30 = tail call i32 @llvm.hexagon.A2.sath(i32 %v29)
  %v31 = shl i32 %v30, 16
  %v32 = icmp sgt i32 undef, %v31
  %v33 = select i1 %v32, i32 %v30, i32 undef
  %v34 = trunc i32 %v33 to i16
  %v35 = trunc i32 %v24 to i16
  call void @foo(i16* nonnull undef, i32* nonnull undef, i16* %a4, i16 signext %v35, i16 signext %v34, i16 signext 2) #4
  %v36 = call i32 @llvm.hexagon.S2.asr.r.r.sat(i32 %v18, i32 undef)
  %v37 = call i32 @llvm.hexagon.A2.asrh(i32 %v36)
  %v38 = call i32 @llvm.hexagon.A2.sub(i32 %v13, i32 undef)
  %v39 = call i32 @llvm.hexagon.A2.sath(i32 %v38)
  %v40 = call i32 @llvm.hexagon.A2.sxth(i32 %v39)
  %v41 = call i32 @llvm.hexagon.A2.sub(i32 %v40, i32 %v14)
  %v42 = call i32 @llvm.hexagon.A2.sath(i32 %v41)
  %v43 = select i1 undef, i32 %v42, i32 %v37
  %v44 = trunc i32 %v43 to i16
  call void @foo(i16* nonnull undef, i32* nonnull undef, i16* %a4, i16 signext undef, i16 signext %v44, i16 signext 2) #4
  %v45 = call i32 @llvm.hexagon.A2.sath(i32 undef)
  %v46 = select i1 undef, i32 undef, i32 %v45
  %v47 = trunc i32 %v46 to i16
  call void @foo(i16* nonnull undef, i32* nonnull undef, i16* %a4, i16 signext %v47, i16 signext undef, i16 signext 2) #4
  %v48 = call i32 @llvm.hexagon.A2.sub(i32 undef, i32 %v15)
  %v49 = call i32 @llvm.hexagon.A2.sath(i32 %v48)
  %v50 = trunc i32 %v49 to i16
  store i16 %v50, i16* undef, align 2
  store i16 %a3, i16* %a0, align 2
  %v51 = sext i16 %a10 to i32
  %v52 = call i32 @llvm.hexagon.A2.sxth(i32 %v51)
  %v53 = call i32 @llvm.hexagon.A2.add(i32 undef, i32 %v52)
  %v54 = call i32 @llvm.hexagon.A2.sath(i32 %v53)
  %v55 = trunc i32 %v54 to i16
  store i16 %v55, i16* %a1, align 2
  store i16 %a7, i16* %a2, align 2
  %v56 = sext i16 %a9 to i32
  %v57 = call i32 @llvm.hexagon.A2.sxth(i32 %v56)
  br i1 undef, label %b58, label %b62

b58:                                              ; preds = %b11
  %v59 = call i32 @llvm.hexagon.A2.add(i32 %v57, i32 %v52)
  %v60 = call i32 @llvm.hexagon.A2.sath(i32 %v59)
  %v61 = trunc i32 %v60 to i16
  store i16 %v61, i16* %a1, align 2
  br label %b63

b62:                                              ; preds = %b11
  br label %b63

b63:                                              ; preds = %b62, %b58
  %v64 = phi i16 [ undef, %b58 ], [ %a9, %b62 ]
  %v65 = icmp slt i16 undef, %v64
  br i1 %v65, label %b66, label %b67

b66:                                              ; preds = %b63
  br i1 undef, label %b67, label %b68

b67:                                              ; preds = %b66, %b63
  store i16 0, i16* %a2, align 2
  br label %b68

b68:                                              ; preds = %b67, %b66
  ret void
}

declare i32 @llvm.hexagon.A2.sath(i32) #2
declare i32 @llvm.hexagon.A2.add(i32, i32) #2
declare i32 @llvm.hexagon.A2.sxth(i32) #2
declare i32 @llvm.hexagon.A2.sub(i32, i32) #2
declare i32 @llvm.hexagon.A2.asrh(i32) #2
declare i32 @llvm.hexagon.S2.asr.r.r.sat(i32, i32) #2
declare i32 @llvm.hexagon.A2.aslh(i32) #2
declare void @foo(i16*, i32*, i16*, i16 signext, i16 signext, i16 signext) local_unnamed_addr #3

attributes #0 = { nounwind optsize "target-cpu"="hexagonv55" "target-features"="-hvx,-hvx-double,-long-calls" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { optsize "target-cpu"="hexagonv55" "target-features"="-hvx,-hvx-double,-long-calls" }
attributes #4 = { nounwind optsize }

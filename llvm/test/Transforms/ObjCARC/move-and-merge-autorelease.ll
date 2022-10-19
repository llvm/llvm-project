; RUN: opt -S -passes=objc-arc,objc-arc-contract < %s | FileCheck %s

; The optimizer should be able to move the autorelease past two phi nodes
; and fold it with the release in bb65.

; CHECK: bb65:
; CHECK: call i8* @llvm.objc.retainAutorelease
; CHECK: br label %bb76

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.0.0"

%0 = type opaque
%1 = type opaque
%2 = type opaque
%3 = type opaque
%4 = type opaque
%5 = type opaque

@"\01L_OBJC_SELECTOR_REFERENCES_11" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_421455" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_598" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_620" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_622" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_624" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_626" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"

declare i8* @objc_msgSend(i8*, i8*, ...)

declare i8* @llvm.objc.retain(i8*)

declare void @llvm.objc.release(i8*)

declare i8* @llvm.objc.autorelease(i8*)

define hidden %0* @foo(%1* %arg, %3* %arg3) {
bb:
  %tmp16 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_620", align 8
  %tmp17 = bitcast %3* %arg3 to i8*
  %tmp18 = call %4* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %4* (i8*, i8*)*)(i8* %tmp17, i8* %tmp16)
  %tmp19 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_622", align 8
  %tmp20 = bitcast %4* %tmp18 to i8*
  %tmp21 = call %5* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %5* (i8*, i8*)*)(i8* %tmp20, i8* %tmp19)
  %tmp22 = bitcast %5* %tmp21 to i8*
  %tmp23 = call i8* @llvm.objc.retain(i8* %tmp22) nounwind
  %tmp24 = bitcast i8* %tmp23 to %5*
  %tmp26 = icmp eq i8* %tmp23, null
  br i1 %tmp26, label %bb81, label %bb27

bb27:                                             ; preds = %bb
  %tmp29 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_11", align 8
  %tmp30 = bitcast %1* %arg to i8*
  %tmp31 = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* %tmp30, i8* %tmp29)
  %tmp34 = call i8* @llvm.objc.retain(i8* %tmp31) nounwind
  %tmp37 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_421455", align 8
  %tmp39 = call %0* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %0* (i8*, i8*)*)(i8* %tmp34, i8* %tmp37)
  %tmp40 = bitcast %0* %tmp39 to i8*
  %tmp41 = call i8* @llvm.objc.retain(i8* %tmp40) nounwind
  %tmp42 = bitcast i8* %tmp41 to %0*
  %tmp44 = icmp eq i8* %tmp41, null
  br i1 %tmp44, label %bb45, label %bb55

bb45:                                             ; preds = %bb27
  %tmp47 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_624", align 8
  %tmp49 = call %0* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %0* (i8*, i8*)*)(i8* %tmp34, i8* %tmp47)
  %tmp51 = bitcast %0* %tmp49 to i8*
  %tmp52 = call i8* @llvm.objc.retain(i8* %tmp51) nounwind
  call void @llvm.objc.release(i8* %tmp41) nounwind
  br label %bb55

bb55:                                             ; preds = %bb27, %bb45
  %tmp13.0 = phi %0* [ %tmp42, %bb27 ], [ %tmp49, %bb45 ]
  %tmp57 = icmp eq %0* %tmp13.0, null
  br i1 %tmp57, label %bb76, label %bb58

bb58:                                             ; preds = %bb55
  %tmp60 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_598", align 8
  %tmp61 = bitcast %0* %tmp13.0 to i8*
  %tmp62 = call signext i8 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8 (i8*, i8*)*)(i8* %tmp61, i8* %tmp60)
  %tmp64 = icmp eq i8 %tmp62, 0
  br i1 %tmp64, label %bb76, label %bb65

bb65:                                             ; preds = %bb58
  %tmp68 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_626", align 8
  %tmp69 = bitcast %0* %tmp13.0 to i8*
  %tmp70 = call %0* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %0* (i8*, i8*, %5*)*)(i8* %tmp69, i8* %tmp68, %5* %tmp24)
  %tmp72 = bitcast %0* %tmp70 to i8*
  %tmp73 = call i8* @llvm.objc.retain(i8* %tmp72) nounwind
  br label %bb76

bb76:                                             ; preds = %bb58, %bb55, %bb65
  %tmp10.0 = phi %0* [ %tmp70, %bb65 ], [ null, %bb58 ], [ null, %bb55 ]
  %tmp78 = bitcast %0* %tmp13.0 to i8*
  call void @llvm.objc.release(i8* %tmp78) nounwind
  call void @llvm.objc.release(i8* %tmp34) nounwind
  br label %bb81

bb81:                                             ; preds = %bb, %bb76
  %tmp10.1 = phi %0* [ %tmp10.0, %bb76 ], [ null, %bb ]
  %tmp83 = bitcast %0* %tmp10.1 to i8*
  %tmp84 = call i8* @llvm.objc.retain(i8* %tmp83) nounwind
  call void @llvm.objc.release(i8* %tmp23) nounwind
  %tmp87 = call i8* @llvm.objc.autorelease(i8* %tmp84) nounwind
  %tmp88 = bitcast i8* %tmp87 to %0*
  %tmp92 = bitcast %0* %tmp10.1 to i8*
  call void @llvm.objc.release(i8* %tmp92) nounwind
  ret %0* %tmp88
}

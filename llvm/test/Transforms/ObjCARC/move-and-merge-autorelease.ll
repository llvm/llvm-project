; RUN: opt -S -passes=objc-arc,objc-arc-contract < %s | FileCheck %s

; The optimizer should be able to move the autorelease past two phi nodes
; and fold it with the release in bb65.

; CHECK: bb65:
; CHECK: call ptr @llvm.objc.retainAutorelease
; CHECK: br label %bb76

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.0.0"

%0 = type opaque
%1 = type opaque
%2 = type opaque
%3 = type opaque
%4 = type opaque
%5 = type opaque

@"\01L_OBJC_SELECTOR_REFERENCES_11" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_421455" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_598" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_620" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_622" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_624" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_626" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"

declare ptr @objc_msgSend(ptr, ptr, ...)

declare ptr @llvm.objc.retain(ptr)

declare void @llvm.objc.release(ptr)

declare ptr @llvm.objc.autorelease(ptr)

define hidden ptr @foo(ptr %arg, ptr %arg3) {
bb:
  %tmp16 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_620", align 8
  %tmp18 = call ptr @objc_msgSend(ptr %arg3, ptr %tmp16)
  %tmp19 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_622", align 8
  %tmp21 = call ptr @objc_msgSend(ptr %tmp18, ptr %tmp19)
  %tmp23 = call ptr @llvm.objc.retain(ptr %tmp21) nounwind
  %tmp26 = icmp eq ptr %tmp23, null
  br i1 %tmp26, label %bb81, label %bb27

bb27:                                             ; preds = %bb
  %tmp29 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_11", align 8
  %tmp31 = call ptr @objc_msgSend(ptr %arg, ptr %tmp29)
  %tmp34 = call ptr @llvm.objc.retain(ptr %tmp31) nounwind
  %tmp37 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_421455", align 8
  %tmp39 = call ptr @objc_msgSend(ptr %tmp34, ptr %tmp37)
  %tmp41 = call ptr @llvm.objc.retain(ptr %tmp39) nounwind
  %tmp44 = icmp eq ptr %tmp41, null
  br i1 %tmp44, label %bb45, label %bb55

bb45:                                             ; preds = %bb27
  %tmp47 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_624", align 8
  %tmp49 = call ptr @objc_msgSend(ptr %tmp34, ptr %tmp47)
  %tmp52 = call ptr @llvm.objc.retain(ptr %tmp49) nounwind
  call void @llvm.objc.release(ptr %tmp41) nounwind
  br label %bb55

bb55:                                             ; preds = %bb27, %bb45
  %tmp13.0 = phi ptr [ %tmp41, %bb27 ], [ %tmp49, %bb45 ]
  %tmp57 = icmp eq ptr %tmp13.0, null
  br i1 %tmp57, label %bb76, label %bb58

bb58:                                             ; preds = %bb55
  %tmp60 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_598", align 8
  %tmp62 = call signext i8 @objc_msgSend(ptr %tmp13.0, ptr %tmp60)
  %tmp64 = icmp eq i8 %tmp62, 0
  br i1 %tmp64, label %bb76, label %bb65

bb65:                                             ; preds = %bb58
  %tmp68 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_626", align 8
  %tmp70 = call ptr @objc_msgSend(ptr %tmp13.0, ptr %tmp68, ptr %tmp23)
  %tmp73 = call ptr @llvm.objc.retain(ptr %tmp70) nounwind
  br label %bb76

bb76:                                             ; preds = %bb58, %bb55, %bb65
  %tmp10.0 = phi ptr [ %tmp70, %bb65 ], [ null, %bb58 ], [ null, %bb55 ]
  call void @llvm.objc.release(ptr %tmp13.0) nounwind
  call void @llvm.objc.release(ptr %tmp34) nounwind
  br label %bb81

bb81:                                             ; preds = %bb, %bb76
  %tmp10.1 = phi ptr [ %tmp10.0, %bb76 ], [ null, %bb ]
  %tmp84 = call ptr @llvm.objc.retain(ptr %tmp10.1) nounwind
  call void @llvm.objc.release(ptr %tmp23) nounwind
  %tmp87 = call ptr @llvm.objc.autorelease(ptr %tmp84) nounwind
  call void @llvm.objc.release(ptr %tmp10.1) nounwind
  ret ptr %tmp87
}

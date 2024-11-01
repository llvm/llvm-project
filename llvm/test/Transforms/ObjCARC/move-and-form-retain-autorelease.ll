; RUN: opt -S -passes=objc-arc-contract < %s | FileCheck %s

; The optimizer should be able to move the autorelease past a control triangle
; and various scary looking things and fold it into an objc_retainAutorelease.

; CHECK: bb57:
; CHECK: tail call ptr @llvm.objc.retainAutorelease(ptr %tmp71x) [[NUW:#[0-9]+]]
; CHECK: bb99:

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.0.0"

%0 = type { ptr, ptr }
%1 = type { ptr, ptr }
%2 = type { ptr, ptr, ptr, ptr, ptr }
%3 = type opaque
%4 = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%5 = type { i32, i32, [0 x %6] }
%6 = type { ptr, ptr, ptr }
%7 = type { i64, [0 x ptr] }
%8 = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32 }
%9 = type { i32, i32, [0 x %1] }
%10 = type { i32, i32, [0 x %11] }
%11 = type { ptr, ptr, ptr, i32, i32 }
%12 = type { ptr, i32, ptr, i64 }
%13 = type opaque
%14 = type opaque
%15 = type opaque
%16 = type opaque
%17 = type opaque
%18 = type opaque
%19 = type opaque
%20 = type opaque
%21 = type opaque
%22 = type opaque
%23 = type opaque
%24 = type opaque
%25 = type opaque

@"\01l_objc_msgSend_fixup_alloc" = external hidden global %0, section "__DATA, __objc_msgrefs, coalesced", align 16
@"\01L_OBJC_SELECTOR_REFERENCES_8" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_3725" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_CLASSLIST_REFERENCES_$_40" = external hidden global ptr, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_SELECTOR_REFERENCES_4227" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_4631" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_CLASSLIST_REFERENCES_$_70" = external hidden global ptr, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_SELECTOR_REFERENCES_148" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_159" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_188" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_328" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01l_objc_msgSend_fixup_objectAtIndex_" = external hidden global %0, section "__DATA, __objc_msgrefs, coalesced", align 16
@_unnamed_cfstring_386 = external hidden constant %12, section "__DATA,__cfstring"
@"\01l_objc_msgSend_fixup_count" = external hidden global %0, section "__DATA, __objc_msgrefs, coalesced", align 16
@"\01L_OBJC_SELECTOR_REFERENCES_389" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_391" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_393" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@NSPrintHeaderAndFooter = external constant ptr
@"\01L_OBJC_SELECTOR_REFERENCES_395" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_CLASSLIST_REFERENCES_$_396" = external hidden global ptr, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_SELECTOR_REFERENCES_398" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_400" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_402" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_404" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_406" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_408" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_CLASSLIST_REFERENCES_$_409" = external hidden global ptr, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_SELECTOR_REFERENCES_411" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_413" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_415" = external hidden global ptr, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"

declare ptr @objc_msgSend(ptr, ptr, ...)

declare ptr @llvm.objc.retain(ptr)

declare void @llvm.objc.release(ptr)

declare ptr @llvm.objc.autorelease(ptr)

declare ptr @llvm.objc.explicit_autorelease(ptr)

define hidden ptr @foo(ptr %arg, ptr %arg2) {
bb:
  %tmp = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_3725", align 8
  %tmp5 = tail call ptr @objc_msgSend(ptr %arg, ptr %tmp)
  %tmp7 = tail call ptr @llvm.objc.retain(ptr %tmp5) nounwind
  %tmp8 = load ptr, ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_40", align 8
  %tmp9 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_4227", align 8
  %tmp11 = tail call ptr @objc_msgSend(ptr %tmp8, ptr %tmp9)
  %tmp12 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_4631", align 8
  %tmp14 = tail call signext i8 @objc_msgSend(ptr %tmp11, ptr %tmp12, ptr @_unnamed_cfstring_386)
  %tmp16 = load ptr, ptr @"\01l_objc_msgSend_fixup_count", align 16
  %tmp18 = tail call i64 %tmp16(ptr %arg2, ptr @"\01l_objc_msgSend_fixup_count")
  %tmp19 = icmp eq i64 %tmp18, 0
  br i1 %tmp19, label %bb22, label %bb20

bb20:                                             ; preds = %bb
  %tmp21 = icmp eq i8 %tmp14, 0
  br label %bb25

bb22:                                             ; preds = %bb
  %tmp24 = icmp eq i8 %tmp14, 0
  br i1 %tmp24, label %bb46, label %bb25

bb25:                                             ; preds = %bb22, %bb20
  %tmp26 = phi i1 [ %tmp21, %bb20 ], [ false, %bb22 ]
  %tmp27 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_188", align 8
  %tmp28 = tail call ptr @objc_msgSend(ptr %tmp7, ptr %tmp27)
  %tmp29 = tail call ptr @llvm.objc.explicit_autorelease(ptr %tmp28) nounwind
  tail call void @llvm.objc.release(ptr %tmp7) nounwind
  %tmp31 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_389", align 8
  %tmp32 = tail call ptr @objc_msgSend(ptr %tmp29, ptr %tmp31)
  %tmp33 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_391", align 8
  tail call void @objc_msgSend(ptr %tmp32, ptr %tmp33, ptr %arg2)
  br i1 %tmp26, label %bb46, label %bb35

bb35:                                             ; preds = %bb25
  %tmp36 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_389", align 8
  %tmp37 = tail call ptr @objc_msgSend(ptr %tmp29, ptr %tmp36)
  %tmp38 = load ptr, ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_70", align 8
  %tmp39 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_393", align 8
  %tmp41 = tail call ptr @objc_msgSend(ptr %tmp38, ptr %tmp39, i8 signext 1)
  %tmp43 = load ptr, ptr @NSPrintHeaderAndFooter, align 8
  %tmp44 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_159", align 8
  tail call void @objc_msgSend(ptr %tmp37, ptr %tmp44, ptr %tmp41, ptr %tmp43)
  br label %bb46

bb46:                                             ; preds = %bb35, %bb25, %bb22
  %tmp47 = phi ptr [ %tmp29, %bb35 ], [ %tmp29, %bb25 ], [ %tmp7, %bb22 ]
  %tmp48 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_328", align 8
  %tmp49 = tail call ptr @objc_msgSend(ptr %arg, ptr %tmp48)
  %tmp51 = load ptr, ptr @"\01l_objc_msgSend_fixup_count", align 16
  %tmp53 = tail call i64 %tmp51(ptr %tmp49, ptr @"\01l_objc_msgSend_fixup_count")
  %tmp54 = icmp eq i64 %tmp53, 0
  br i1 %tmp54, label %bb55, label %bb57

bb55:                                             ; preds = %bb46
  %tmp56 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_395", align 8
  tail call void @objc_msgSend(ptr %arg, ptr %tmp56)
  br label %bb57

bb57:                                             ; preds = %bb55, %bb46
  %tmp58 = load ptr, ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_396", align 8
  %tmp59 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_328", align 8
  %tmp60 = tail call ptr @objc_msgSend(ptr %arg, ptr %tmp59)
  %tmp62 = load ptr, ptr @"\01l_objc_msgSend_fixup_objectAtIndex_", align 16
  %tmp64 = tail call ptr %tmp62(ptr %tmp60, ptr @"\01l_objc_msgSend_fixup_objectAtIndex_", i64 0)
  %tmp65 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_398", align 8
  %tmp66 = tail call ptr @objc_msgSend(ptr %tmp64, ptr %tmp65)
  %tmp68 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_400", align 8
  %tmp70 = tail call ptr @objc_msgSend(ptr %tmp58, ptr %tmp68, ptr %tmp66, ptr %tmp47)
  ; hack to prevent the optimize from using objc_retainAutoreleasedReturnValue.
  %tmp71x = getelementptr i8, ptr %tmp70, i64 1
  %tmp72 = tail call ptr @llvm.objc.retain(ptr %tmp71x) nounwind
  %tmp73 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_402", align 8
  tail call void @objc_msgSend(ptr %tmp72, ptr %tmp73, i8 signext 1)
  %tmp74 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_404", align 8
  tail call void @objc_msgSend(ptr %tmp72, ptr %tmp74, i8 signext 1)
  %tmp75 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_328", align 8
  %tmp76 = tail call ptr @objc_msgSend(ptr %arg, ptr %tmp75)
  %tmp78 = load ptr, ptr @"\01l_objc_msgSend_fixup_objectAtIndex_", align 16
  %tmp80 = tail call ptr %tmp78(ptr %tmp76, ptr @"\01l_objc_msgSend_fixup_objectAtIndex_", i64 0)
  %tmp81 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_406", align 8
  tail call void @objc_msgSend(ptr %tmp80, ptr %tmp81, i64 9223372036854775807)
  %tmp82 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_408", align 8
  %tmp83 = tail call ptr @objc_msgSend(ptr %tmp72, ptr %tmp82)
  %tmp85 = tail call ptr @llvm.objc.retain(ptr %tmp83) nounwind
  %tmp86 = load ptr, ptr @"\01L_OBJC_CLASSLIST_REFERENCES_$_409", align 8
  %tmp88 = load ptr, ptr @"\01l_objc_msgSend_fixup_alloc", align 16
  %tmp90 = tail call ptr %tmp88(ptr %tmp86, ptr @"\01l_objc_msgSend_fixup_alloc")
  %tmp91 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_8", align 8
  %tmp92 = tail call ptr @objc_msgSend(ptr %tmp90, ptr %tmp91)
  %tmp93 = tail call ptr @llvm.objc.explicit_autorelease(ptr %tmp92) nounwind
  %tmp95 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_411", align 8
  tail call void @objc_msgSend(ptr %tmp85, ptr %tmp95, ptr %tmp93)
  tail call void @llvm.objc.release(ptr %tmp93) nounwind
  %tmp96 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_148", align 8
  %tmp97 = tail call signext i8 @objc_msgSend(ptr %arg, ptr %tmp96)
  %tmp98 = icmp eq i8 %tmp97, 0
  br i1 %tmp98, label %bb99, label %bb104

bb99:                                             ; preds = %bb57
  %tmp100 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_413", align 8
  %tmp101 = tail call i64 @objc_msgSend(ptr %tmp85, ptr %tmp100)
  %tmp102 = or i64 %tmp101, 12
  %tmp103 = load ptr, ptr @"\01L_OBJC_SELECTOR_REFERENCES_415", align 8
  tail call void @objc_msgSend(ptr %tmp85, ptr %tmp103, i64 %tmp102)
  br label %bb104

bb104:                                            ; preds = %bb99, %bb57
  %tmp105 = call ptr @llvm.objc.autorelease(ptr %tmp72) nounwind
  tail call void @llvm.objc.release(ptr %tmp85) nounwind
  tail call void @llvm.objc.release(ptr %tmp47) nounwind
  ret ptr %tmp105
}

; CHECK: attributes [[NUW]] = { nounwind }

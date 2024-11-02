; RUN: opt -S -passes=objc-arc-contract < %s | FileCheck %s

; The optimizer should be able to move the autorelease past a control triangle
; and various scary looking things and fold it into an objc_retainAutorelease.

; CHECK: bb57:
; CHECK: tail call i8* @llvm.objc.retainAutorelease(i8* %tmp71x) [[NUW:#[0-9]+]]
; CHECK: bb99:

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin11.0.0"

%0 = type { i8* (i8*, %1*, ...)*, i8* }
%1 = type { i8*, i8* }
%2 = type { %2*, %2*, %3*, i8* (i8*, i8*)**, %4* }
%3 = type opaque
%4 = type { i32, i32, i32, i8*, i8*, %5*, %7*, %10*, i8*, %9* }
%5 = type { i32, i32, [0 x %6] }
%6 = type { i8*, i8*, i8* }
%7 = type { i64, [0 x %8*] }
%8 = type { i8*, i8*, %7*, %5*, %5*, %5*, %5*, %9*, i32, i32 }
%9 = type { i32, i32, [0 x %1] }
%10 = type { i32, i32, [0 x %11] }
%11 = type { i64*, i8*, i8*, i32, i32 }
%12 = type { i32*, i32, i8*, i64 }
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
@"\01L_OBJC_SELECTOR_REFERENCES_8" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_3725" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_CLASSLIST_REFERENCES_$_40" = external hidden global %2*, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_SELECTOR_REFERENCES_4227" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_4631" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_CLASSLIST_REFERENCES_$_70" = external hidden global %2*, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_SELECTOR_REFERENCES_148" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_159" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_188" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_328" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01l_objc_msgSend_fixup_objectAtIndex_" = external hidden global %0, section "__DATA, __objc_msgrefs, coalesced", align 16
@_unnamed_cfstring_386 = external hidden constant %12, section "__DATA,__cfstring"
@"\01l_objc_msgSend_fixup_count" = external hidden global %0, section "__DATA, __objc_msgrefs, coalesced", align 16
@"\01L_OBJC_SELECTOR_REFERENCES_389" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_391" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_393" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@NSPrintHeaderAndFooter = external constant %13*
@"\01L_OBJC_SELECTOR_REFERENCES_395" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_CLASSLIST_REFERENCES_$_396" = external hidden global %2*, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_SELECTOR_REFERENCES_398" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_400" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_402" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_404" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_406" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_408" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_CLASSLIST_REFERENCES_$_409" = external hidden global %2*, section "__DATA, __objc_classrefs, regular, no_dead_strip", align 8
@"\01L_OBJC_SELECTOR_REFERENCES_411" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_413" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"
@"\01L_OBJC_SELECTOR_REFERENCES_415" = external hidden global i8*, section "__DATA, __objc_selrefs, literal_pointers, no_dead_strip"

declare i8* @objc_msgSend(i8*, i8*, ...)

declare i8* @llvm.objc.retain(i8*)

declare void @llvm.objc.release(i8*)

declare i8* @llvm.objc.autorelease(i8*)

declare i8* @llvm.objc.explicit_autorelease(i8*)

define hidden %14* @foo(%15* %arg, %16* %arg2) {
bb:
  %tmp = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_3725", align 8
  %tmp4 = bitcast %15* %arg to i8*
  %tmp5 = tail call %18* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %18* (i8*, i8*)*)(i8* %tmp4, i8* %tmp)
  %tmp6 = bitcast %18* %tmp5 to i8*
  %tmp7 = tail call i8* @llvm.objc.retain(i8* %tmp6) nounwind
  %tmp8 = load %2*, %2** @"\01L_OBJC_CLASSLIST_REFERENCES_$_40", align 8
  %tmp9 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_4227", align 8
  %tmp10 = bitcast %2* %tmp8 to i8*
  %tmp11 = tail call %19* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %19* (i8*, i8*)*)(i8* %tmp10, i8* %tmp9)
  %tmp12 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_4631", align 8
  %tmp13 = bitcast %19* %tmp11 to i8*
  %tmp14 = tail call signext i8 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8 (i8*, i8*, %13*)*)(i8* %tmp13, i8* %tmp12, %13* bitcast (%12* @_unnamed_cfstring_386 to %13*))
  %tmp15 = bitcast %16* %arg2 to i8*
  %tmp16 = load i8*, i8** bitcast (%0* @"\01l_objc_msgSend_fixup_count" to i8**), align 16
  %tmp17 = bitcast i8* %tmp16 to i64 (i8*, %1*)*
  %tmp18 = tail call i64 %tmp17(i8* %tmp15, %1* bitcast (%0* @"\01l_objc_msgSend_fixup_count" to %1*))
  %tmp19 = icmp eq i64 %tmp18, 0
  br i1 %tmp19, label %bb22, label %bb20

bb20:                                             ; preds = %bb
  %tmp21 = icmp eq i8 %tmp14, 0
  br label %bb25

bb22:                                             ; preds = %bb
  %tmp23 = bitcast i8* %tmp7 to %18*
  %tmp24 = icmp eq i8 %tmp14, 0
  br i1 %tmp24, label %bb46, label %bb25

bb25:                                             ; preds = %bb22, %bb20
  %tmp26 = phi i1 [ %tmp21, %bb20 ], [ false, %bb22 ]
  %tmp27 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_188", align 8
  %tmp28 = tail call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* %tmp7, i8* %tmp27)
  %tmp29 = tail call i8* @llvm.objc.explicit_autorelease(i8* %tmp28) nounwind
  %tmp30 = bitcast i8* %tmp29 to %18*
  tail call void @llvm.objc.release(i8* %tmp7) nounwind
  %tmp31 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_389", align 8
  %tmp32 = tail call %20* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %20* (i8*, i8*)*)(i8* %tmp29, i8* %tmp31)
  %tmp33 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_391", align 8
  %tmp34 = bitcast %20* %tmp32 to i8*
  tail call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, %16*)*)(i8* %tmp34, i8* %tmp33, %16* %arg2)
  br i1 %tmp26, label %bb46, label %bb35

bb35:                                             ; preds = %bb25
  %tmp36 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_389", align 8
  %tmp37 = tail call %20* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %20* (i8*, i8*)*)(i8* %tmp29, i8* %tmp36)
  %tmp38 = load %2*, %2** @"\01L_OBJC_CLASSLIST_REFERENCES_$_70", align 8
  %tmp39 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_393", align 8
  %tmp40 = bitcast %2* %tmp38 to i8*
  %tmp41 = tail call %21* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %21* (i8*, i8*, i8)*)(i8* %tmp40, i8* %tmp39, i8 signext 1)
  %tmp42 = bitcast %21* %tmp41 to i8*
  %tmp43 = load %13*, %13** @NSPrintHeaderAndFooter, align 8
  %tmp44 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_159", align 8
  %tmp45 = bitcast %20* %tmp37 to i8*
  tail call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8*, %13*)*)(i8* %tmp45, i8* %tmp44, i8* %tmp42, %13* %tmp43)
  br label %bb46

bb46:                                             ; preds = %bb35, %bb25, %bb22
  %tmp47 = phi %18* [ %tmp30, %bb35 ], [ %tmp30, %bb25 ], [ %tmp23, %bb22 ]
  %tmp48 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_328", align 8
  %tmp49 = tail call %22* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %22* (i8*, i8*)*)(i8* %tmp4, i8* %tmp48)
  %tmp50 = bitcast %22* %tmp49 to i8*
  %tmp51 = load i8*, i8** bitcast (%0* @"\01l_objc_msgSend_fixup_count" to i8**), align 16
  %tmp52 = bitcast i8* %tmp51 to i64 (i8*, %1*)*
  %tmp53 = tail call i64 %tmp52(i8* %tmp50, %1* bitcast (%0* @"\01l_objc_msgSend_fixup_count" to %1*))
  %tmp54 = icmp eq i64 %tmp53, 0
  br i1 %tmp54, label %bb55, label %bb57

bb55:                                             ; preds = %bb46
  %tmp56 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_395", align 8
  tail call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*)*)(i8* %tmp4, i8* %tmp56)
  br label %bb57

bb57:                                             ; preds = %bb55, %bb46
  %tmp58 = load %2*, %2** @"\01L_OBJC_CLASSLIST_REFERENCES_$_396", align 8
  %tmp59 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_328", align 8
  %tmp60 = tail call %22* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %22* (i8*, i8*)*)(i8* %tmp4, i8* %tmp59)
  %tmp61 = bitcast %22* %tmp60 to i8*
  %tmp62 = load i8*, i8** bitcast (%0* @"\01l_objc_msgSend_fixup_objectAtIndex_" to i8**), align 16
  %tmp63 = bitcast i8* %tmp62 to i8* (i8*, %1*, i64)*
  %tmp64 = tail call i8* %tmp63(i8* %tmp61, %1* bitcast (%0* @"\01l_objc_msgSend_fixup_objectAtIndex_" to %1*), i64 0)
  %tmp65 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_398", align 8
  %tmp66 = tail call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* %tmp64, i8* %tmp65)
  %tmp67 = bitcast i8* %tmp66 to %23*
  %tmp68 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_400", align 8
  %tmp69 = bitcast %2* %tmp58 to i8*
  %tmp70 = tail call %14* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %14* (i8*, i8*, %23*, %18*)*)(i8* %tmp69, i8* %tmp68, %23* %tmp67, %18* %tmp47)
  %tmp71 = bitcast %14* %tmp70 to i8*
  ; hack to prevent the optimize from using objc_retainAutoreleasedReturnValue.
  %tmp71x = getelementptr i8, i8* %tmp71, i64 1
  %tmp72 = tail call i8* @llvm.objc.retain(i8* %tmp71x) nounwind
  %tmp73 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_402", align 8
  tail call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8)*)(i8* %tmp72, i8* %tmp73, i8 signext 1)
  %tmp74 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_404", align 8
  tail call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i8)*)(i8* %tmp72, i8* %tmp74, i8 signext 1)
  %tmp75 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_328", align 8
  %tmp76 = tail call %22* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %22* (i8*, i8*)*)(i8* %tmp4, i8* %tmp75)
  %tmp77 = bitcast %22* %tmp76 to i8*
  %tmp78 = load i8*, i8** bitcast (%0* @"\01l_objc_msgSend_fixup_objectAtIndex_" to i8**), align 16
  %tmp79 = bitcast i8* %tmp78 to i8* (i8*, %1*, i64)*
  %tmp80 = tail call i8* %tmp79(i8* %tmp77, %1* bitcast (%0* @"\01l_objc_msgSend_fixup_objectAtIndex_" to %1*), i64 0)
  %tmp81 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_406", align 8
  tail call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i64)*)(i8* %tmp80, i8* %tmp81, i64 9223372036854775807)
  %tmp82 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_408", align 8
  %tmp83 = tail call %24* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %24* (i8*, i8*)*)(i8* %tmp72, i8* %tmp82)
  %tmp84 = bitcast %24* %tmp83 to i8*
  %tmp85 = tail call i8* @llvm.objc.retain(i8* %tmp84) nounwind
  %tmp86 = load %2*, %2** @"\01L_OBJC_CLASSLIST_REFERENCES_$_409", align 8
  %tmp87 = bitcast %2* %tmp86 to i8*
  %tmp88 = load i8*, i8** bitcast (%0* @"\01l_objc_msgSend_fixup_alloc" to i8**), align 16
  %tmp89 = bitcast i8* %tmp88 to i8* (i8*, %1*)*
  %tmp90 = tail call i8* %tmp89(i8* %tmp87, %1* bitcast (%0* @"\01l_objc_msgSend_fixup_alloc" to %1*))
  %tmp91 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_8", align 8
  %tmp92 = tail call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* %tmp90, i8* %tmp91)
  %tmp93 = tail call i8* @llvm.objc.explicit_autorelease(i8* %tmp92) nounwind
  %tmp94 = bitcast i8* %tmp93 to %25*
  %tmp95 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_411", align 8
  tail call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, %25*)*)(i8* %tmp85, i8* %tmp95, %25* %tmp94)
  tail call void @llvm.objc.release(i8* %tmp93) nounwind
  %tmp96 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_148", align 8
  %tmp97 = tail call signext i8 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8 (i8*, i8*)*)(i8* %tmp4, i8* %tmp96)
  %tmp98 = icmp eq i8 %tmp97, 0
  br i1 %tmp98, label %bb99, label %bb104

bb99:                                             ; preds = %bb57
  %tmp100 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_413", align 8
  %tmp101 = tail call i64 bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i64 (i8*, i8*)*)(i8* %tmp85, i8* %tmp100)
  %tmp102 = or i64 %tmp101, 12
  %tmp103 = load i8*, i8** @"\01L_OBJC_SELECTOR_REFERENCES_415", align 8
  tail call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, i64)*)(i8* %tmp85, i8* %tmp103, i64 %tmp102)
  br label %bb104

bb104:                                            ; preds = %bb99, %bb57
  %tmp105 = call i8* @llvm.objc.autorelease(i8* %tmp72) nounwind
  %tmp106 = bitcast i8* %tmp105 to %14*
  tail call void @llvm.objc.release(i8* %tmp85) nounwind
  %tmp107 = bitcast %18* %tmp47 to i8*
  tail call void @llvm.objc.release(i8* %tmp107) nounwind
  ret %14* %tmp106
}

; CHECK: attributes [[NUW]] = { nounwind }

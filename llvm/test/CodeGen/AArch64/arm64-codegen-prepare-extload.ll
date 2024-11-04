; RUN: opt -codegenprepare < %s -mtriple=aarch64-apple-ios -S | FileCheck -enable-var-scope %s --check-prefix=OPTALL --check-prefix=OPT --check-prefix=NONSTRESS
; RUN: opt -codegenprepare < %s -mtriple=aarch64-apple-ios -S -stress-cgp-ext-ld-promotion | FileCheck -enable-var-scope %s --check-prefix=OPTALL --check-prefix=OPT --check-prefix=STRESS
; RUN: opt -codegenprepare < %s -mtriple=aarch64-apple-ios -S -disable-cgp-ext-ld-promotion | FileCheck -enable-var-scope %s --check-prefix=OPTALL --check-prefix=DISABLE

; CodeGenPrepare should move the zext into the block with the load
; so that SelectionDAG can select it with the load.
;
; OPTALL-LABEL: @foo
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %p
; OPTALL-NEXT: [[ZEXT:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; OPTALL: store i32 [[ZEXT]], ptr %q
; OPTALL: ret
define void @foo(ptr %p, ptr %q) {
entry:
  %t = load i8, ptr %p
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = zext i8 %t to i32
  store i32 %s, ptr %q
  ret void
false:
  ret void
}

; Check that we manage to form a zextload is an operation with only one
; argument to explicitly extend is in the way.
; OPTALL-LABEL: @promoteOneArg
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %p
; OPT-NEXT: [[ZEXT:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; OPT-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nuw i32 [[ZEXT]], 2
; Make sure the operation is not promoted when the promotion pass is disabled.
; DISABLE: [[ADD:%[a-zA-Z_0-9-]+]] = add nuw i8 [[LD]], 2
; DISABLE: [[RES:%[a-zA-Z_0-9-]+]] = zext i8 [[ADD]] to i32
; OPTALL: store i32 [[RES]], ptr %q
; OPTALL: ret
define void @promoteOneArg(ptr %p, ptr %q) {
entry:
  %t = load i8, ptr %p
  %add = add nuw i8 %t, 2
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = zext i8 %add to i32
  store i32 %s, ptr %q
  ret void
false:
  ret void
}

; Check that we manage to form a sextload is an operation with only one
; argument to explicitly extend is in the way.
; Version with sext.
; OPTALL-LABEL: @promoteOneArgSExt
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %p
; OPT-NEXT: [[SEXT:%[a-zA-Z_0-9-]+]] = sext i8 [[LD]] to i32
; OPT-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nsw i32 [[SEXT]], 2
; DISABLE: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i8 [[LD]], 2
; DISABLE: [[RES:%[a-zA-Z_0-9-]+]] = sext i8 [[ADD]] to i32
; OPTALL: store i32 [[RES]], ptr %q
; OPTALL: ret
define void @promoteOneArgSExt(ptr %p, ptr %q) {
entry:
  %t = load i8, ptr %p
  %add = add nsw i8 %t, 2
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = sext i8 %add to i32
  store i32 %s, ptr %q
  ret void
false:
  ret void
}

; Check that we manage to form a zextload is an operation with two
; arguments to explicitly extend is in the way.
; Extending %add will create two extensions:
; 1. One for %b.
; 2. One for %t.
; #1 will not be removed as we do not know anything about %b.
; #2 may not be merged with the load because %t is used in a comparison.
; Since two extensions may be emitted in the end instead of one before the
; transformation, the regular heuristic does not apply the optimization.
;
; OPTALL-LABEL: @promoteTwoArgZext
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %p
;
; STRESS-NEXT: [[ZEXTLD:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; STRESS-NEXT: [[ZEXTB:%[a-zA-Z_0-9-]+]] = zext i8 %b to i32
; STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nuw i32 [[ZEXTLD]], [[ZEXTB]]
;
; NONSTRESS: [[ADD:%[a-zA-Z_0-9-]+]] = add nuw i8 [[LD]], %b
; NONSTRESS: [[RES:%[a-zA-Z_0-9-]+]] = zext i8 [[ADD]] to i32
;
; DISABLE: [[ADD:%[a-zA-Z_0-9-]+]] = add nuw i8 [[LD]], %b
; DISABLE: [[RES:%[a-zA-Z_0-9-]+]] = zext i8 [[ADD]] to i32
;
; OPTALL: store i32 [[RES]], ptr %q
; OPTALL: ret
define void @promoteTwoArgZext(ptr %p, ptr %q, i8 %b) {
entry:
  %t = load i8, ptr %p
  %add = add nuw i8 %t, %b
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = zext i8 %add to i32
  store i32 %s, ptr %q
  ret void
false:
  ret void
}

; Check that we manage to form a sextload is an operation with two
; arguments to explicitly extend is in the way.
; Version with sext.
; OPTALL-LABEL: @promoteTwoArgSExt
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %p
;
; STRESS-NEXT: [[SEXTLD:%[a-zA-Z_0-9-]+]] = sext i8 [[LD]] to i32
; STRESS-NEXT: [[SEXTB:%[a-zA-Z_0-9-]+]] = sext i8 %b to i32
; STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nsw i32 [[SEXTLD]], [[SEXTB]]
;
; NONSTRESS: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i8 [[LD]], %b
; NONSTRESS: [[RES:%[a-zA-Z_0-9-]+]] = sext i8 [[ADD]] to i32
;
; DISABLE: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i8 [[LD]], %b
; DISABLE: [[RES:%[a-zA-Z_0-9-]+]] = sext i8 [[ADD]] to i32
; OPTALL: store i32 [[RES]], ptr %q
; OPTALL: ret
define void @promoteTwoArgSExt(ptr %p, ptr %q, i8 %b) {
entry:
  %t = load i8, ptr %p
  %add = add nsw i8 %t, %b
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = sext i8 %add to i32
  store i32 %s, ptr %q
  ret void
false:
  ret void
}

; Check that we do not a zextload if we need to introduce more than
; one additional extension.
; OPTALL-LABEL: @promoteThreeArgZext
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %p
;
; STRESS-NEXT: [[ZEXTLD:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; STRESS-NEXT: [[ZEXTB:%[a-zA-Z_0-9-]+]] = zext i8 %b to i32
; STRESS-NEXT: [[TMP:%[a-zA-Z_0-9-]+]] = add nuw i32 [[ZEXTLD]], [[ZEXTB]]
; STRESS-NEXT: [[ZEXTC:%[a-zA-Z_0-9-]+]] = zext i8 %c to i32
; STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nuw i32 [[TMP]], [[ZEXTC]]
;
; NONSTRESS-NEXT: [[TMP:%[a-zA-Z_0-9-]+]] = add nuw i8 [[LD]], %b
; NONSTRESS-NEXT: [[ADD:%[a-zA-Z_0-9-]+]] = add nuw i8 [[TMP]], %c
; NONSTRESS: [[RES:%[a-zA-Z_0-9-]+]] = zext i8 [[ADD]] to i32
;
; DISABLE: add nuw i8
; DISABLE: [[ADD:%[a-zA-Z_0-9-]+]] = add nuw i8
; DISABLE: [[RES:%[a-zA-Z_0-9-]+]] = zext i8 [[ADD]] to i32
;
; OPTALL: store i32 [[RES]], ptr %q
; OPTALL: ret
define void @promoteThreeArgZext(ptr %p, ptr %q, i8 %b, i8 %c) {
entry:
  %t = load i8, ptr %p
  %tmp = add nuw i8 %t, %b
  %add = add nuw i8 %tmp, %c
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = zext i8 %add to i32
  store i32 %s, ptr %q
  ret void
false:
  ret void
}

; Check that we manage to form a zextload after promoting and merging
; two extensions.
; OPTALL-LABEL: @promoteMergeExtArgZExt
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %p
;
; STRESS-NEXT: [[ZEXTLD:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; STRESS-NEXT: [[ZEXTB:%[a-zA-Z_0-9-]+]] = zext i16 %b to i32
; STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nuw i32 [[ZEXTLD]], [[ZEXTB]]
;
; NONSTRESS: [[ZEXTLD:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i16
; NONSTRESS: [[ADD:%[a-zA-Z_0-9-]+]] = add nuw i16 [[ZEXTLD]], %b
; NONSTRESS: [[RES:%[a-zA-Z_0-9-]+]] = zext i16 [[ADD]] to i32
;
; DISABLE: [[ZEXTLD:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i16
; DISABLE: [[ADD:%[a-zA-Z_0-9-]+]] = add nuw i16 [[ZEXTLD]], %b
; DISABLE: [[RES:%[a-zA-Z_0-9-]+]] = zext i16 [[ADD]] to i32
;
; OPTALL: store i32 [[RES]], ptr %q
; OPTALL: ret
define void @promoteMergeExtArgZExt(ptr %p, ptr %q, i16 %b) {
entry:
  %t = load i8, ptr %p
  %ext = zext i8 %t to i16
  %add = add nuw i16 %ext, %b
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = zext i16 %add to i32
  store i32 %s, ptr %q
  ret void
false:
  ret void
}

; Check that we manage to form a sextload after promoting and merging
; two extensions.
; Version with sext.
; OPTALL-LABEL: @promoteMergeExtArgSExt
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %p
;
; STRESS-NEXT: [[ZEXTLD:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; STRESS-NEXT: [[ZEXTB:%[a-zA-Z_0-9-]+]] = sext i16 %b to i32
; STRESS-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nsw i32 [[ZEXTLD]], [[ZEXTB]]
;
; NONSTRESS: [[ZEXTLD:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i16
; NONSTRESS: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i16 [[ZEXTLD]], %b
; NONSTRESS: [[RES:%[a-zA-Z_0-9-]+]] = sext i16 [[ADD]] to i32
;
; DISABLE: [[ZEXTLD:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i16
; DISABLE: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i16 [[ZEXTLD]], %b
; DISABLE: [[RES:%[a-zA-Z_0-9-]+]] = sext i16 [[ADD]] to i32
; OPTALL: store i32 [[RES]], ptr %q
; OPTALL: ret
define void @promoteMergeExtArgSExt(ptr %p, ptr %q, i16 %b) {
entry:
  %t = load i8, ptr %p
  %ext = zext i8 %t to i16
  %add = add nsw i16 %ext, %b
  %a = icmp slt i8 %t, 20
  br i1 %a, label %true, label %false
true:
  %s = sext i16 %add to i32
  store i32 %s, ptr %q
  ret void
false:
  ret void
}

; Check that we manage to catch all the extload opportunities that are exposed
; by the different iterations of codegen prepare.
; Moreover, check that we do not promote more than we need to.
; Here is what is happening in this test (not necessarly in this order):
; 1. We try to promote the operand of %sextadd.
;    a. This creates one sext of %ld2 and one of %zextld
;    b. The sext of %ld2 can be combine with %ld2, so we remove one sext but
;       introduced one. This is fine with the current heuristic: neutral.
;    => We have one zext of %zextld left and we created one sext of %ld2.
; 2. We try to promote the operand of %sextaddza.
;    a. This creates one sext of %zexta and one of %zextld
;    b. The sext of %zexta can be combined with the zext of %a.
;    c. The sext of %zextld leads to %ld and can be combined with it. This is
;       done by promoting %zextld. This is fine with the current heuristic:
;       neutral.
;    => We have created a new zext of %ld and we created one sext of %zexta.
; 3. We try to promote the operand of %sextaddb.
;    a. This creates one sext of %b and one of %zextld
;    b. The sext of %b is a dead-end, nothing to be done.
;    c. Same thing as 2.c. happens.
;    => We have created a new zext of %ld and we created one sext of %b.
; 4. We try to promote the operand of the zext of %zextld introduced in #1.
;    a. Same thing as 2.c. happens.
;    b. %zextld does not have any other uses. It is dead coded.
;    => We have created a new zext of %ld and we removed a zext of %zextld and
;       a zext of %ld.
; Currently we do not try to reuse existing extensions, so in the end we have
; 3 identical zext of %ld. The extensions will be CSE'ed by SDag.
;
; OPTALL-LABEL: @severalPromotions
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %addr1
; OPT-NEXT: [[ZEXTLD1_1:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i64
; OPT-NEXT: [[ZEXTLD1_2:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i64
; OPT-NEXT: [[LD2:%[a-zA-Z_0-9-]+]] = load i32, ptr %addr2
; OPT-NEXT: [[SEXTLD2:%[a-zA-Z_0-9-]+]] = sext i32 [[LD2]] to i64
; OPT-NEXT: [[ZEXTLD1_3:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i64
; OPT-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nsw i64 [[SEXTLD2]], [[ZEXTLD1_3]]
; OPT-NEXT: [[ZEXTLD1_4:%[a-zA-Z_0-9-]+]] = zext i8 %a to i64
; OPT-NEXT: [[RESZA:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ZEXTLD1_4]], [[ZEXTLD1_2]]
; OPT-NEXT: [[SEXTB:%[a-zA-Z_0-9-]+]] = sext i32 %b to i64
; OPT-NEXT: [[RESB:%[a-zA-Z_0-9-]+]] = add nsw i64 [[SEXTB]], [[ZEXTLD1_1]]
;
; DISABLE: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i32
; DISABLE: [[RES:%[a-zA-Z_0-9-]+]]  = sext i32 [[ADD]] to i64
; DISABLE: [[ADDZA:%[a-zA-Z_0-9-]+]] = add nsw i32
; DISABLE: [[RESZA:%[a-zA-Z_0-9-]+]]  = sext i32 [[ADDZA]] to i64
; DISABLE: [[ADDB:%[a-zA-Z_0-9-]+]] = add nsw i32
; DISABLE: [[RESB:%[a-zA-Z_0-9-]+]]  = sext i32 [[ADDB]] to i64
;
; OPTALL: call void @dummy(i64 [[RES]], i64 [[RESZA]], i64 [[RESB]])
; OPTALL: ret
define void @severalPromotions(ptr %addr1, ptr %addr2, i8 %a, i32 %b) {
  %ld = load i8, ptr %addr1
  %zextld = zext i8 %ld to i32
  %ld2 = load i32, ptr %addr2
  %add = add nsw i32 %ld2, %zextld
  %sextadd = sext i32 %add to i64
  %zexta = zext i8 %a to i32
  %addza = add nsw i32 %zexta, %zextld
  %sextaddza = sext i32 %addza to i64
  %addb = add nsw i32 %b, %zextld
  %sextaddb = sext i32 %addb to i64
  call void @dummy(i64 %sextadd, i64 %sextaddza, i64 %sextaddb)
  ret void
}

declare void @dummy(i64, i64, i64)

; Make sure we do not try to promote vector types since the type promotion
; helper does not support them for now.
; OPTALL-LABEL: @vectorPromotion
; OPTALL: [[SHL:%[a-zA-Z_0-9-]+]] = shl nuw nsw <2 x i32> zeroinitializer, <i32 8, i32 8>
; OPTALL: [[ZEXT:%[a-zA-Z_0-9-]+]] = zext <2 x i32> [[SHL]] to <2 x i64>
; OPTALL: ret
define void @vectorPromotion() {
entry:
  %a = shl nuw nsw <2 x i32> zeroinitializer, <i32 8, i32 8>
  %b = zext <2 x i32> %a to <2 x i64>
  ret void
}

@a = common global i32 0, align 4
@c = common global [2 x i32] zeroinitializer, align 4

; Make sure we support promotion of operands that produces a Value as opposed
; to an instruction.
; This used to cause a crash.
; OPTALL-LABEL: @promotionOfArgEndsUpInValue
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i16, ptr %addr
;
; OPT-NEXT: [[SEXT:%[a-zA-Z_0-9-]+]] = sext i16 [[LD]] to i32
; OPT-NEXT: [[SEXT2:%[a-zA-Z_0-9-]+]] = zext i1 icmp ne (ptr getelementptr inbounds ([2 x i32], ptr @c, i64 0, i64 1), ptr @a) to i32
; OPT-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nuw nsw i32 [[SEXT]], [[SEXT2]]
;
; DISABLE-NEXT: [[EXT:%[a-zA-Z_0-9-]+]] = zext i1 icmp ne (ptr getelementptr inbounds ([2 x i32], ptr @c, i64 0, i64 1), ptr @a) to i16
; DISABLE-NEXT: [[ADD:%[a-zA-Z_0-9-]+]] = add nuw nsw i16 [[LD]], [[EXT]]
; DISABLE-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = sext i16 [[ADD]] to i32
;
; OPTALL-NEXT: ret i32 [[RES]]
define i32 @promotionOfArgEndsUpInValue(ptr %addr) {
entry:
  %val = load i16, ptr %addr
  %ext = zext i1 icmp ne (ptr getelementptr inbounds ([2 x i32], ptr @c, i64 0, i64 1), ptr @a) to i16
  %add = add nuw nsw i16 %val, %ext
  %conv3 = sext i16 %add to i32
  ret i32 %conv3
}

; Check that we see that one zext can be derived from the other for free.
; OPTALL-LABEL: @promoteTwoArgZextWithSourceExtendedTwice
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %p
;
; OPT-NEXT: [[ZEXT64:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i64
; OPT-NEXT: [[ZEXT32:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; OPT-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = add nuw i32 [[ZEXT32]], %b
; OPT-NEXT: [[RES64:%[a-zA-Z_0-9-]+]] = add nuw i64 [[ZEXT64]], 12
; OPT-NEXT: store i32 [[RES32]], ptr %addr
; OPT-NEXT: store i64 [[RES64]], ptr %q
;
; DISABLE-NEXT: [[ZEXT32:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; DISABLE-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = add nuw i32 [[ZEXT32]], %b
; DISABLE-NEXT: [[RES2_32:%[a-zA-Z_0-9-]+]] = add nuw i32 [[ZEXT32]], 12
; DISABLE-NEXT: store i32 [[RES32]], ptr %addr
; DISABLE-NEXT: [[ZEXT64:%[a-zA-Z_0-9-]+]] = zext i32 [[RES2_32]] to i64
; DISABLE-NEXT: store i64 [[ZEXT64]], ptr %q
;
; OPTALL-NEXT: ret void
define void @promoteTwoArgZextWithSourceExtendedTwice(ptr %p, ptr %q, i32 %b, ptr %addr) {
entry:
  %t = load i8, ptr %p
  %zextt = zext i8 %t to i32
  %add = add nuw i32 %zextt, %b
  %add2 = add nuw i32 %zextt, 12
  store i32 %add, ptr %addr
  %s = zext i32 %add2 to i64
  store i64 %s, ptr %q
  ret void
}

; Check that we do not increase the cost of the code.
; The input has one free zext and one free sext. If we would have promoted
; all the way through the load we would end up with a free zext and a
; non-free sext (of %b).
; OPTALL-LABEL: @doNotPromoteFreeSExtFromAddrMode
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %p
;
; STRESS-NEXT: [[ZEXT64:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i64
; STRESS-NEXT: [[SEXTB:%[a-zA-Z_0-9-]+]] = sext i32 %b to i64
; STRESS-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ZEXT64]], [[SEXTB]]
; STRESS-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = trunc i64 [[IDX64]] to i32
;
; NONSTRESS-NEXT: [[ZEXT32:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; NONSTRESS-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = add nsw i32 [[ZEXT32]], %b
; NONSTRESS-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = sext i32 [[RES32]] to i64
;
; DISABLE-NEXT: [[ZEXT32:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; DISABLE-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = add nsw i32 [[ZEXT32]], %b
; DISABLE-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = sext i32 [[RES32]] to i64
;
; OPTALL-NEXT: [[GEP:%[a-zA-Z_0-9-]+]] = getelementptr inbounds i32, ptr %addr, i64 [[IDX64]]
; OPTALL-NEXT: store i32 [[RES32]], ptr [[GEP]]
; OPTALL-NEXT: ret void
define void @doNotPromoteFreeSExtFromAddrMode(ptr %p, i32 %b, ptr %addr) {
entry:
  %t = load i8, ptr %p
  %zextt = zext i8 %t to i32
  %add = add nsw i32 %zextt, %b
  %idx64 = sext i32 %add to i64
  %staddr = getelementptr inbounds i32, ptr %addr, i64 %idx64
  store i32 %add, ptr %staddr
  ret void
}

; Check that we do not increase the cost of the code.
; The input has one free zext and one free sext. If we would have promoted
; all the way through the load we would end up with a free zext and a
; non-free sext (of %b).
; OPTALL-LABEL: @doNotPromoteFreeSExtFromAddrMode64
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %p
;
; STRESS-NEXT: [[ZEXT64:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i64
; STRESS-NEXT: [[SEXTB:%[a-zA-Z_0-9-]+]] = sext i32 %b to i64
; STRESS-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ZEXT64]], [[SEXTB]]
;
; NONSTRESS-NEXT: [[ZEXT32:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; NONSTRESS-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = add nsw i32 [[ZEXT32]], %b
; NONSTRESS-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = sext i32 [[RES32]] to i64
;
; DISABLE-NEXT: [[ZEXT32:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; DISABLE-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = add nsw i32 [[ZEXT32]], %b
; DISABLE-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = sext i32 [[RES32]] to i64
;
; OPTALL-NEXT: [[GEP:%[a-zA-Z_0-9-]+]] = getelementptr inbounds i64, ptr %addr, i64 [[IDX64]]
; OPTALL-NEXT: store i64 %stuff, ptr [[GEP]]
; OPTALL-NEXT: ret void
define void @doNotPromoteFreeSExtFromAddrMode64(ptr %p, i32 %b, ptr %addr, i64 %stuff) {
entry:
  %t = load i8, ptr %p
  %zextt = zext i8 %t to i32
  %add = add nsw i32 %zextt, %b
  %idx64 = sext i32 %add to i64
  %staddr = getelementptr inbounds i64, ptr %addr, i64 %idx64
  store i64 %stuff, ptr %staddr
  ret void
}

; Check that we do not increase the cost of the code.
; The input has one free zext and one free sext. If we would have promoted
; all the way through the load we would end up with a free zext and a
; non-free sext (of %b).
; OPTALL-LABEL: @doNotPromoteFreeSExtFromAddrMode128
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %p
;
; STRESS-NEXT: [[ZEXT64:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i64
; STRESS-NEXT: [[SEXTB:%[a-zA-Z_0-9-]+]] = sext i32 %b to i64
; STRESS-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ZEXT64]], [[SEXTB]]
;
; NONSTRESS-NEXT: [[ZEXT32:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; NONSTRESS-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = add nsw i32 [[ZEXT32]], %b
; NONSTRESS-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = sext i32 [[RES32]] to i64
;
; DISABLE-NEXT: [[ZEXT32:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; DISABLE-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = add nsw i32 [[ZEXT32]], %b
; DISABLE-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = sext i32 [[RES32]] to i64
;
; OPTALL-NEXT: [[GEP:%[a-zA-Z_0-9-]+]] = getelementptr inbounds i128, ptr %addr, i64 [[IDX64]]
; OPTALL-NEXT: store i128 %stuff, ptr [[GEP]]
; OPTALL-NEXT: ret void
define void @doNotPromoteFreeSExtFromAddrMode128(ptr %p, i32 %b, ptr %addr, i128 %stuff) {
entry:
  %t = load i8, ptr %p
  %zextt = zext i8 %t to i32
  %add = add nsw i32 %zextt, %b
  %idx64 = sext i32 %add to i64
  %staddr = getelementptr inbounds i128, ptr %addr, i64 %idx64
  store i128 %stuff, ptr %staddr
  ret void
}


; Check that we do not increase the cost of the code.
; The input has one free zext and one free sext. If we would have promoted
; all the way through the load we would end up with a free zext and a
; non-free sext (of %b).
; OPTALL-LABEL: @promoteSExtFromAddrMode256
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %p
;
; OPT-NEXT: [[ZEXT64:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i64
; OPT-NEXT: [[SEXTB:%[a-zA-Z_0-9-]+]] = sext i32 %b to i64
; OPT-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ZEXT64]], [[SEXTB]]
;
; DISABLE-NEXT: [[ZEXT32:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; DISABLE-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = add nsw i32 [[ZEXT32]], %b
; DISABLE-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = sext i32 [[RES32]] to i64
;
; OPTALL-NEXT: [[GEP:%[a-zA-Z_0-9-]+]] = getelementptr inbounds i256, ptr %addr, i64 [[IDX64]]
; OPTALL-NEXT: store i256 %stuff, ptr [[GEP]]
; OPTALL-NEXT: ret void
define void @promoteSExtFromAddrMode256(ptr %p, i32 %b, ptr %addr, i256 %stuff) {
entry:
  %t = load i8, ptr %p
  %zextt = zext i8 %t to i32
  %add = add nsw i32 %zextt, %b
  %idx64 = sext i32 %add to i64
  %staddr = getelementptr inbounds i256, ptr %addr, i64 %idx64
  store i256 %stuff, ptr %staddr
  ret void
}

; Check that we do not increase the cost of the code.
; The input has one free zext and one free zext.
; When we promote all the way through the load, we end up with
; a free zext and a non-free zext (of %b).
; However, the current target lowering says zext i32 to i64 is free
; so the promotion happens because the cost did not change and may
; expose more opportunities.
; This would need to be fixed at some point.
; OPTALL-LABEL: @doNotPromoteFreeZExtFromAddrMode
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %p
;
; This transformation should really happen only for stress mode.
; OPT-NEXT: [[ZEXT64:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i64
; OPT-NEXT: [[ZEXTB:%[a-zA-Z_0-9-]+]] = zext i32 %b to i64
; OPT-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = add nuw i64 [[ZEXT64]], [[ZEXTB]]
; OPT-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = trunc i64 [[IDX64]] to i32
;
; DISABLE-NEXT: [[ZEXT32:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; DISABLE-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = add nuw i32 [[ZEXT32]], %b
; DISABLE-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = zext i32 [[RES32]] to i64
;
; OPTALL-NEXT: [[GEP:%[a-zA-Z_0-9-]+]] = getelementptr inbounds i32, ptr %addr, i64 [[IDX64]]
; OPTALL-NEXT: store i32 [[RES32]], ptr [[GEP]]
; OPTALL-NEXT: ret void
define void @doNotPromoteFreeZExtFromAddrMode(ptr %p, i32 %b, ptr %addr) {
entry:
  %t = load i8, ptr %p
  %zextt = zext i8 %t to i32
  %add = add nuw i32 %zextt, %b
  %idx64 = zext i32 %add to i64
  %staddr = getelementptr inbounds i32, ptr %addr, i64 %idx64
  store i32 %add, ptr %staddr
  ret void
}

; OPTALL-LABEL: @doNotPromoteFreeSExtFromShift
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %p
;
; STRESS-NEXT: [[ZEXT64:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i64
; STRESS-NEXT: [[SEXTB:%[a-zA-Z_0-9-]+]] = sext i32 %b to i64
; STRESS-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ZEXT64]], [[SEXTB]]
;
; NONSTRESS-NEXT: [[ZEXT32:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; NONSTRESS-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = add nsw i32 [[ZEXT32]], %b
; NONSTRESS-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = sext i32 [[RES32]] to i64
;
; DISABLE-NEXT: [[ZEXT32:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; DISABLE-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = add nsw i32 [[ZEXT32]], %b
; DISABLE-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = sext i32 [[RES32]] to i64
;
; OPTALL-NEXT: [[RES64:%[a-zA-Z_0-9-]+]] = shl i64 [[IDX64]], 12
; OPTALL-NEXT: ret i64 %staddr
define i64 @doNotPromoteFreeSExtFromShift(ptr %p, i32 %b) {
entry:
  %t = load i8, ptr %p
  %zextt = zext i8 %t to i32
  %add = add nsw i32 %zextt, %b
  %idx64 = sext i32 %add to i64
  %staddr = shl i64 %idx64, 12
  ret i64 %staddr
}

; Same comment as doNotPromoteFreeZExtFromAddrMode.
; OPTALL-LABEL: @doNotPromoteFreeZExtFromShift
; OPTALL: [[LD:%[a-zA-Z_0-9-]+]] = load i8, ptr %p
;
; This transformation should really happen only for stress mode.
; OPT-NEXT: [[ZEXT64:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i64
; OPT-NEXT: [[ZEXTB:%[a-zA-Z_0-9-]+]] = zext i32 %b to i64
; OPT-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = add nuw i64 [[ZEXT64]], [[ZEXTB]]
;
; DISABLE-NEXT: [[ZEXT32:%[a-zA-Z_0-9-]+]] = zext i8 [[LD]] to i32
; DISABLE-NEXT: [[RES32:%[a-zA-Z_0-9-]+]] = add nuw i32 [[ZEXT32]], %b
; DISABLE-NEXT: [[IDX64:%[a-zA-Z_0-9-]+]] = zext i32 [[RES32]] to i64
;
; OPTALL-NEXT: [[RES64:%[a-zA-Z_0-9-]+]] = shl i64 [[IDX64]], 12
; OPTALL-NEXT: ret i64 %staddr
define i64 @doNotPromoteFreeZExtFromShift(ptr %p, i32 %b) {
entry:
  %t = load i8, ptr %p
  %zextt = zext i8 %t to i32
  %add = add nuw i32 %zextt, %b
  %idx64 = zext i32 %add to i64
  %staddr = shl i64 %idx64, 12
  ret i64 %staddr
}

; The input has one free zext and one non-free sext.
; When we promote all the way through to the load, we end up with
; a free zext, a free sext (%ld1), and a non-free sext (of %cst).
; However, we when generate load pair and the free sext(%ld1) becomes
; non-free. So technically, we trade a non-free sext to two non-free
; sext.
; This would need to be fixed at some point.
; OPTALL-LABEL: @doNotPromoteBecauseOfPairedLoad
; OPTALL: [[LD0:%[a-zA-Z_0-9-]+]] = load i32, ptr %p
; OPTALL: [[GEP:%[a-zA-Z_0-9-]+]] = getelementptr inbounds i32, ptr %p, i64 1
; OPTALL: [[LD1:%[a-zA-Z_0-9-]+]] = load i32, ptr [[GEP]]
;
; This transformation should really happen only for stress mode.
; OPT-NEXT: [[SEXTLD1:%[a-zA-Z_0-9-]+]] = sext i32 [[LD1]] to i64
; OPT-NEXT: [[SEXTCST:%[a-zA-Z_0-9-]+]] = sext i32 %cst to i64
; OPT-NEXT: [[SEXTRES:%[a-zA-Z_0-9-]+]] = add nsw i64 [[SEXTLD1]], [[SEXTCST]]
;
; DISABLE-NEXT: [[RES:%[a-zA-Z_0-9-]+]] = add nsw i32 [[LD1]], %cst
; DISABLE-NEXT: [[SEXTRES:%[a-zA-Z_0-9-]+]] = sext i32 [[RES]] to i64
;
; OPTALL-NEXT: [[ZEXTLD0:%[a-zA-Z_0-9-]+]] = zext i32 [[LD0]] to i64
; OPTALL-NEXT: [[FINAL:%[a-zA-Z_0-9-]+]] = add i64 [[SEXTRES]], [[ZEXTLD0]]
; OPTALL-NEXT: ret i64 [[FINAL]]
define i64 @doNotPromoteBecauseOfPairedLoad(ptr %p, i32 %cst) {
  %ld0 = load i32, ptr %p
  %idxLd1 = getelementptr inbounds i32, ptr %p, i64 1
  %ld1 = load i32, ptr %idxLd1
  %res = add nsw i32 %ld1, %cst
  %sextres = sext i32 %res to i64
  %zextLd0 = zext i32 %ld0 to i64
  %final = add i64 %sextres, %zextLd0
  ret i64 %final
}

define i64 @promoteZextShl(i1 %c, ptr %P) {
entry:
; OPTALL-LABEL: promoteZextShl
; OPTALL: entry:
; OPT: %[[LD:.*]] = load i16, ptr %P
; OPT: %[[EXT:.*]] = zext i16 %[[LD]] to i64
; OPT: if.then:
; OPT: shl nsw i64 %[[EXT]], 1
; DISABLE: if.then:
; DISABLE: %r = sext i32 %shl2 to i64
  %ld = load i16, ptr %P
  br i1 %c, label %end, label %if.then
if.then:
  %z = zext i16 %ld to i32
  %shl2 = shl nsw i32 %z, 1
  %r = sext i32 %shl2 to i64
  ret i64 %r
end:
  ret i64 0
}

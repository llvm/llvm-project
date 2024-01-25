; RUN: opt -S -passes='require<profile-summary>,function(codegenprepare)' %s -o - | FileCheck %s
; This file tests the different cases what are involved when codegen prepare
; tries to get sign/zero extension out of the way of addressing mode.
; This tests require an actual target as addressing mode decisions depends
; on the target.

target datalayout = "e-i64:64-f80:128-s:64-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"


; Check that we correctly promote both operands of the promotable add.
; CHECK-LABEL: @twoArgsPromotion
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i32 %arg1 to i64
; CHECK: [[ARG2SEXT:%[a-zA-Z_0-9-]+]] = sext i32 %arg2 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT]], [[ARG2SEXT]]
; CHECK: inttoptr i64 [[PROMOTED]] to ptr
; CHECK: ret
define i8 @twoArgsPromotion(i32 %arg1, i32 %arg2) {
  %add = add nsw i32 %arg1, %arg2 
  %sextadd = sext i32 %add to i64
  %base = inttoptr i64 %sextadd to ptr
  %res = load i8, ptr %base
  ret i8 %res
}

; Check that we do not promote both operands of the promotable add when
; the instruction will not be folded into the addressing mode.
; Otherwise, we will increase the number of instruction executed.
; (This is a heuristic of course, because the new sext could have been
; merged with something else.)
; CHECK-LABEL: @twoArgsNoPromotion
; CHECK: add nsw i32 %arg1, %arg2
; CHECK: ret
define i8 @twoArgsNoPromotion(i32 %arg1, i32 %arg2, ptr %base) {
  %add = add nsw i32 %arg1, %arg2 
  %sextadd = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %sextadd
  %res = load i8, ptr %arrayidx
  ret i8 %res
}

; Check that we do not promote when the related instruction does not have
; the nsw flag.
; CHECK-LABEL: @noPromotion
; CHECK-NOT: add i64
; CHECK: ret
define i8 @noPromotion(i32 %arg1, i32 %arg2, ptr %base) {
  %add = add i32 %arg1, %arg2 
  %sextadd = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %sextadd
  %res = load i8, ptr %arrayidx
  ret i8 %res
}

; Check that we correctly promote constant arguments.
; CHECK-LABEL: @oneArgPromotion
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i32 %arg1 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT]], 1
; CHECK: getelementptr inbounds i8, ptr %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotion(i32 %arg1, ptr %base) {
  %add = add nsw i32 %arg1, 1 
  %sextadd = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %sextadd
  %res = load i8, ptr %arrayidx
  ret i8 %res
}

; Check that we are able to merge a sign extension with a zero extension.
; CHECK-LABEL: @oneArgPromotionZExt
; CHECK: [[ARG1ZEXT:%[a-zA-Z_0-9-]+]] = zext i8 %arg1 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1ZEXT]], 1
; CHECK: getelementptr inbounds i8, ptr %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotionZExt(i8 %arg1, ptr %base) {
  %zext = zext i8 %arg1 to i32
  %add = add nsw i32 %zext, 1 
  %sextadd = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %sextadd
  %res = load i8, ptr %arrayidx
  ret i8 %res
}

; When promoting a constant zext, the IR builder returns a constant,
; not an instruction. Make sure this is properly handled. This used
; to crash.
; Note: The constant zext is promoted, but does not help matching
; more thing in the addressing mode. Therefore the modification is
; rolled back.
; Still, this test case exercises the desired code path.
; CHECK-LABEL: @oneArgPromotionCstZExt
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 0, 1
; CHECK: getelementptr inbounds i8, ptr %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotionCstZExt(ptr %base) {
  %cst = zext i16 undef to i32
  %add = add nsw i32 %cst, 1
  %sextadd = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %sextadd
  %res = load i8, ptr %arrayidx
  ret i8 %res
}

; Check that we do not promote truncate when we cannot determine the
; bits that are dropped.
; CHECK-LABEL: @oneArgPromotionBlockTrunc1
; CHECK: [[ARG1TRUNC:%[a-zA-Z_0-9-]+]] = trunc i32 %arg1 to i8
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i8 [[ARG1TRUNC]] to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT]], 1
; CHECK: getelementptr inbounds i8, ptr %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotionBlockTrunc1(i32 %arg1, ptr %base) {
  %trunc = trunc i32 %arg1 to i8
  %add = add nsw i8 %trunc, 1 
  %sextadd = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %sextadd
  %res = load i8, ptr %arrayidx
  ret i8 %res
}

; Check that we do not promote truncate when we cannot determine all the
; bits that are dropped.
; CHECK-LABEL: @oneArgPromotionBlockTrunc2
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i16 %arg1 to i32
; CHECK: [[ARG1TRUNC:%[a-zA-Z_0-9-]+]] = trunc i32 [[ARG1SEXT]] to i8
; CHECK: [[ARG1SEXT64:%[a-zA-Z_0-9-]+]] = sext i8 [[ARG1TRUNC]] to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT64]], 1
; CHECK: getelementptr inbounds i8, ptr %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotionBlockTrunc2(i16 %arg1, ptr %base) {
  %sextarg1 = sext i16 %arg1 to i32
  %trunc = trunc i32 %sextarg1 to i8
  %add = add nsw i8 %trunc, 1 
  %sextadd = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %sextadd
  %res = load i8, ptr %arrayidx
  ret i8 %res
}

; Check that we are able to promote truncate when we know all the bits
; that are dropped.
; CHECK-LABEL: @oneArgPromotionPassTruncKeepSExt
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i1 %arg1 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT]], 1
; CHECK: getelementptr inbounds i8, ptr %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotionPassTruncKeepSExt(i1 %arg1, ptr %base) {
  %sextarg1 = sext i1 %arg1 to i32
  %trunc = trunc i32 %sextarg1 to i8
  %add = add nsw i8 %trunc, 1 
  %sextadd = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %sextadd
  %res = load i8, ptr %arrayidx
  ret i8 %res
}

; On X86 truncate are free. Check that we are able to promote the add
; to be used as addressing mode and that we insert a truncate for the other
; use. 
; CHECK-LABEL: @oneArgPromotionTruncInsert
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i8 %arg1 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT]], 1
; CHECK: [[TRUNC:%[a-zA-Z_0-9-]+]] = trunc i64 [[PROMOTED]] to i8
; CHECK: [[GEP:%[a-zA-Z_0-9-]+]] = getelementptr inbounds i8, ptr %base, i64 [[PROMOTED]]
; CHECK: [[LOAD:%[a-zA-Z_0-9-]+]] = load i8, ptr [[GEP]]
; CHECK: add i8 [[LOAD]], [[TRUNC]]
; CHECK: ret
define i8 @oneArgPromotionTruncInsert(i8 %arg1, ptr %base) {
  %add = add nsw i8 %arg1, 1 
  %sextadd = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %sextadd
  %res = load i8, ptr %arrayidx
  %finalres = add i8 %res, %add
  ret i8 %finalres
}

; Cannot sext from a larger type than the promoted type.
; CHECK-LABEL: @oneArgPromotionLargerType
; CHECK: [[ARG1TRUNC:%[a-zA-Z_0-9-]+]] = trunc i128 %arg1 to i8
; CHECK: [[ARG1SEXT64:%[a-zA-Z_0-9-]+]] = sext i8 [[ARG1TRUNC]] to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT64]], 1
; CHECK: getelementptr inbounds i8, ptr %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotionLargerType(i128 %arg1, ptr %base) {
  %trunc = trunc i128 %arg1 to i8
  %add = add nsw i8 %trunc, 1 
  %sextadd = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %sextadd
  %res = load i8, ptr %arrayidx
  %finalres = add i8 %res, %add
  ret i8 %finalres
}

; Use same inserted trunc
; On X86 truncate are free. Check that we are able to promote the add
; to be used as addressing mode and that we insert a truncate for
; *all* the other uses. 
; CHECK-LABEL: @oneArgPromotionTruncInsertSeveralUse
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i8 %arg1 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT]], 1
; CHECK: [[TRUNC:%[a-zA-Z_0-9-]+]] = trunc i64 [[PROMOTED]] to i8
; CHECK: [[GEP:%[a-zA-Z_0-9-]+]] = getelementptr inbounds i8, ptr %base, i64 [[PROMOTED]]
; CHECK: [[LOAD:%[a-zA-Z_0-9-]+]] = load i8, ptr [[GEP]]
; CHECK: [[ADDRES:%[a-zA-Z_0-9-]+]] = add i8 [[LOAD]], [[TRUNC]]
; CHECK: add i8 [[ADDRES]], [[TRUNC]]
; CHECK: ret
define i8 @oneArgPromotionTruncInsertSeveralUse(i8 %arg1, ptr %base) {
  %add = add nsw i8 %arg1, 1 
  %sextadd = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %sextadd
  %res = load i8, ptr %arrayidx
  %almostfinalres = add i8 %res, %add
  %finalres = add i8 %almostfinalres, %add
  ret i8 %finalres
}

; Check that the promoted instruction is used for all uses of the original
; sign extension.
; CHECK-LABEL: @oneArgPromotionSExtSeveralUse
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i8 %arg1 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nsw i64 [[ARG1SEXT]], 1
; CHECK: [[GEP:%[a-zA-Z_0-9-]+]] = getelementptr inbounds i8, ptr %base, i64 [[PROMOTED]]
; CHECK: [[LOAD:%[a-zA-Z_0-9-]+]] = load i8, ptr [[GEP]]
; CHECK: [[ADDRES:%[a-zA-Z_0-9-]+]] = zext i8 [[LOAD]] to i64
; CHECK: add i64 [[ADDRES]], [[PROMOTED]]
; CHECK: ret
define i64 @oneArgPromotionSExtSeveralUse(i8 %arg1, ptr %base) {
  %add = add nsw i8 %arg1, 1 
  %sextadd = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %sextadd
  %res = load i8, ptr %arrayidx
  %almostfinalres = zext i8 %res to i64
  %finalres = add i64 %almostfinalres, %sextadd
  ret i64 %finalres
}

; Check all types of rollback mechanism.
; For this test, the sign extension stays in place.
; However, the matching process goes until promoting both the operands
; of the first promotable add implies.
; At this point the rollback mechanism kicks in and restores the states
; until the addressing mode matcher is able to match something: in that
; case promote nothing.
; Along the way, the promotion mechanism involves:
; - Mutating the type of %promotableadd1 and %promotableadd2.
; - Creating a sext for %arg1 and %arg2.
; - Creating a trunc for a use of %promotableadd1.
; - Replacing a bunch of uses.
; - Setting the operands of the promoted instruction with the promoted values.
; - Moving instruction around (mainly sext when promoting instruction).
; Each type of those promotions has to be undo at least once during this
; specific test. 
; CHECK-LABEL: @twoArgsPromotionNest
; CHECK: [[ORIG:%[a-zA-Z_0-9-]+]] = add nsw i32 %arg1, %arg2
; CHECK: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i32 [[ORIG]], [[ORIG]]
; CHECK: [[SEXT:%[a-zA-Z_0-9-]+]] = sext i32 [[ADD]] to i64
; CHECK: getelementptr inbounds i8, ptr %base, i64 [[SEXT]]
; CHECK: ret
define i8 @twoArgsPromotionNest(i32 %arg1, i32 %arg2, ptr %base) {
  %promotableadd1 = add nsw i32 %arg1, %arg2
  %promotableadd2 = add nsw i32 %promotableadd1, %promotableadd1 
  %sextadd = sext i32 %promotableadd2 to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %sextadd
  %res = load i8, ptr %arrayidx
  ret i8 %res
}

; Test the InstructionRemover undo, which was the only one not
; kicked in the previous test.
; The matcher first promotes the add, removes the trunc and promotes
; the sext of arg1.
; Then, the matcher cannot use an addressing mode r + r + r, thus it
; rolls back. 
; CHECK-LABEL: @twoArgsNoPromotionRemove
; CHECK: [[SEXTARG1:%[a-zA-Z_0-9-]+]] = sext i1 %arg1 to i32
; CHECK: [[TRUNC:%[a-zA-Z_0-9-]+]] = trunc i32 [[SEXTARG1]] to i8
; CHECK: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i8 [[TRUNC]], %arg2
; CHECK: [[SEXT:%[a-zA-Z_0-9-]+]] = sext i8 [[ADD]] to i64
; CHECK: getelementptr inbounds i8, ptr %base, i64 [[SEXT]]
; CHECK: ret
define i8 @twoArgsNoPromotionRemove(i1 %arg1, i8 %arg2, ptr %base) {
  %sextarg1 = sext i1 %arg1 to i32
  %trunc = trunc i32 %sextarg1 to i8
  %add = add nsw i8 %trunc, %arg2 
  %sextadd = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %sextadd
  %res = load i8, ptr %arrayidx
  ret i8 %res
}

; Ensure that when the profitability checks kicks in, the IR is not modified
; will IgnoreProfitability is on.
; The profitabily check happens when a candidate instruction has several uses.
; The matcher will create a new matcher for each use and check if the
; instruction is in the list of the matched instructions of this new matcher.
; All changes made by the new matchers must be dropped before pursuing
; otherwise the state of the original matcher will be wrong.
;
; Without the profitability check, when checking for the second use of
; arrayidx, the matcher promotes everything all the way to %arg1, %arg2.
; Check that we did not promote anything in the final matching.
;
; <rdar://problem/16020230>
; CHECK-LABEL: @checkProfitability
; CHECK-NOT: {{%[a-zA-Z_0-9-]+}} = sext i32 %arg1 to i64
; CHECK-NOT: {{%[a-zA-Z_0-9-]+}} = sext i32 %arg2 to i64
; CHECK: [[SHL:%[a-zA-Z_0-9-]+]] = shl nsw i32 %arg1, 1
; CHECK: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i32 [[SHL]], %arg2
; CHECK: [[SEXTADD:%[a-zA-Z_0-9-]+]] = sext i32 [[ADD]] to i64
; BB then
; CHECK: [[BASE1:%[a-zA-Z_0-9-]+]] = inttoptr i64 [[SEXTADD]] to ptr
; CHECK: [[FULL1:%[a-zA-Z_0-9-]+]] = getelementptr i8, ptr [[BASE1]], i64 48
; CHECK: load i32, ptr [[FULL1]]
; BB else
; CHECK: [[BASE2:%[a-zA-Z_0-9-]+]] = inttoptr i64 [[SEXTADD]] to ptr
; CHECK: [[FULL2:%[a-zA-Z_0-9-]+]] = getelementptr i8, ptr [[BASE2]], i64 48
; CHECK: load i32, ptr [[FULL2]]
; CHECK: ret
define i32 @checkProfitability(i32 %arg1, i32 %arg2, i1 %test) {
  %shl = shl nsw i32 %arg1, 1
  %add1 = add nsw i32 %shl, %arg2
  %sextidx1 = sext i32 %add1 to i64
  %tmpptr = inttoptr i64 %sextidx1 to ptr
  %arrayidx1 = getelementptr i32, ptr %tmpptr, i64 12
  br i1 %test, label %then, label %else
then: 
  %res1 = load i32, ptr %arrayidx1
  br label %end
else:
  %res2 = load i32, ptr %arrayidx1
  br label %end
end:
  %tmp = phi i32 [%res1, %then], [%res2, %else]
  %res = add i32 %tmp, %add1
  %addr = inttoptr i32 %res to ptr
  %final = load i32, ptr %addr
  ret i32 %final
}

%struct.dns_packet = type { i32, i32, %union.anon }
%union.anon = type { i32 }

@a = common global i32 0, align 4
@b = common global i16 0, align 2

; We used to crash on this function because we did not return the right
; promoted instruction for %conv.i.
; Make sure we generate the right code now.
; CHECK-LABEL: @fn3
; %conv.i is used twice and only one of its use is being promoted.
; Use it at the starting point for the matching.
; CHECK: %conv.i = zext i16 [[PLAIN_OPND:%[.a-zA-Z_0-9-]+]] to i32
; CHECK-NEXT: [[PROMOTED_CONV:%[.a-zA-Z_0-9-]+]] = zext i16 [[PLAIN_OPND]] to i64
; CHECK-NEXT: [[ADD:%[a-zA-Z_0-9-]+]] = getelementptr i8, ptr %P, i64 [[PROMOTED_CONV]]
; CHECK-NEXT: [[ADDR:%[a-zA-Z_0-9-]+]] = getelementptr i8, ptr [[ADD]], i64 7
; CHECK-NEXT: load i8, ptr [[ADDR]], align 1
define signext i16 @fn3(ptr nocapture readonly %P) {
entry:
  %tmp = getelementptr inbounds %struct.dns_packet, ptr %P, i64 0, i32 2
  br label %while.body.i.i

while.body.i.i:                                   ; preds = %while.body.i.i, %entry
  %src.addr.0.i.i = phi i16 [ 0, %entry ], [ %inc.i.i, %while.body.i.i ]
  %inc.i.i = add i16 %src.addr.0.i.i, 1
  %idxprom.i.i = sext i16 %src.addr.0.i.i to i64
  %arrayidx.i.i = getelementptr inbounds [0 x i8], ptr %tmp, i64 0, i64 %idxprom.i.i
  %tmp1 = load i8, ptr %arrayidx.i.i, align 1
  %conv2.i.i = zext i8 %tmp1 to i32
  %and.i.i = and i32 %conv2.i.i, 15
  store i32 %and.i.i, ptr @a, align 4
  %tobool.i.i = icmp eq i32 %and.i.i, 0
  br i1 %tobool.i.i, label %while.body.i.i, label %fn1.exit.i

fn1.exit.i:                                       ; preds = %while.body.i.i
  %inc.i.i.lcssa = phi i16 [ %inc.i.i, %while.body.i.i ]
  %conv.i = zext i16 %inc.i.i.lcssa to i32
  %sub.i = add nsw i32 %conv.i, -1
  %idxprom.i = sext i32 %sub.i to i64
  %arrayidx.i = getelementptr inbounds [0 x i8], ptr %tmp, i64 0, i64 %idxprom.i
  %tmp2 = load i8, ptr %arrayidx.i, align 1
  %conv2.i = sext i8 %tmp2 to i16
  store i16 %conv2.i, ptr @b, align 2
  %sub4.i = sub nsw i32 0, %conv.i
  %conv5.i = zext i16 %conv2.i to i32
  %cmp.i = icmp sgt i32 %conv5.i, %sub4.i
  br i1 %cmp.i, label %if.then.i, label %fn2.exit

if.then.i:                                        ; preds = %fn1.exit.i
  %end.i = getelementptr inbounds %struct.dns_packet, ptr %P, i64 0, i32 1
  %tmp3 = load i32, ptr %end.i, align 4
  %sub7.i = add i32 %tmp3, 65535
  %conv8.i = trunc i32 %sub7.i to i16
  br label %fn2.exit

fn2.exit:                                         ; preds = %if.then.i, %fn1.exit.i
  %retval.0.i = phi i16 [ %conv8.i, %if.then.i ], [ undef, %fn1.exit.i ]
  ret i16 %retval.0.i
}

; Check that we do not promote an extension if the non-wrapping flag does not
; match the kind of the extension.
; CHECK-LABEL: @noPromotionFlag
; CHECK: [[ADD:%[a-zA-Z_0-9-]+]] = add nsw i32 %arg1, %arg2
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = zext i32 [[ADD]] to i64
; CHECK: inttoptr i64 [[PROMOTED]] to ptr
; CHECK: ret
define i8 @noPromotionFlag(i32 %arg1, i32 %arg2) {
  %add = add nsw i32 %arg1, %arg2 
  %zextadd = zext i32 %add to i64
  %base = inttoptr i64 %zextadd to ptr
  %res = load i8, ptr %base
  ret i8 %res
}

; Check that we correctly promote both operands of the promotable add with zext.
; CHECK-LABEL: @twoArgsPromotionZExt
; CHECK: [[ARG1ZEXT:%[a-zA-Z_0-9-]+]] = zext i32 %arg1 to i64
; CHECK: [[ARG2ZEXT:%[a-zA-Z_0-9-]+]] = zext i32 %arg2 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nuw i64 [[ARG1ZEXT]], [[ARG2ZEXT]]
; CHECK: inttoptr i64 [[PROMOTED]] to ptr
; CHECK: ret
define i8 @twoArgsPromotionZExt(i32 %arg1, i32 %arg2) {
  %add = add nuw i32 %arg1, %arg2 
  %zextadd = zext i32 %add to i64
  %base = inttoptr i64 %zextadd to ptr
  %res = load i8, ptr %base
  ret i8 %res
}

; Check that we correctly promote constant arguments.
; CHECK-LABEL: @oneArgPromotionNegativeCstZExt
; CHECK: [[ARG1ZEXT:%[a-zA-Z_0-9-]+]] = zext i8 %arg1 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nuw i64 [[ARG1ZEXT]], 255
; CHECK: getelementptr inbounds i8, ptr %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotionNegativeCstZExt(i8 %arg1, ptr %base) {
  %add = add nuw i8 %arg1, -1 
  %zextadd = zext i8 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %zextadd
  %res = load i8, ptr %arrayidx
  ret i8 %res
}

; Check that we are able to merge two zero extensions.
; CHECK-LABEL: @oneArgPromotionZExtZExt
; CHECK: [[ARG1ZEXT:%[a-zA-Z_0-9-]+]] = zext i8 %arg1 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nuw i64 [[ARG1ZEXT]], 1
; CHECK: getelementptr inbounds i8, ptr %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotionZExtZExt(i8 %arg1, ptr %base) {
  %zext = zext i8 %arg1 to i32
  %add = add nuw i32 %zext, 1 
  %zextadd = zext i32 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %zextadd
  %res = load i8, ptr %arrayidx
  ret i8 %res
}

; Check that we do not promote truncate when the dropped bits
; are of a different kind.
; CHECK-LABEL: @oneArgPromotionBlockTruncZExt
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i1 %arg1 to i32
; CHECK: [[ARG1TRUNC:%[a-zA-Z_0-9-]+]] = trunc i32 [[ARG1SEXT]] to i8
; CHECK: [[ARG1ZEXT:%[a-zA-Z_0-9-]+]] = zext i8 [[ARG1TRUNC]] to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nuw i64 [[ARG1ZEXT]], 1
; CHECK: getelementptr inbounds i8, ptr %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotionBlockTruncZExt(i1 %arg1, ptr %base) {
  %sextarg1 = sext i1 %arg1 to i32
  %trunc = trunc i32 %sextarg1 to i8
  %add = add nuw i8 %trunc, 1 
  %zextadd = zext i8 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %zextadd
  %res = load i8, ptr %arrayidx
  ret i8 %res
}

; Check that we are able to promote truncate when we know all the bits
; that are dropped.
; CHECK-LABEL: @oneArgPromotionPassTruncZExt
; CHECK: [[ARG1ZEXT:%[a-zA-Z_0-9-]+]] = zext i1 %arg1 to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nuw i64 [[ARG1ZEXT]], 1
; CHECK: getelementptr inbounds i8, ptr %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotionPassTruncZExt(i1 %arg1, ptr %base) {
  %sextarg1 = zext i1 %arg1 to i32
  %trunc = trunc i32 %sextarg1 to i8
  %add = add nuw i8 %trunc, 1 
  %zextadd = zext i8 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %zextadd
  %res = load i8, ptr %arrayidx
  ret i8 %res
}

; Check that we do not promote sext with zext.
; CHECK-LABEL: @oneArgPromotionBlockSExtZExt
; CHECK: [[ARG1SEXT:%[a-zA-Z_0-9-]+]] = sext i1 %arg1 to i8
; CHECK: [[ARG1ZEXT:%[a-zA-Z_0-9-]+]] = zext i8 [[ARG1SEXT]] to i64
; CHECK: [[PROMOTED:%[a-zA-Z_0-9-]+]] = add nuw i64 [[ARG1ZEXT]], 1
; CHECK: getelementptr inbounds i8, ptr %base, i64 [[PROMOTED]]
; CHECK: ret
define i8 @oneArgPromotionBlockSExtZExt(i1 %arg1, ptr %base) {
  %sextarg1 = sext i1 %arg1 to i8
  %add = add nuw i8 %sextarg1, 1 
  %zextadd = zext i8 %add to i64
  %arrayidx = getelementptr inbounds i8, ptr %base, i64 %zextadd
  %res = load i8, ptr %arrayidx
  ret i8 %res
}

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr8 \
; RUN:   -ppc-asm-full-reg-names < %s | FileCheck %s --check-prefixes=AIX32,AIXDATA
; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr8 \
; RUN:   -ppc-asm-full-reg-names < %s | FileCheck %s --check-prefixes=AIX64,AIXDATA
; RUN: llc -verify-machineinstrs -mtriple powerpc64-unknown-linux -mcpu=pwr8 \
; RUN:   -ppc-asm-full-reg-names < %s | FileCheck %s --check-prefixes=LINUX64BE,LINUXDATA
; RUN: llc -verify-machineinstrs -mtriple powerpc64le-unknown-linux -mcpu=pwr8 \
; RUN:   -ppc-asm-full-reg-names < %s | FileCheck %s --check-prefixes=LINUX64LE,LINUXDATA


;; This @GLOBALSTRING is a user of @.str which causes @.str to not get pooled.
@.str = private unnamed_addr constant [47 x i8] c"This is the global string that is at the top.\0A\00", align 1
@GLOBALSTRING = dso_local local_unnamed_addr global ptr @.str, align 8

@IntArray2 = dso_local global [7 x i32] [i32 5, i32 7, i32 9, i32 11, i32 17, i32 1235, i32 32], align 4
@.str.1 = private unnamed_addr constant [12 x i8] c"str1_STRING\00", align 1
@.str.2 = private unnamed_addr constant [12 x i8] c"str2_STRING\00", align 1
@.str.3 = private unnamed_addr constant [12 x i8] c"str3_STRING\00", align 1
@.str.4 = private unnamed_addr constant [12 x i8] c"str4_STRING\00", align 1
@.str.5 = private unnamed_addr constant [183 x i8] c"longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_STRING\00", align 1
@__const.str6.TheString = private unnamed_addr constant [10 x i8] c"ABCABCABC\00", align 1
@.str.6 = private unnamed_addr constant [12 x i8] c"MixedString\00", align 1
@__const.mixed2.IntArray = private unnamed_addr constant [7 x i32] [i32 5, i32 7, i32 9, i32 11, i32 17, i32 1235, i32 32], align 4
@__const.IntArray3 = private unnamed_addr constant [14 x i64] [i64 15, i64 7, i64 19, i64 11, i64 17, i64 1235, i64 72, i64 51, i64 32, i64 231, i64 86, i64 64, i64 754, i64 281], align 8
@__const.IntArray4 = private unnamed_addr constant [14 x i64] [i64 15, i64 7, i64 19, i64 11, i64 17, i64 1235, i64 72, i64 51, i64 32, i64 231, i64 86, i64 64, i64 754, i64 281], align 8
@__const.IntArray5 = private unnamed_addr constant [17 x i64] [i64 15, i64 7, i64 19, i64 11, i64 17, i64 1235, i64 72, i64 51, i64 32, i64 231, i64 86, i64 64, i64 754, i64 281, i64 61, i64 63, i64 67], align 8
@.str.7 = private unnamed_addr constant [20 x i8] c"Different String 01\00", align 1
@.str.8 = private unnamed_addr constant [15 x i8] c"Static Global\0A\00", align 1

;; Special alignment of 128 on this string will force it to go first and have padding added.
;; TODO: At the momment these will not be pooled because the extra alignment may be lost.
@.str.9 = private unnamed_addr constant [17 x i8] c"str9_STRING.....\00", align 128
@.str.10 = private unnamed_addr constant [17 x i8] c"str10_STRING....\00", align 128

;; Undef constant.
@.str.11 = private unnamed_addr constant [10 x i8] undef, align 1

define dso_local signext i32 @str1() local_unnamed_addr #0 {
; AIX32-LABEL: str1:
; AIX32:       # %bb.0: # %entry
; AIX32-NEXT:    mflr r0
; AIX32-NEXT:    stwu r1, -64(r1)
; AIX32-NEXT:    lwz r3, L..C0(r2) # @_MergedGlobals
; AIX32-NEXT:    stw r0, 72(r1)
; AIX32-NEXT:    addi r3, r3, 20
; AIX32-NEXT:    bl .callee[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    addi r1, r1, 64
; AIX32-NEXT:    lwz r0, 8(r1)
; AIX32-NEXT:    mtlr r0
; AIX32-NEXT:    blr
;
; AIX64-LABEL: str1:
; AIX64:       # %bb.0: # %entry
; AIX64-NEXT:    mflr r0
; AIX64-NEXT:    stdu r1, -112(r1)
; AIX64-NEXT:    ld r3, L..C0(r2) # @_MergedGlobals
; AIX64-NEXT:    std r0, 128(r1)
; AIX64-NEXT:    addi r3, r3, 20
; AIX64-NEXT:    bl .callee[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    addi r1, r1, 112
; AIX64-NEXT:    ld r0, 16(r1)
; AIX64-NEXT:    mtlr r0
; AIX64-NEXT:    blr
;
; LINUX64BE-LABEL: str1:
; LINUX64BE:       # %bb.0: # %entry
; LINUX64BE-NEXT:    mflr r0
; LINUX64BE-NEXT:    stdu r1, -112(r1)
; LINUX64BE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64BE-NEXT:    std r0, 128(r1)
; LINUX64BE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64BE-NEXT:    addi r3, r3, 20
; LINUX64BE-NEXT:    bl callee
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    addi r1, r1, 112
; LINUX64BE-NEXT:    ld r0, 16(r1)
; LINUX64BE-NEXT:    mtlr r0
; LINUX64BE-NEXT:    blr
;
; LINUX64LE-LABEL: str1:
; LINUX64LE:       # %bb.0: # %entry
; LINUX64LE-NEXT:    mflr r0
; LINUX64LE-NEXT:    stdu r1, -32(r1)
; LINUX64LE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64LE-NEXT:    std r0, 48(r1)
; LINUX64LE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64LE-NEXT:    addi r3, r3, 20
; LINUX64LE-NEXT:    bl callee
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    addi r1, r1, 32
; LINUX64LE-NEXT:    ld r0, 16(r1)
; LINUX64LE-NEXT:    mtlr r0
; LINUX64LE-NEXT:    blr
entry:
  %call = tail call signext i32 @callee(ptr noundef nonnull @.str.1)
  ret i32 %call
}

declare signext i32 @callee(ptr noundef) local_unnamed_addr

define dso_local signext i32 @str2() local_unnamed_addr #0 {
; AIX32-LABEL: str2:
; AIX32:       # %bb.0: # %entry
; AIX32-NEXT:    mflr r0
; AIX32-NEXT:    stwu r1, -64(r1)
; AIX32-NEXT:    lwz r3, L..C0(r2) # @_MergedGlobals
; AIX32-NEXT:    stw r0, 72(r1)
; AIX32-NEXT:    addi r3, r3, 32
; AIX32-NEXT:    bl .callee[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    addi r1, r1, 64
; AIX32-NEXT:    lwz r0, 8(r1)
; AIX32-NEXT:    mtlr r0
; AIX32-NEXT:    blr
;
; AIX64-LABEL: str2:
; AIX64:       # %bb.0: # %entry
; AIX64-NEXT:    mflr r0
; AIX64-NEXT:    stdu r1, -112(r1)
; AIX64-NEXT:    ld r3, L..C0(r2) # @_MergedGlobals
; AIX64-NEXT:    std r0, 128(r1)
; AIX64-NEXT:    addi r3, r3, 32
; AIX64-NEXT:    bl .callee[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    addi r1, r1, 112
; AIX64-NEXT:    ld r0, 16(r1)
; AIX64-NEXT:    mtlr r0
; AIX64-NEXT:    blr
;
; LINUX64BE-LABEL: str2:
; LINUX64BE:       # %bb.0: # %entry
; LINUX64BE-NEXT:    mflr r0
; LINUX64BE-NEXT:    stdu r1, -112(r1)
; LINUX64BE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64BE-NEXT:    std r0, 128(r1)
; LINUX64BE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64BE-NEXT:    addi r3, r3, 32
; LINUX64BE-NEXT:    bl callee
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    addi r1, r1, 112
; LINUX64BE-NEXT:    ld r0, 16(r1)
; LINUX64BE-NEXT:    mtlr r0
; LINUX64BE-NEXT:    blr
;
; LINUX64LE-LABEL: str2:
; LINUX64LE:       # %bb.0: # %entry
; LINUX64LE-NEXT:    mflr r0
; LINUX64LE-NEXT:    stdu r1, -32(r1)
; LINUX64LE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64LE-NEXT:    std r0, 48(r1)
; LINUX64LE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64LE-NEXT:    addi r3, r3, 32
; LINUX64LE-NEXT:    bl callee
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    addi r1, r1, 32
; LINUX64LE-NEXT:    ld r0, 16(r1)
; LINUX64LE-NEXT:    mtlr r0
; LINUX64LE-NEXT:    blr
entry:
  %call = tail call signext i32 @callee(ptr noundef nonnull @.str.2)
  ret i32 %call
}

define dso_local signext i32 @str3() local_unnamed_addr #0 {
; AIX32-LABEL: str3:
; AIX32:       # %bb.0: # %entry
; AIX32-NEXT:    mflr r0
; AIX32-NEXT:    stwu r1, -64(r1)
; AIX32-NEXT:    stw r0, 72(r1)
; AIX32-NEXT:    stw r30, 56(r1) # 4-byte Folded Spill
; AIX32-NEXT:    lwz r30, L..C0(r2) # @_MergedGlobals
; AIX32-NEXT:    addi r3, r30, 44
; AIX32-NEXT:    stw r31, 60(r1) # 4-byte Folded Spill
; AIX32-NEXT:    bl .callee[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    mr r31, r3
; AIX32-NEXT:    addi r3, r30, 32
; AIX32-NEXT:    bl .callee[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    add r3, r3, r31
; AIX32-NEXT:    lwz r31, 60(r1) # 4-byte Folded Reload
; AIX32-NEXT:    lwz r30, 56(r1) # 4-byte Folded Reload
; AIX32-NEXT:    addi r1, r1, 64
; AIX32-NEXT:    lwz r0, 8(r1)
; AIX32-NEXT:    mtlr r0
; AIX32-NEXT:    blr
;
; AIX64-LABEL: str3:
; AIX64:       # %bb.0: # %entry
; AIX64-NEXT:    mflr r0
; AIX64-NEXT:    stdu r1, -128(r1)
; AIX64-NEXT:    std r0, 144(r1)
; AIX64-NEXT:    std r30, 112(r1) # 8-byte Folded Spill
; AIX64-NEXT:    ld r30, L..C0(r2) # @_MergedGlobals
; AIX64-NEXT:    addi r3, r30, 44
; AIX64-NEXT:    std r31, 120(r1) # 8-byte Folded Spill
; AIX64-NEXT:    bl .callee[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    mr r31, r3
; AIX64-NEXT:    addi r3, r30, 32
; AIX64-NEXT:    bl .callee[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    add r3, r3, r31
; AIX64-NEXT:    ld r31, 120(r1) # 8-byte Folded Reload
; AIX64-NEXT:    ld r30, 112(r1) # 8-byte Folded Reload
; AIX64-NEXT:    extsw r3, r3
; AIX64-NEXT:    addi r1, r1, 128
; AIX64-NEXT:    ld r0, 16(r1)
; AIX64-NEXT:    mtlr r0
; AIX64-NEXT:    blr
;
; LINUX64BE-LABEL: str3:
; LINUX64BE:       # %bb.0: # %entry
; LINUX64BE-NEXT:    mflr r0
; LINUX64BE-NEXT:    stdu r1, -144(r1)
; LINUX64BE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64BE-NEXT:    std r0, 160(r1)
; LINUX64BE-NEXT:    std r29, 120(r1) # 8-byte Folded Spill
; LINUX64BE-NEXT:    addi r29, r3, .L_MergedGlobals@toc@l
; LINUX64BE-NEXT:    std r30, 128(r1) # 8-byte Folded Spill
; LINUX64BE-NEXT:    addi r3, r29, 44
; LINUX64BE-NEXT:    bl callee
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    mr r30, r3
; LINUX64BE-NEXT:    addi r3, r29, 32
; LINUX64BE-NEXT:    bl callee
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    add r3, r3, r30
; LINUX64BE-NEXT:    ld r30, 128(r1) # 8-byte Folded Reload
; LINUX64BE-NEXT:    ld r29, 120(r1) # 8-byte Folded Reload
; LINUX64BE-NEXT:    extsw r3, r3
; LINUX64BE-NEXT:    addi r1, r1, 144
; LINUX64BE-NEXT:    ld r0, 16(r1)
; LINUX64BE-NEXT:    mtlr r0
; LINUX64BE-NEXT:    blr
;
; LINUX64LE-LABEL: str3:
; LINUX64LE:       # %bb.0: # %entry
; LINUX64LE-NEXT:    mflr r0
; LINUX64LE-NEXT:    std r29, -24(r1) # 8-byte Folded Spill
; LINUX64LE-NEXT:    std r30, -16(r1) # 8-byte Folded Spill
; LINUX64LE-NEXT:    stdu r1, -64(r1)
; LINUX64LE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64LE-NEXT:    std r0, 80(r1)
; LINUX64LE-NEXT:    addi r29, r3, .L_MergedGlobals@toc@l
; LINUX64LE-NEXT:    addi r3, r29, 44
; LINUX64LE-NEXT:    bl callee
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    mr r30, r3
; LINUX64LE-NEXT:    addi r3, r29, 32
; LINUX64LE-NEXT:    bl callee
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    add r3, r3, r30
; LINUX64LE-NEXT:    extsw r3, r3
; LINUX64LE-NEXT:    addi r1, r1, 64
; LINUX64LE-NEXT:    ld r0, 16(r1)
; LINUX64LE-NEXT:    ld r30, -16(r1) # 8-byte Folded Reload
; LINUX64LE-NEXT:    ld r29, -24(r1) # 8-byte Folded Reload
; LINUX64LE-NEXT:    mtlr r0
; LINUX64LE-NEXT:    blr
entry:
  %call = tail call signext i32 @callee(ptr noundef nonnull @.str.3)
  %call1 = tail call signext i32 @callee(ptr noundef nonnull @.str.2)
  %add = add nsw i32 %call1, %call
  ret i32 %add
}

define dso_local signext i32 @str4() local_unnamed_addr #0 {
; AIX32-LABEL: str4:
; AIX32:       # %bb.0: # %entry
; AIX32-NEXT:    mflr r0
; AIX32-NEXT:    stwu r1, -64(r1)
; AIX32-NEXT:    lwz r3, L..C0(r2) # @_MergedGlobals
; AIX32-NEXT:    stw r0, 72(r1)
; AIX32-NEXT:    addi r3, r3, 56
; AIX32-NEXT:    bl .callee[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    addi r1, r1, 64
; AIX32-NEXT:    lwz r0, 8(r1)
; AIX32-NEXT:    mtlr r0
; AIX32-NEXT:    blr
;
; AIX64-LABEL: str4:
; AIX64:       # %bb.0: # %entry
; AIX64-NEXT:    mflr r0
; AIX64-NEXT:    stdu r1, -112(r1)
; AIX64-NEXT:    ld r3, L..C0(r2) # @_MergedGlobals
; AIX64-NEXT:    std r0, 128(r1)
; AIX64-NEXT:    addi r3, r3, 56
; AIX64-NEXT:    bl .callee[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    addi r1, r1, 112
; AIX64-NEXT:    ld r0, 16(r1)
; AIX64-NEXT:    mtlr r0
; AIX64-NEXT:    blr
;
; LINUX64BE-LABEL: str4:
; LINUX64BE:       # %bb.0: # %entry
; LINUX64BE-NEXT:    mflr r0
; LINUX64BE-NEXT:    stdu r1, -112(r1)
; LINUX64BE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64BE-NEXT:    std r0, 128(r1)
; LINUX64BE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64BE-NEXT:    addi r3, r3, 56
; LINUX64BE-NEXT:    bl callee
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    addi r1, r1, 112
; LINUX64BE-NEXT:    ld r0, 16(r1)
; LINUX64BE-NEXT:    mtlr r0
; LINUX64BE-NEXT:    blr
;
; LINUX64LE-LABEL: str4:
; LINUX64LE:       # %bb.0: # %entry
; LINUX64LE-NEXT:    mflr r0
; LINUX64LE-NEXT:    stdu r1, -32(r1)
; LINUX64LE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64LE-NEXT:    std r0, 48(r1)
; LINUX64LE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64LE-NEXT:    addi r3, r3, 56
; LINUX64LE-NEXT:    bl callee
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    addi r1, r1, 32
; LINUX64LE-NEXT:    ld r0, 16(r1)
; LINUX64LE-NEXT:    mtlr r0
; LINUX64LE-NEXT:    blr
entry:
  %call = tail call signext i32 @callee(ptr noundef nonnull @.str.4)
  ret i32 %call
}

define dso_local signext i32 @str5() local_unnamed_addr #0 {
; AIX32-LABEL: str5:
; AIX32:       # %bb.0: # %entry
; AIX32-NEXT:    mflr r0
; AIX32-NEXT:    stwu r1, -64(r1)
; AIX32-NEXT:    lwz r3, L..C0(r2) # @_MergedGlobals
; AIX32-NEXT:    stw r0, 72(r1)
; AIX32-NEXT:    addi r3, r3, 736
; AIX32-NEXT:    bl .callee[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    addi r1, r1, 64
; AIX32-NEXT:    lwz r0, 8(r1)
; AIX32-NEXT:    mtlr r0
; AIX32-NEXT:    blr
;
; AIX64-LABEL: str5:
; AIX64:       # %bb.0: # %entry
; AIX64-NEXT:    mflr r0
; AIX64-NEXT:    stdu r1, -112(r1)
; AIX64-NEXT:    ld r3, L..C0(r2) # @_MergedGlobals
; AIX64-NEXT:    std r0, 128(r1)
; AIX64-NEXT:    addi r3, r3, 736
; AIX64-NEXT:    bl .callee[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    addi r1, r1, 112
; AIX64-NEXT:    ld r0, 16(r1)
; AIX64-NEXT:    mtlr r0
; AIX64-NEXT:    blr
;
; LINUX64BE-LABEL: str5:
; LINUX64BE:       # %bb.0: # %entry
; LINUX64BE-NEXT:    mflr r0
; LINUX64BE-NEXT:    stdu r1, -112(r1)
; LINUX64BE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64BE-NEXT:    std r0, 128(r1)
; LINUX64BE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64BE-NEXT:    addi r3, r3, 736
; LINUX64BE-NEXT:    bl callee
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    addi r1, r1, 112
; LINUX64BE-NEXT:    ld r0, 16(r1)
; LINUX64BE-NEXT:    mtlr r0
; LINUX64BE-NEXT:    blr
;
; LINUX64LE-LABEL: str5:
; LINUX64LE:       # %bb.0: # %entry
; LINUX64LE-NEXT:    mflr r0
; LINUX64LE-NEXT:    stdu r1, -32(r1)
; LINUX64LE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64LE-NEXT:    std r0, 48(r1)
; LINUX64LE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64LE-NEXT:    addi r3, r3, 736
; LINUX64LE-NEXT:    bl callee
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    addi r1, r1, 32
; LINUX64LE-NEXT:    ld r0, 16(r1)
; LINUX64LE-NEXT:    mtlr r0
; LINUX64LE-NEXT:    blr
entry:
  %call = tail call signext i32 @callee(ptr noundef nonnull @.str.5)
  ret i32 %call
}

define dso_local signext i32 @array1() local_unnamed_addr #0 {
; AIX32-LABEL: array1:
; AIX32:       # %bb.0: # %entry
; AIX32-NEXT:    mflr r0
; AIX32-NEXT:    stwu r1, -96(r1)
; AIX32-NEXT:    lwz r5, L..C0(r2) # @_MergedGlobals
; AIX32-NEXT:    li r6, 308
; AIX32-NEXT:    li r4, 12
; AIX32-NEXT:    addi r3, r1, 64
; AIX32-NEXT:    stw r0, 104(r1)
; AIX32-NEXT:    rlwimi r4, r3, 0, 30, 27
; AIX32-NEXT:    lxvw4x vs0, r5, r6
; AIX32-NEXT:    stxvw4x vs0, 0, r4
; AIX32-NEXT:    li r4, 296
; AIX32-NEXT:    lxvw4x vs0, r5, r4
; AIX32-NEXT:    stxvw4x vs0, 0, r3
; AIX32-NEXT:    bl .calleeInt[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    addi r1, r1, 96
; AIX32-NEXT:    lwz r0, 8(r1)
; AIX32-NEXT:    mtlr r0
; AIX32-NEXT:    blr
;
; AIX64-LABEL: array1:
; AIX64:       # %bb.0: # %entry
; AIX64-NEXT:    mflr r0
; AIX64-NEXT:    stdu r1, -144(r1)
; AIX64-NEXT:    ld r3, L..C0(r2) # @_MergedGlobals
; AIX64-NEXT:    li r4, 308
; AIX64-NEXT:    std r0, 160(r1)
; AIX64-NEXT:    lxvw4x vs0, r3, r4
; AIX64-NEXT:    addi r4, r1, 124
; AIX64-NEXT:    stxvw4x vs0, 0, r4
; AIX64-NEXT:    li r4, 296
; AIX64-NEXT:    lxvw4x vs0, r3, r4
; AIX64-NEXT:    addi r3, r1, 112
; AIX64-NEXT:    stxvw4x vs0, 0, r3
; AIX64-NEXT:    bl .calleeInt[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    addi r1, r1, 144
; AIX64-NEXT:    ld r0, 16(r1)
; AIX64-NEXT:    mtlr r0
; AIX64-NEXT:    blr
;
; LINUX64BE-LABEL: array1:
; LINUX64BE:       # %bb.0: # %entry
; LINUX64BE-NEXT:    mflr r0
; LINUX64BE-NEXT:    stdu r1, -144(r1)
; LINUX64BE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64BE-NEXT:    li r4, 308
; LINUX64BE-NEXT:    std r0, 160(r1)
; LINUX64BE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64BE-NEXT:    lxvw4x vs0, r3, r4
; LINUX64BE-NEXT:    addi r4, r1, 124
; LINUX64BE-NEXT:    stxvw4x vs0, 0, r4
; LINUX64BE-NEXT:    li r4, 296
; LINUX64BE-NEXT:    lxvw4x vs0, r3, r4
; LINUX64BE-NEXT:    addi r3, r1, 112
; LINUX64BE-NEXT:    stxvw4x vs0, 0, r3
; LINUX64BE-NEXT:    bl calleeInt
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    addi r1, r1, 144
; LINUX64BE-NEXT:    ld r0, 16(r1)
; LINUX64BE-NEXT:    mtlr r0
; LINUX64BE-NEXT:    blr
;
; LINUX64LE-LABEL: array1:
; LINUX64LE:       # %bb.0: # %entry
; LINUX64LE-NEXT:    mflr r0
; LINUX64LE-NEXT:    stdu r1, -64(r1)
; LINUX64LE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64LE-NEXT:    li r4, 308
; LINUX64LE-NEXT:    std r0, 80(r1)
; LINUX64LE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64LE-NEXT:    lxvd2x vs0, r3, r4
; LINUX64LE-NEXT:    addi r4, r1, 44
; LINUX64LE-NEXT:    stxvd2x vs0, 0, r4
; LINUX64LE-NEXT:    li r4, 296
; LINUX64LE-NEXT:    lxvd2x vs0, r3, r4
; LINUX64LE-NEXT:    addi r3, r1, 32
; LINUX64LE-NEXT:    stxvd2x vs0, 0, r3
; LINUX64LE-NEXT:    bl calleeInt
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    addi r1, r1, 64
; LINUX64LE-NEXT:    ld r0, 16(r1)
; LINUX64LE-NEXT:    mtlr r0
; LINUX64LE-NEXT:    blr
entry:
  %IntArray = alloca [7 x i32], align 4
  call void @llvm.lifetime.start.p0(i64 28, ptr nonnull %IntArray)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(28) %IntArray, ptr noundef nonnull align 4 dereferenceable(28) @__const.mixed2.IntArray, i64 28, i1 false)
  %call = call signext i32 @calleeInt(ptr noundef nonnull %IntArray)
  call void @llvm.lifetime.end.p0(i64 28, ptr nonnull %IntArray)
  ret i32 %call
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare signext i32 @calleeInt(ptr noundef) local_unnamed_addr
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

define dso_local double @DValue1() local_unnamed_addr #0 {
; LINUX-LABEL: DValue1:
; LINUX:       # %bb.0: # %entry
; LINUX-NEXT:    addis 3, 2, .LCPI6_0@toc@ha
; LINUX-NEXT:    lfd 1, .LCPI6_0@toc@l(3)
; LINUX-NEXT:    blr
; AIX32-LABEL: DValue1:
; AIX32:       # %bb.0: # %entry
; AIX32-NEXT:    lwz r3, L..C1(r2) # %const.0
; AIX32-NEXT:    lfd f1, 0(r3)
; AIX32-NEXT:    blr
;
; AIX64-LABEL: DValue1:
; AIX64:       # %bb.0: # %entry
; AIX64-NEXT:    ld r3, L..C1(r2) # %const.0
; AIX64-NEXT:    lfd f1, 0(r3)
; AIX64-NEXT:    blr
;
; LINUX64BE-LABEL: DValue1:
; LINUX64BE:       # %bb.0: # %entry
; LINUX64BE-NEXT:    addis r3, r2, .LCPI6_0@toc@ha
; LINUX64BE-NEXT:    lfd f1, .LCPI6_0@toc@l(r3)
; LINUX64BE-NEXT:    blr
;
; LINUX64LE-LABEL: DValue1:
; LINUX64LE:       # %bb.0: # %entry
; LINUX64LE-NEXT:    addis r3, r2, .LCPI6_0@toc@ha
; LINUX64LE-NEXT:    lfd f1, .LCPI6_0@toc@l(r3)
; LINUX64LE-NEXT:    blr
entry:
  ret double 3.141590e+00
}

define dso_local signext i32 @str6() local_unnamed_addr #0 {
; AIX32-LABEL: str6:
; AIX32:       # %bb.0: # %entry
; AIX32-NEXT:    mflr r0
; AIX32-NEXT:    stwu r1, -80(r1)
; AIX32-NEXT:    li r3, 17152
; AIX32-NEXT:    stw r0, 88(r1)
; AIX32-NEXT:    stw r31, 76(r1) # 4-byte Folded Spill
; AIX32-NEXT:    sth r3, 72(r1)
; AIX32-NEXT:    lis r3, 16963
; AIX32-NEXT:    ori r3, r3, 16706
; AIX32-NEXT:    stw r3, 68(r1)
; AIX32-NEXT:    lis r3, 16706
; AIX32-NEXT:    ori r3, r3, 17217
; AIX32-NEXT:    stw r3, 64(r1)
; AIX32-NEXT:    addi r3, r1, 64
; AIX32-NEXT:    bl .callee[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    mr r31, r3
; AIX32-NEXT:    addi r3, r1, 69
; AIX32-NEXT:    bl .callee[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    add r3, r3, r31
; AIX32-NEXT:    lwz r31, 76(r1) # 4-byte Folded Reload
; AIX32-NEXT:    addi r1, r1, 80
; AIX32-NEXT:    lwz r0, 8(r1)
; AIX32-NEXT:    mtlr r0
; AIX32-NEXT:    blr
;
; AIX64-LABEL: str6:
; AIX64:       # %bb.0: # %entry
; AIX64-NEXT:    mflr r0
; AIX64-NEXT:    stdu r1, -144(r1)
; AIX64-NEXT:    li r3, 17152
; AIX64-NEXT:    std r0, 160(r1)
; AIX64-NEXT:    std r31, 136(r1) # 8-byte Folded Spill
; AIX64-NEXT:    sth r3, 128(r1)
; AIX64-NEXT:    lis r3, 16706
; AIX64-NEXT:    ori r3, r3, 17217
; AIX64-NEXT:    rldic r3, r3, 32, 1
; AIX64-NEXT:    oris r3, r3, 16963
; AIX64-NEXT:    ori r3, r3, 16706
; AIX64-NEXT:    std r3, 120(r1)
; AIX64-NEXT:    addi r3, r1, 120
; AIX64-NEXT:    bl .callee[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    mr r31, r3
; AIX64-NEXT:    addi r3, r1, 125
; AIX64-NEXT:    bl .callee[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    add r3, r3, r31
; AIX64-NEXT:    ld r31, 136(r1) # 8-byte Folded Reload
; AIX64-NEXT:    extsw r3, r3
; AIX64-NEXT:    addi r1, r1, 144
; AIX64-NEXT:    ld r0, 16(r1)
; AIX64-NEXT:    mtlr r0
; AIX64-NEXT:    blr
;
; LINUX64BE-LABEL: str6:
; LINUX64BE:       # %bb.0: # %entry
; LINUX64BE-NEXT:    mflr r0
; LINUX64BE-NEXT:    stdu r1, -144(r1)
; LINUX64BE-NEXT:    li r3, 17152
; LINUX64BE-NEXT:    std r0, 160(r1)
; LINUX64BE-NEXT:    std r30, 128(r1) # 8-byte Folded Spill
; LINUX64BE-NEXT:    sth r3, 120(r1)
; LINUX64BE-NEXT:    lis r3, 16706
; LINUX64BE-NEXT:    ori r3, r3, 17217
; LINUX64BE-NEXT:    rldic r3, r3, 32, 1
; LINUX64BE-NEXT:    oris r3, r3, 16963
; LINUX64BE-NEXT:    ori r3, r3, 16706
; LINUX64BE-NEXT:    std r3, 112(r1)
; LINUX64BE-NEXT:    addi r3, r1, 112
; LINUX64BE-NEXT:    bl callee
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    mr r30, r3
; LINUX64BE-NEXT:    addi r3, r1, 117
; LINUX64BE-NEXT:    bl callee
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    add r3, r3, r30
; LINUX64BE-NEXT:    ld r30, 128(r1) # 8-byte Folded Reload
; LINUX64BE-NEXT:    extsw r3, r3
; LINUX64BE-NEXT:    addi r1, r1, 144
; LINUX64BE-NEXT:    ld r0, 16(r1)
; LINUX64BE-NEXT:    mtlr r0
; LINUX64BE-NEXT:    blr
;
; LINUX64LE-LABEL: str6:
; LINUX64LE:       # %bb.0: # %entry
; LINUX64LE-NEXT:    mflr r0
; LINUX64LE-NEXT:    std r30, -16(r1) # 8-byte Folded Spill
; LINUX64LE-NEXT:    stdu r1, -64(r1)
; LINUX64LE-NEXT:    li r3, 67
; LINUX64LE-NEXT:    std r0, 80(r1)
; LINUX64LE-NEXT:    sth r3, 40(r1)
; LINUX64LE-NEXT:    lis r3, 8480
; LINUX64LE-NEXT:    ori r3, r3, 41377
; LINUX64LE-NEXT:    rldic r3, r3, 33, 1
; LINUX64LE-NEXT:    oris r3, r3, 16707
; LINUX64LE-NEXT:    ori r3, r3, 16961
; LINUX64LE-NEXT:    std r3, 32(r1)
; LINUX64LE-NEXT:    addi r3, r1, 32
; LINUX64LE-NEXT:    bl callee
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    mr r30, r3
; LINUX64LE-NEXT:    addi r3, r1, 37
; LINUX64LE-NEXT:    bl callee
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    add r3, r3, r30
; LINUX64LE-NEXT:    extsw r3, r3
; LINUX64LE-NEXT:    addi r1, r1, 64
; LINUX64LE-NEXT:    ld r0, 16(r1)
; LINUX64LE-NEXT:    ld r30, -16(r1) # 8-byte Folded Reload
; LINUX64LE-NEXT:    mtlr r0
; LINUX64LE-NEXT:    blr
entry:
  %TheString = alloca [10 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 10, ptr nonnull %TheString)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(10) %TheString, ptr noundef nonnull align 1 dereferenceable(10) @__const.str6.TheString, i64 10, i1 false)
  %call = call signext i32 @callee(ptr noundef nonnull %TheString)
  %add.ptr = getelementptr inbounds i8, ptr %TheString, i64 5
  %call2 = call signext i32 @callee(ptr noundef nonnull %add.ptr)
  %add = add nsw i32 %call2, %call
  call void @llvm.lifetime.end.p0(i64 10, ptr nonnull %TheString)
  ret i32 %add
}

define dso_local signext i32 @str7() local_unnamed_addr #0 {
; AIX32-LABEL: str7:
; AIX32:       # %bb.0: # %entry
; AIX32-NEXT:    mflr r0
; AIX32-NEXT:    stwu r1, -64(r1)
; AIX32-NEXT:    lwz r3, L..C2(r2) # @GLOBALSTRING
; AIX32-NEXT:    stw r0, 72(r1)
; AIX32-NEXT:    stw r31, 60(r1) # 4-byte Folded Spill
; AIX32-NEXT:    lwz r3, 0(r3)
; AIX32-NEXT:    bl .callee[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    mr r31, r3
; AIX32-NEXT:    lwz r3, L..C0(r2) # @_MergedGlobals
; AIX32-NEXT:    addi r3, r3, 80
; AIX32-NEXT:    bl .callee[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    add r3, r3, r31
; AIX32-NEXT:    lwz r31, 60(r1) # 4-byte Folded Reload
; AIX32-NEXT:    addi r1, r1, 64
; AIX32-NEXT:    lwz r0, 8(r1)
; AIX32-NEXT:    mtlr r0
; AIX32-NEXT:    blr
;
; AIX64-LABEL: str7:
; AIX64:       # %bb.0: # %entry
; AIX64-NEXT:    mflr r0
; AIX64-NEXT:    stdu r1, -128(r1)
; AIX64-NEXT:    ld r3, L..C2(r2) # @GLOBALSTRING
; AIX64-NEXT:    std r0, 144(r1)
; AIX64-NEXT:    std r31, 120(r1) # 8-byte Folded Spill
; AIX64-NEXT:    ld r3, 0(r3)
; AIX64-NEXT:    bl .callee[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    mr r31, r3
; AIX64-NEXT:    ld r3, L..C0(r2) # @_MergedGlobals
; AIX64-NEXT:    addi r3, r3, 80
; AIX64-NEXT:    bl .callee[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    add r3, r3, r31
; AIX64-NEXT:    ld r31, 120(r1) # 8-byte Folded Reload
; AIX64-NEXT:    extsw r3, r3
; AIX64-NEXT:    addi r1, r1, 128
; AIX64-NEXT:    ld r0, 16(r1)
; AIX64-NEXT:    mtlr r0
; AIX64-NEXT:    blr
;
; LINUX64BE-LABEL: str7:
; LINUX64BE:       # %bb.0: # %entry
; LINUX64BE-NEXT:    mflr r0
; LINUX64BE-NEXT:    stdu r1, -128(r1)
; LINUX64BE-NEXT:    std r0, 144(r1)
; LINUX64BE-NEXT:    addis r3, r2, GLOBALSTRING@toc@ha
; LINUX64BE-NEXT:    ld r3, GLOBALSTRING@toc@l(r3)
; LINUX64BE-NEXT:    std r30, 112(r1) # 8-byte Folded Spill
; LINUX64BE-NEXT:    bl callee
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    mr r30, r3
; LINUX64BE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64BE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64BE-NEXT:    addi r3, r3, 80
; LINUX64BE-NEXT:    bl callee
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    add r3, r3, r30
; LINUX64BE-NEXT:    ld r30, 112(r1) # 8-byte Folded Reload
; LINUX64BE-NEXT:    extsw r3, r3
; LINUX64BE-NEXT:    addi r1, r1, 128
; LINUX64BE-NEXT:    ld r0, 16(r1)
; LINUX64BE-NEXT:    mtlr r0
; LINUX64BE-NEXT:    blr
;
; LINUX64LE-LABEL: str7:
; LINUX64LE:       # %bb.0: # %entry
; LINUX64LE-NEXT:    mflr r0
; LINUX64LE-NEXT:    std r30, -16(r1) # 8-byte Folded Spill
; LINUX64LE-NEXT:    stdu r1, -48(r1)
; LINUX64LE-NEXT:    std r0, 64(r1)
; LINUX64LE-NEXT:    addis r3, r2, GLOBALSTRING@toc@ha
; LINUX64LE-NEXT:    ld r3, GLOBALSTRING@toc@l(r3)
; LINUX64LE-NEXT:    bl callee
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    mr r30, r3
; LINUX64LE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64LE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64LE-NEXT:    addi r3, r3, 80
; LINUX64LE-NEXT:    bl callee
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    add r3, r3, r30
; LINUX64LE-NEXT:    extsw r3, r3
; LINUX64LE-NEXT:    addi r1, r1, 48
; LINUX64LE-NEXT:    ld r0, 16(r1)
; LINUX64LE-NEXT:    ld r30, -16(r1) # 8-byte Folded Reload
; LINUX64LE-NEXT:    mtlr r0
; LINUX64LE-NEXT:    blr
entry:
  %0 = load ptr, ptr @GLOBALSTRING, align 8
  %call = tail call signext i32 @callee(ptr noundef %0)
  %call1 = tail call signext i32 @callee(ptr noundef nonnull @.str.8)
  %add = add nsw i32 %call1, %call
  ret i32 %add
}

define dso_local signext i32 @mixed1() local_unnamed_addr #0 {
; AIX32-LABEL: mixed1:
; AIX32:       # %bb.0: # %entry
; AIX32-NEXT:    mflr r0
; AIX32-NEXT:    stwu r1, -64(r1)
; AIX32-NEXT:    lwz r3, L..C3(r2) # @IntArray2
; AIX32-NEXT:    stw r0, 72(r1)
; AIX32-NEXT:    stw r31, 60(r1) # 4-byte Folded Spill
; AIX32-NEXT:    bl .calleeInt[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    mr r31, r3
; AIX32-NEXT:    lwz r3, L..C0(r2) # @_MergedGlobals
; AIX32-NEXT:    addi r3, r3, 68
; AIX32-NEXT:    bl .callee[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    add r3, r3, r31
; AIX32-NEXT:    lwz r31, 60(r1) # 4-byte Folded Reload
; AIX32-NEXT:    addi r1, r1, 64
; AIX32-NEXT:    lwz r0, 8(r1)
; AIX32-NEXT:    mtlr r0
; AIX32-NEXT:    blr
;
; AIX64-LABEL: mixed1:
; AIX64:       # %bb.0: # %entry
; AIX64-NEXT:    mflr r0
; AIX64-NEXT:    stdu r1, -128(r1)
; AIX64-NEXT:    ld r3, L..C3(r2) # @IntArray2
; AIX64-NEXT:    std r0, 144(r1)
; AIX64-NEXT:    std r31, 120(r1) # 8-byte Folded Spill
; AIX64-NEXT:    bl .calleeInt[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    mr r31, r3
; AIX64-NEXT:    ld r3, L..C0(r2) # @_MergedGlobals
; AIX64-NEXT:    addi r3, r3, 68
; AIX64-NEXT:    bl .callee[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    add r3, r3, r31
; AIX64-NEXT:    ld r31, 120(r1) # 8-byte Folded Reload
; AIX64-NEXT:    extsw r3, r3
; AIX64-NEXT:    addi r1, r1, 128
; AIX64-NEXT:    ld r0, 16(r1)
; AIX64-NEXT:    mtlr r0
; AIX64-NEXT:    blr
;
; LINUX64BE-LABEL: mixed1:
; LINUX64BE:       # %bb.0: # %entry
; LINUX64BE-NEXT:    mflr r0
; LINUX64BE-NEXT:    stdu r1, -128(r1)
; LINUX64BE-NEXT:    addis r3, r2, IntArray2@toc@ha
; LINUX64BE-NEXT:    std r0, 144(r1)
; LINUX64BE-NEXT:    std r30, 112(r1) # 8-byte Folded Spill
; LINUX64BE-NEXT:    addi r3, r3, IntArray2@toc@l
; LINUX64BE-NEXT:    bl calleeInt
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    mr r30, r3
; LINUX64BE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64BE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64BE-NEXT:    addi r3, r3, 68
; LINUX64BE-NEXT:    bl callee
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    add r3, r3, r30
; LINUX64BE-NEXT:    ld r30, 112(r1) # 8-byte Folded Reload
; LINUX64BE-NEXT:    extsw r3, r3
; LINUX64BE-NEXT:    addi r1, r1, 128
; LINUX64BE-NEXT:    ld r0, 16(r1)
; LINUX64BE-NEXT:    mtlr r0
; LINUX64BE-NEXT:    blr
;
; LINUX64LE-LABEL: mixed1:
; LINUX64LE:       # %bb.0: # %entry
; LINUX64LE-NEXT:    mflr r0
; LINUX64LE-NEXT:    std r30, -16(r1) # 8-byte Folded Spill
; LINUX64LE-NEXT:    stdu r1, -48(r1)
; LINUX64LE-NEXT:    addis r3, r2, IntArray2@toc@ha
; LINUX64LE-NEXT:    std r0, 64(r1)
; LINUX64LE-NEXT:    addi r3, r3, IntArray2@toc@l
; LINUX64LE-NEXT:    bl calleeInt
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    mr r30, r3
; LINUX64LE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64LE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64LE-NEXT:    addi r3, r3, 68
; LINUX64LE-NEXT:    bl callee
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    add r3, r3, r30
; LINUX64LE-NEXT:    extsw r3, r3
; LINUX64LE-NEXT:    addi r1, r1, 48
; LINUX64LE-NEXT:    ld r0, 16(r1)
; LINUX64LE-NEXT:    ld r30, -16(r1) # 8-byte Folded Reload
; LINUX64LE-NEXT:    mtlr r0
; LINUX64LE-NEXT:    blr
entry:
  %call = tail call signext i32 @calleeInt(ptr noundef nonnull @IntArray2)
  %call1 = tail call signext i32 @callee(ptr noundef nonnull @.str.6)
  %add = add nsw i32 %call1, %call
  ret i32 %add
}

define dso_local signext i32 @mixed2() local_unnamed_addr #0 {
; AIX32-LABEL: mixed2:
; AIX32:       # %bb.0: # %entry
; AIX32-NEXT:    mflr r0
; AIX32-NEXT:    stwu r1, -112(r1)
; AIX32-NEXT:    stw r0, 120(r1)
; AIX32-NEXT:    stw r30, 104(r1) # 4-byte Folded Spill
; AIX32-NEXT:    lwz r30, L..C0(r2) # @_MergedGlobals
; AIX32-NEXT:    li r5, 308
; AIX32-NEXT:    li r4, 12
; AIX32-NEXT:    addi r3, r1, 64
; AIX32-NEXT:    stw r31, 108(r1) # 4-byte Folded Spill
; AIX32-NEXT:    rlwimi r4, r3, 0, 30, 27
; AIX32-NEXT:    lxvw4x vs0, r30, r5
; AIX32-NEXT:    stxvw4x vs0, 0, r4
; AIX32-NEXT:    li r4, 296
; AIX32-NEXT:    lxvw4x vs0, r30, r4
; AIX32-NEXT:    stxvw4x vs0, 0, r3
; AIX32-NEXT:    bl .calleeInt[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    mr r31, r3
; AIX32-NEXT:    lwz r3, L..C3(r2) # @IntArray2
; AIX32-NEXT:    bl .calleeInt[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    add r31, r3, r31
; AIX32-NEXT:    addi r3, r30, 68
; AIX32-NEXT:    bl .callee[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    add r31, r31, r3
; AIX32-NEXT:    addi r3, r30, 273
; AIX32-NEXT:    bl .callee[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    add r3, r31, r3
; AIX32-NEXT:    lwz r31, 108(r1) # 4-byte Folded Reload
; AIX32-NEXT:    lwz r30, 104(r1) # 4-byte Folded Reload
; AIX32-NEXT:    addi r1, r1, 112
; AIX32-NEXT:    lwz r0, 8(r1)
; AIX32-NEXT:    mtlr r0
; AIX32-NEXT:    blr
;
; AIX64-LABEL: mixed2:
; AIX64:       # %bb.0: # %entry
; AIX64-NEXT:    mflr r0
; AIX64-NEXT:    stdu r1, -160(r1)
; AIX64-NEXT:    std r0, 176(r1)
; AIX64-NEXT:    std r30, 144(r1) # 8-byte Folded Spill
; AIX64-NEXT:    ld r30, L..C0(r2) # @_MergedGlobals
; AIX64-NEXT:    li r3, 308
; AIX64-NEXT:    std r31, 152(r1) # 8-byte Folded Spill
; AIX64-NEXT:    lxvw4x vs0, r30, r3
; AIX64-NEXT:    addi r3, r1, 124
; AIX64-NEXT:    stxvw4x vs0, 0, r3
; AIX64-NEXT:    li r3, 296
; AIX64-NEXT:    lxvw4x vs0, r30, r3
; AIX64-NEXT:    addi r3, r1, 112
; AIX64-NEXT:    stxvw4x vs0, 0, r3
; AIX64-NEXT:    bl .calleeInt[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    mr r31, r3
; AIX64-NEXT:    ld r3, L..C3(r2) # @IntArray2
; AIX64-NEXT:    bl .calleeInt[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    add r31, r3, r31
; AIX64-NEXT:    addi r3, r30, 68
; AIX64-NEXT:    bl .callee[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    add r31, r31, r3
; AIX64-NEXT:    addi r3, r30, 273
; AIX64-NEXT:    bl .callee[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    add r3, r31, r3
; AIX64-NEXT:    ld r31, 152(r1) # 8-byte Folded Reload
; AIX64-NEXT:    ld r30, 144(r1) # 8-byte Folded Reload
; AIX64-NEXT:    extsw r3, r3
; AIX64-NEXT:    addi r1, r1, 160
; AIX64-NEXT:    ld r0, 16(r1)
; AIX64-NEXT:    mtlr r0
; AIX64-NEXT:    blr
;
; LINUX64BE-LABEL: mixed2:
; LINUX64BE:       # %bb.0: # %entry
; LINUX64BE-NEXT:    mflr r0
; LINUX64BE-NEXT:    stdu r1, -176(r1)
; LINUX64BE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64BE-NEXT:    std r0, 192(r1)
; LINUX64BE-NEXT:    std r29, 152(r1) # 8-byte Folded Spill
; LINUX64BE-NEXT:    li r4, 308
; LINUX64BE-NEXT:    std r30, 160(r1) # 8-byte Folded Spill
; LINUX64BE-NEXT:    addi r29, r3, .L_MergedGlobals@toc@l
; LINUX64BE-NEXT:    addi r3, r1, 124
; LINUX64BE-NEXT:    lxvw4x vs0, r29, r4
; LINUX64BE-NEXT:    stxvw4x vs0, 0, r3
; LINUX64BE-NEXT:    li r3, 296
; LINUX64BE-NEXT:    lxvw4x vs0, r29, r3
; LINUX64BE-NEXT:    addi r3, r1, 112
; LINUX64BE-NEXT:    stxvw4x vs0, 0, r3
; LINUX64BE-NEXT:    bl calleeInt
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    mr r30, r3
; LINUX64BE-NEXT:    addis r3, r2, IntArray2@toc@ha
; LINUX64BE-NEXT:    addi r3, r3, IntArray2@toc@l
; LINUX64BE-NEXT:    bl calleeInt
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    add r30, r3, r30
; LINUX64BE-NEXT:    addi r3, r29, 68
; LINUX64BE-NEXT:    bl callee
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    add r30, r30, r3
; LINUX64BE-NEXT:    addi r3, r29, 273
; LINUX64BE-NEXT:    bl callee
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    add r3, r30, r3
; LINUX64BE-NEXT:    ld r30, 160(r1) # 8-byte Folded Reload
; LINUX64BE-NEXT:    ld r29, 152(r1) # 8-byte Folded Reload
; LINUX64BE-NEXT:    extsw r3, r3
; LINUX64BE-NEXT:    addi r1, r1, 176
; LINUX64BE-NEXT:    ld r0, 16(r1)
; LINUX64BE-NEXT:    mtlr r0
; LINUX64BE-NEXT:    blr
;
; LINUX64LE-LABEL: mixed2:
; LINUX64LE:       # %bb.0: # %entry
; LINUX64LE-NEXT:    mflr r0
; LINUX64LE-NEXT:    std r29, -24(r1) # 8-byte Folded Spill
; LINUX64LE-NEXT:    std r30, -16(r1) # 8-byte Folded Spill
; LINUX64LE-NEXT:    stdu r1, -96(r1)
; LINUX64LE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64LE-NEXT:    li r4, 308
; LINUX64LE-NEXT:    std r0, 112(r1)
; LINUX64LE-NEXT:    addi r29, r3, .L_MergedGlobals@toc@l
; LINUX64LE-NEXT:    addi r3, r1, 44
; LINUX64LE-NEXT:    lxvd2x vs0, r29, r4
; LINUX64LE-NEXT:    stxvd2x vs0, 0, r3
; LINUX64LE-NEXT:    li r3, 296
; LINUX64LE-NEXT:    lxvd2x vs0, r29, r3
; LINUX64LE-NEXT:    addi r3, r1, 32
; LINUX64LE-NEXT:    stxvd2x vs0, 0, r3
; LINUX64LE-NEXT:    bl calleeInt
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    mr r30, r3
; LINUX64LE-NEXT:    addis r3, r2, IntArray2@toc@ha
; LINUX64LE-NEXT:    addi r3, r3, IntArray2@toc@l
; LINUX64LE-NEXT:    bl calleeInt
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    add r30, r3, r30
; LINUX64LE-NEXT:    addi r3, r29, 68
; LINUX64LE-NEXT:    bl callee
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    add r30, r30, r3
; LINUX64LE-NEXT:    addi r3, r29, 273
; LINUX64LE-NEXT:    bl callee
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    add r3, r30, r3
; LINUX64LE-NEXT:    extsw r3, r3
; LINUX64LE-NEXT:    addi r1, r1, 96
; LINUX64LE-NEXT:    ld r0, 16(r1)
; LINUX64LE-NEXT:    ld r30, -16(r1) # 8-byte Folded Reload
; LINUX64LE-NEXT:    ld r29, -24(r1) # 8-byte Folded Reload
; LINUX64LE-NEXT:    mtlr r0
; LINUX64LE-NEXT:    blr
entry:
  %IntArray = alloca [7 x i32], align 4
  call void @llvm.lifetime.start.p0(i64 28, ptr nonnull %IntArray)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(28) %IntArray, ptr noundef nonnull align 4 dereferenceable(28) @__const.mixed2.IntArray, i64 28, i1 false)
  %call = call signext i32 @calleeInt(ptr noundef nonnull %IntArray)
  %call1 = call signext i32 @calleeInt(ptr noundef nonnull @IntArray2)
  %add = add nsw i32 %call1, %call
  %call2 = call signext i32 @callee(ptr noundef nonnull @.str.6)
  %add3 = add nsw i32 %add, %call2
  %call4 = call signext i32 @callee(ptr noundef nonnull @.str.7)
  %add5 = add nsw i32 %add3, %call4
  call void @llvm.lifetime.end.p0(i64 28, ptr nonnull %IntArray)
  ret i32 %add5
}

define dso_local signext i32 @str9() local_unnamed_addr #0 {
; AIX32-LABEL: str9:
; AIX32:       # %bb.0: # %entry
; AIX32-NEXT:    mflr r0
; AIX32-NEXT:    stwu r1, -64(r1)
; AIX32-NEXT:    lwz r3, L..C0(r2) # @_MergedGlobals
; AIX32-NEXT:    stw r0, 72(r1)
; AIX32-NEXT:    addi r3, r3, 128
; AIX32-NEXT:    bl .callee[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    addi r1, r1, 64
; AIX32-NEXT:    lwz r0, 8(r1)
; AIX32-NEXT:    mtlr r0
; AIX32-NEXT:    blr
;
; AIX64-LABEL: str9:
; AIX64:       # %bb.0: # %entry
; AIX64-NEXT:    mflr r0
; AIX64-NEXT:    stdu r1, -112(r1)
; AIX64-NEXT:    ld r3, L..C0(r2) # @_MergedGlobals
; AIX64-NEXT:    std r0, 128(r1)
; AIX64-NEXT:    addi r3, r3, 128
; AIX64-NEXT:    bl .callee[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    addi r1, r1, 112
; AIX64-NEXT:    ld r0, 16(r1)
; AIX64-NEXT:    mtlr r0
; AIX64-NEXT:    blr
;
; LINUX64BE-LABEL: str9:
; LINUX64BE:       # %bb.0: # %entry
; LINUX64BE-NEXT:    mflr r0
; LINUX64BE-NEXT:    stdu r1, -112(r1)
; LINUX64BE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64BE-NEXT:    std r0, 128(r1)
; LINUX64BE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64BE-NEXT:    addi r3, r3, 128
; LINUX64BE-NEXT:    bl callee
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    addi r1, r1, 112
; LINUX64BE-NEXT:    ld r0, 16(r1)
; LINUX64BE-NEXT:    mtlr r0
; LINUX64BE-NEXT:    blr
;
; LINUX64LE-LABEL: str9:
; LINUX64LE:       # %bb.0: # %entry
; LINUX64LE-NEXT:    mflr r0
; LINUX64LE-NEXT:    stdu r1, -32(r1)
; LINUX64LE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64LE-NEXT:    std r0, 48(r1)
; LINUX64LE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64LE-NEXT:    addi r3, r3, 128
; LINUX64LE-NEXT:    bl callee
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    addi r1, r1, 32
; LINUX64LE-NEXT:    ld r0, 16(r1)
; LINUX64LE-NEXT:    mtlr r0
; LINUX64LE-NEXT:    blr
entry:
  %call = tail call signext i32 @callee(ptr noundef nonnull @.str.9)
  ret i32 %call
}

define dso_local signext i32 @str10() local_unnamed_addr #0 {
; AIX32-LABEL: str10:
; AIX32:       # %bb.0: # %entry
; AIX32-NEXT:    mflr r0
; AIX32-NEXT:    stwu r1, -64(r1)
; AIX32-NEXT:    lwz r3, L..C0(r2) # @_MergedGlobals
; AIX32-NEXT:    stw r0, 72(r1)
; AIX32-NEXT:    addi r3, r3, 256
; AIX32-NEXT:    bl .callee[PR]
; AIX32-NEXT:    nop
; AIX32-NEXT:    addi r1, r1, 64
; AIX32-NEXT:    lwz r0, 8(r1)
; AIX32-NEXT:    mtlr r0
; AIX32-NEXT:    blr
;
; AIX64-LABEL: str10:
; AIX64:       # %bb.0: # %entry
; AIX64-NEXT:    mflr r0
; AIX64-NEXT:    stdu r1, -112(r1)
; AIX64-NEXT:    ld r3, L..C0(r2) # @_MergedGlobals
; AIX64-NEXT:    std r0, 128(r1)
; AIX64-NEXT:    addi r3, r3, 256
; AIX64-NEXT:    bl .callee[PR]
; AIX64-NEXT:    nop
; AIX64-NEXT:    addi r1, r1, 112
; AIX64-NEXT:    ld r0, 16(r1)
; AIX64-NEXT:    mtlr r0
; AIX64-NEXT:    blr
;
; LINUX64BE-LABEL: str10:
; LINUX64BE:       # %bb.0: # %entry
; LINUX64BE-NEXT:    mflr r0
; LINUX64BE-NEXT:    stdu r1, -112(r1)
; LINUX64BE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64BE-NEXT:    std r0, 128(r1)
; LINUX64BE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64BE-NEXT:    addi r3, r3, 256
; LINUX64BE-NEXT:    bl callee
; LINUX64BE-NEXT:    nop
; LINUX64BE-NEXT:    addi r1, r1, 112
; LINUX64BE-NEXT:    ld r0, 16(r1)
; LINUX64BE-NEXT:    mtlr r0
; LINUX64BE-NEXT:    blr
;
; LINUX64LE-LABEL: str10:
; LINUX64LE:       # %bb.0: # %entry
; LINUX64LE-NEXT:    mflr r0
; LINUX64LE-NEXT:    stdu r1, -32(r1)
; LINUX64LE-NEXT:    addis r3, r2, .L_MergedGlobals@toc@ha
; LINUX64LE-NEXT:    std r0, 48(r1)
; LINUX64LE-NEXT:    addi r3, r3, .L_MergedGlobals@toc@l
; LINUX64LE-NEXT:    addi r3, r3, 256
; LINUX64LE-NEXT:    bl callee
; LINUX64LE-NEXT:    nop
; LINUX64LE-NEXT:    addi r1, r1, 32
; LINUX64LE-NEXT:    ld r0, 16(r1)
; LINUX64LE-NEXT:    mtlr r0
; LINUX64LE-NEXT:    blr
entry:
  %call = tail call signext i32 @callee(ptr noundef nonnull @.str.10)
  ret i32 %call
}

attributes #0 = { nounwind }

; AIXDATA: .csect L.._MergedGlobals[RO],7
; AIXDATA:       .align  7                               # @_MergedGlobals
; AIXDATA:       .string "ABCABCABC"
; AIXDATA:       .string "str1_STRING"
; AIXDATA:       .string "str2_STRING"
; AIXDATA:       .string "str3_STRING"
; AIXDATA:       .string "str4_STRING"
; AIXDATA:       .string "MixedString"
; AIXDATA:       .byte   'S,'t,'a,'t,'i,'c,' ,'G,'l,'o,'b,'a,'l,0012,0000
; AIXDATA:       .string "str9_STRING....."
; AIXDATA:       .string "str10_STRING...."
; AIXDATA:       .string "Different String 01"
; AIXDATA:       .vbyte  4, 5                            # 0x5
; AIXDATA:       .vbyte  4, 7                            # 0x7
; AIXDATA:       .vbyte  4, 9                            # 0x9
; AIXDATA:       .vbyte  4, 11                           # 0xb
; AIXDATA:       .vbyte  4, 17                           # 0x11
; AIXDATA:       .vbyte  4, 1235                         # 0x4d3
; AIXDATA:       .vbyte  4, 32                           # 0x20
; AIXDATA:       .string "longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_STRING"

; LINUXDATA: .L_MergedGlobals:
; LINUXDATA:    .asciz  "ABCABCABC"
; LINUXDATA:    .asciz  "str1_STRING"
; LINUXDATA:    .asciz  "str2_STRING"
; LINUXDATA:    .asciz  "str3_STRING"
; LINUXDATA:    .asciz  "str4_STRING"
; LINUXDATA:    .asciz  "MixedString"
; LINUXDATA:    .asciz  "Static Global\n"
; LINUXDATA:    .asciz  "str9_STRING....."
; LINUXDATA:    .asciz  "str10_STRING...."
; LINUXDATA:    .asciz  "Different String 01"
; LINUXDATA:    .long   5                               # 0x5
; LINUXDATA:    .long   7                               # 0x7
; LINUXDATA:    .long   9                               # 0x9
; LINUXDATA:    .long   11                              # 0xb
; LINUXDATA:    .long   17                              # 0x11
; LINUXDATA:    .long   1235                            # 0x4d3
; LINUXDATA:    .long   32                              # 0x20
; LINUXDATA:    .asciz  "This is the global string that is at the top.\n"
; LINUXDATA:    .asciz  "longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_longerstr5_STRING"

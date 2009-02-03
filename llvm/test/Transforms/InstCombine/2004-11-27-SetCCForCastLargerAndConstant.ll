; This test case tests the InstructionCombining optimization that
; reduces things like:
;   %Y = sext i8 %X to i32 
;   %C = icmp ult i32 %Y, 1024
; to
;   %C = i1 true
; It includes test cases for different constant values, signedness of the
; cast operands, and types of setCC operators. In all cases, the cast should
; be eliminated. In many cases the setCC is also eliminated based on the
; constant value and the range of the casted value.
;
; RUN: llvm-as %s -o - | opt -instcombine | llvm-dis | \
; RUN:    notcast .*int
; END.
define i1 @lt_signed_to_large_unsigned(i8 %SB) {
        %Y = sext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp ult i32 %Y, 1024              ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @lt_signed_to_large_signed(i8 %SB) {
        %Y = sext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp slt i32 %Y, 1024              ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @lt_signed_to_large_negative(i8 %SB) {
        %Y = sext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp slt i32 %Y, -1024             ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @lt_signed_to_small_signed(i8 %SB) {
        %Y = sext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp slt i32 %Y, 17                ; <i1> [#uses=1]
        ret i1 %C
}
define i1 @lt_signed_to_small_negative(i8 %SB) {
        %Y = sext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp slt i32 %Y, -17               ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @lt_unsigned_to_large_unsigned(i8 %SB) {
        %Y = zext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp ult i32 %Y, 1024              ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @lt_unsigned_to_large_signed(i8 %SB) {
        %Y = zext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp slt i32 %Y, 1024              ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @lt_unsigned_to_large_negative(i8 %SB) {
        %Y = zext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp slt i32 %Y, -1024             ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @lt_unsigned_to_small_unsigned(i8 %SB) {
        %Y = zext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp ult i32 %Y, 17                ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @lt_unsigned_to_small_negative(i8 %SB) {
        %Y = zext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp slt i32 %Y, -17               ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @gt_signed_to_large_unsigned(i8 %SB) {
        %Y = sext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp ugt i32 %Y, 1024              ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @gt_signed_to_large_signed(i8 %SB) {
        %Y = sext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp sgt i32 %Y, 1024              ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @gt_signed_to_large_negative(i8 %SB) {
        %Y = sext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp sgt i32 %Y, -1024             ; <i1> [#uses=1]
        ret i1 %C
}
define i1 @gt_signed_to_small_signed(i8 %SB) {
        %Y = sext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp sgt i32 %Y, 17                ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @gt_signed_to_small_negative(i8 %SB) {
        %Y = sext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp sgt i32 %Y, -17               ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @gt_unsigned_to_large_unsigned(i8 %SB) {
        %Y = zext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp ugt i32 %Y, 1024              ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @gt_unsigned_to_large_signed(i8 %SB) {
        %Y = zext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp sgt i32 %Y, 1024              ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @gt_unsigned_to_large_negative(i8 %SB) {
        %Y = zext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp sgt i32 %Y, -1024             ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @gt_unsigned_to_small_unsigned(i8 %SB) {
        %Y = zext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp ugt i32 %Y, 17                ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @gt_unsigned_to_small_negative(i8 %SB) {
        %Y = zext i8 %SB to i32         ; <i32> [#uses=1]
        %C = icmp sgt i32 %Y, -17               ; <i1> [#uses=1]
        ret i1 %C
}


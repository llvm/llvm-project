; RUN: opt < %s -instcombine -S -data-layout=e-n32 | FileCheck %s --check-prefix=ALL --check-prefix=LE
; RUN: opt < %s -instcombine -S -data-layout=E-n32 | FileCheck %s --check-prefix=ALL --check-prefix=BE

declare i32 @memcmp(i8*, i8*, i64)

; TODO: The alignment of this constant does not matter. We constant fold the load.

@charbuf = private unnamed_addr constant [4 x i8] [i8 0, i8 0, i8 0, i8 1], align 1

define i1 @memcmp_4bytes_unaligned_constant_i8(i8* align 4 %x) {
; ALL-LABEL: @memcmp_4bytes_unaligned_constant_i8(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* %x, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @charbuf, i64 0, i64 0), i64 4)
; ALL-NEXT:    [[CMPEQ0:%.*]] = icmp eq i32 [[CALL]], 0
; ALL-NEXT:    ret i1 [[CMPEQ0]]
;
  %call = tail call i32 @memcmp(i8* %x, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @charbuf, i64 0, i64 0), i64 4)
  %cmpeq0 = icmp eq i32 %call, 0
  ret i1 %cmpeq0
}

; TODO: We still don't care about alignment of the constant. We are not limited to constant folding only i8 arrays.
; It doesn't matter if the constant operand is the first operand to the memcmp.

@intbuf_unaligned = private unnamed_addr constant [4 x i16] [i16 1, i16 2, i16 3, i16 4], align 1

define i1 @memcmp_4bytes_unaligned_constant_i16(i8* align 4 %x) {
; ALL-LABEL: @memcmp_4bytes_unaligned_constant_i16(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* bitcast ([4 x i16]* @intbuf_unaligned to i8*), i8* %x, i64 4)
; ALL-NEXT:    [[CMPEQ0:%.*]] = icmp eq i32 [[CALL]], 0
; ALL-NEXT:    ret i1 [[CMPEQ0]]
;
  %call = tail call i32 @memcmp(i8* bitcast (i16* getelementptr inbounds ([4 x i16], [4 x i16]* @intbuf_unaligned, i64 0, i64 0) to i8*), i8* %x, i64 4)
  %cmpeq0 = icmp eq i32 %call, 0
  ret i1 %cmpeq0
}

; TODO: Any memcmp where all arguments are constants should be constant folded. Currently, we only handle i8 array constants.

@intbuf = private unnamed_addr constant [2 x i32] [i32 0, i32 1], align 4

define i1 @memcmp_3bytes_aligned_constant_i32(i8* align 4 %x) {
; ALL-LABEL: @memcmp_3bytes_aligned_constant_i32(
; ALL-NEXT:    [[CALL:%.*]] = tail call i32 @memcmp(i8* bitcast (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @intbuf, i64 0, i64 1) to i8*), i8* bitcast ([2 x i32]* @intbuf to i8*), i64 3)
; ALL-NEXT:    [[CMPEQ0:%.*]] = icmp eq i32 [[CALL]], 0
; ALL-NEXT:    ret i1 [[CMPEQ0]]
;
  %call = tail call i32 @memcmp(i8* bitcast (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @intbuf, i64 0, i64 1) to i8*), i8* bitcast (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @intbuf, i64 0, i64 0) to i8*), i64 3)
  %cmpeq0 = icmp eq i32 %call, 0
  ret i1 %cmpeq0
}


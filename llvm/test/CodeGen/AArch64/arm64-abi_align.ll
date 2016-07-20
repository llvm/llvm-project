; RUN: llc < %s -mtriple=arm64-apple-darwin -mcpu=cyclone -enable-misched=false -disable-fp-elim | FileCheck %s
; RUN: llc < %s -mtriple=arm64-apple-darwin -O0 -disable-fp-elim | FileCheck -check-prefix=FAST %s

; rdar://12648441
; Generated from arm64-arguments.c with -O2.
; Test passing structs with size < 8, < 16 and > 16
; with alignment of 16 and without

; Structs with size < 8
%struct.s38 = type { i32, i16 }
; With alignment of 16, the size will be padded to multiple of 16 bytes.
%struct.s39 = type { i32, i16, [10 x i8] }
; Structs with size < 16
%struct.s40 = type { i32, i16, i32, i16 }
%struct.s41 = type { i32, i16, i32, i16 }
; Structs with size > 16
%struct.s42 = type { i32, i16, i32, i16, i32, i16 }
%struct.s43 = type { i32, i16, i32, i16, i32, i16, [10 x i8] }

@g38 = common global %struct.s38 zeroinitializer, align 4
@g38_2 = common global %struct.s38 zeroinitializer, align 4
@g39 = common global %struct.s39 zeroinitializer, align 16
@g39_2 = common global %struct.s39 zeroinitializer, align 16
@g40 = common global %struct.s40 zeroinitializer, align 4
@g40_2 = common global %struct.s40 zeroinitializer, align 4
@g41 = common global %struct.s41 zeroinitializer, align 16
@g41_2 = common global %struct.s41 zeroinitializer, align 16
@g42 = common global %struct.s42 zeroinitializer, align 4
@g42_2 = common global %struct.s42 zeroinitializer, align 4
@g43 = common global %struct.s43 zeroinitializer, align 16
@g43_2 = common global %struct.s43 zeroinitializer, align 16

; structs with size < 8 bytes, passed via i64 in x1 and x2
define i32 @f38(i32 %i, i64 %s1.coerce, i64 %s2.coerce) #0 {
entry:
; CHECK-LABEL: f38
; CHECK: add w[[A:[0-9]+]], w1, w0
; CHECK: add {{w[0-9]+}}, w[[A]], w2
  %s1.sroa.0.0.extract.trunc = trunc i64 %s1.coerce to i32
  %s1.sroa.1.4.extract.shift = lshr i64 %s1.coerce, 32
  %s2.sroa.0.0.extract.trunc = trunc i64 %s2.coerce to i32
  %s2.sroa.1.4.extract.shift = lshr i64 %s2.coerce, 32
  %sext8 = shl nuw nsw i64 %s1.sroa.1.4.extract.shift, 16
  %sext = trunc i64 %sext8 to i32
  %conv = ashr exact i32 %sext, 16
  %sext1011 = shl nuw nsw i64 %s2.sroa.1.4.extract.shift, 16
  %sext10 = trunc i64 %sext1011 to i32
  %conv6 = ashr exact i32 %sext10, 16
  %add = add i32 %s1.sroa.0.0.extract.trunc, %i
  %add3 = add i32 %add, %s2.sroa.0.0.extract.trunc
  %add4 = add i32 %add3, %conv
  %add7 = add i32 %add4, %conv6
  ret i32 %add7
}

define i32 @caller38() #1 {
entry:
; CHECK-LABEL: caller38
; CHECK: ldr x1,
; CHECK: ldr x2,
  %0 = load i64, i64* bitcast (%struct.s38* @g38 to i64*), align 4
  %1 = load i64, i64* bitcast (%struct.s38* @g38_2 to i64*), align 4
  %call = tail call i32 @f38(i32 3, i64 %0, i64 %1) #5
  ret i32 %call
}

declare i32 @f38_stack(i32 %i, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6,
                i32 %i7, i32 %i8, i32 %i9, i64 %s1.coerce, i64 %s2.coerce) #0

; structs with size < 8 bytes, passed on stack at [sp+8] and [sp+16]
; i9 at [sp]
define i32 @caller38_stack() #1 {
entry:
; CHECK-LABEL: caller38_stack
; CHECK: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #8]
; CHECK: mov w[[C:[0-9]+]], #9
; CHECK: str w[[C]], [sp]
  %0 = load i64, i64* bitcast (%struct.s38* @g38 to i64*), align 4
  %1 = load i64, i64* bitcast (%struct.s38* @g38_2 to i64*), align 4
  %call = tail call i32 @f38_stack(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6,
                                   i32 7, i32 8, i32 9, i64 %0, i64 %1) #5
  ret i32 %call
}

; structs with size < 8 bytes, alignment of 16
; passed via i128 in x1 and x3
define i32 @f39(i32 %i, i128 %s1.coerce, i128 %s2.coerce) #0 {
entry:
; CHECK-LABEL: f39
; CHECK: add w[[A:[0-9]+]], w1, w0
; CHECK: add {{w[0-9]+}}, w[[A]], w3
  %s1.sroa.0.0.extract.trunc = trunc i128 %s1.coerce to i32
  %s1.sroa.1.4.extract.shift = lshr i128 %s1.coerce, 32
  %s2.sroa.0.0.extract.trunc = trunc i128 %s2.coerce to i32
  %s2.sroa.1.4.extract.shift = lshr i128 %s2.coerce, 32
  %sext8 = shl nuw nsw i128 %s1.sroa.1.4.extract.shift, 16
  %sext = trunc i128 %sext8 to i32
  %conv = ashr exact i32 %sext, 16
  %sext1011 = shl nuw nsw i128 %s2.sroa.1.4.extract.shift, 16
  %sext10 = trunc i128 %sext1011 to i32
  %conv6 = ashr exact i32 %sext10, 16
  %add = add i32 %s1.sroa.0.0.extract.trunc, %i
  %add3 = add i32 %add, %s2.sroa.0.0.extract.trunc
  %add4 = add i32 %add3, %conv
  %add7 = add i32 %add4, %conv6
  ret i32 %add7
}

define i32 @caller39() #1 {
entry:
; CHECK-LABEL: caller39
; CHECK: ldp x1, x2,
; CHECK: ldp x3, x4,
  %0 = load i128, i128* bitcast (%struct.s39* @g39 to i128*), align 16
  %1 = load i128, i128* bitcast (%struct.s39* @g39_2 to i128*), align 16
  %call = tail call i32 @f39(i32 3, i128 %0, i128 %1) #5
  ret i32 %call
}

declare i32 @f39_stack(i32 %i, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6,
                i32 %i7, i32 %i8, i32 %i9, i128 %s1.coerce, i128 %s2.coerce) #0

; structs with size < 8 bytes, alignment 16
; passed on stack at [sp+16] and [sp+32]
define i32 @caller39_stack() #1 {
entry:
; CHECK-LABEL: caller39_stack
; CHECK: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #32]
; CHECK: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #16]
; CHECK: mov w[[C:[0-9]+]], #9
; CHECK: str w[[C]], [sp]
  %0 = load i128, i128* bitcast (%struct.s39* @g39 to i128*), align 16
  %1 = load i128, i128* bitcast (%struct.s39* @g39_2 to i128*), align 16
  %call = tail call i32 @f39_stack(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6,
                                   i32 7, i32 8, i32 9, i128 %0, i128 %1) #5
  ret i32 %call
}

; structs with size < 16 bytes
; passed via i128 in x1 and x3
define i32 @f40(i32 %i, [2 x i64] %s1.coerce, [2 x i64] %s2.coerce) #0 {
entry:
; CHECK-LABEL: f40
; CHECK: add w[[A:[0-9]+]], w1, w0
; CHECK: add {{w[0-9]+}}, w[[A]], w3
  %s1.coerce.fca.0.extract = extractvalue [2 x i64] %s1.coerce, 0
  %s2.coerce.fca.0.extract = extractvalue [2 x i64] %s2.coerce, 0
  %s1.sroa.0.0.extract.trunc = trunc i64 %s1.coerce.fca.0.extract to i32
  %s2.sroa.0.0.extract.trunc = trunc i64 %s2.coerce.fca.0.extract to i32
  %s1.sroa.0.4.extract.shift = lshr i64 %s1.coerce.fca.0.extract, 32
  %sext8 = shl nuw nsw i64 %s1.sroa.0.4.extract.shift, 16
  %sext = trunc i64 %sext8 to i32
  %conv = ashr exact i32 %sext, 16
  %s2.sroa.0.4.extract.shift = lshr i64 %s2.coerce.fca.0.extract, 32
  %sext1011 = shl nuw nsw i64 %s2.sroa.0.4.extract.shift, 16
  %sext10 = trunc i64 %sext1011 to i32
  %conv6 = ashr exact i32 %sext10, 16
  %add = add i32 %s1.sroa.0.0.extract.trunc, %i
  %add3 = add i32 %add, %s2.sroa.0.0.extract.trunc
  %add4 = add i32 %add3, %conv
  %add7 = add i32 %add4, %conv6
  ret i32 %add7
}

define i32 @caller40() #1 {
entry:
; CHECK-LABEL: caller40
; CHECK: ldp x1, x2,
; CHECK: ldp x3, x4,
  %0 = load [2 x i64], [2 x i64]* bitcast (%struct.s40* @g40 to [2 x i64]*), align 4
  %1 = load [2 x i64], [2 x i64]* bitcast (%struct.s40* @g40_2 to [2 x i64]*), align 4
  %call = tail call i32 @f40(i32 3, [2 x i64] %0, [2 x i64] %1) #5
  ret i32 %call
}

declare i32 @f40_stack(i32 %i, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6,
                i32 %i7, i32 %i8, i32 %i9, [2 x i64] %s1.coerce, [2 x i64] %s2.coerce) #0

; structs with size < 16 bytes
; passed on stack at [sp+8] and [sp+24]
define i32 @caller40_stack() #1 {
entry:
; CHECK-LABEL: caller40_stack
; CHECK: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #24]
; CHECK: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #8]
; CHECK: mov w[[C:[0-9]+]], #9
; CHECK: str w[[C]], [sp]
  %0 = load [2 x i64], [2 x i64]* bitcast (%struct.s40* @g40 to [2 x i64]*), align 4
  %1 = load [2 x i64], [2 x i64]* bitcast (%struct.s40* @g40_2 to [2 x i64]*), align 4
  %call = tail call i32 @f40_stack(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6,
                         i32 7, i32 8, i32 9, [2 x i64] %0, [2 x i64] %1) #5
  ret i32 %call
}

; structs with size < 16 bytes, alignment of 16
; passed via i128 in x1 and x3
define i32 @f41(i32 %i, i128 %s1.coerce, i128 %s2.coerce) #0 {
entry:
; CHECK-LABEL: f41
; CHECK: add w[[A:[0-9]+]], w1, w0
; CHECK: add {{w[0-9]+}}, w[[A]], w3
  %s1.sroa.0.0.extract.trunc = trunc i128 %s1.coerce to i32
  %s1.sroa.1.4.extract.shift = lshr i128 %s1.coerce, 32
  %s2.sroa.0.0.extract.trunc = trunc i128 %s2.coerce to i32
  %s2.sroa.1.4.extract.shift = lshr i128 %s2.coerce, 32
  %sext8 = shl nuw nsw i128 %s1.sroa.1.4.extract.shift, 16
  %sext = trunc i128 %sext8 to i32
  %conv = ashr exact i32 %sext, 16
  %sext1011 = shl nuw nsw i128 %s2.sroa.1.4.extract.shift, 16
  %sext10 = trunc i128 %sext1011 to i32
  %conv6 = ashr exact i32 %sext10, 16
  %add = add i32 %s1.sroa.0.0.extract.trunc, %i
  %add3 = add i32 %add, %s2.sroa.0.0.extract.trunc
  %add4 = add i32 %add3, %conv
  %add7 = add i32 %add4, %conv6
  ret i32 %add7
}

define i32 @caller41() #1 {
entry:
; CHECK-LABEL: caller41
; CHECK: ldp x1, x2,
; CHECK: ldp x3, x4,
  %0 = load i128, i128* bitcast (%struct.s41* @g41 to i128*), align 16
  %1 = load i128, i128* bitcast (%struct.s41* @g41_2 to i128*), align 16
  %call = tail call i32 @f41(i32 3, i128 %0, i128 %1) #5
  ret i32 %call
}

declare i32 @f41_stack(i32 %i, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6,
                i32 %i7, i32 %i8, i32 %i9, i128 %s1.coerce, i128 %s2.coerce) #0

; structs with size < 16 bytes, alignment of 16
; passed on stack at [sp+16] and [sp+32]
define i32 @caller41_stack() #1 {
entry:
; CHECK-LABEL: caller41_stack
; CHECK: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #32]
; CHECK: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp, #16]
; CHECK: mov w[[C:[0-9]+]], #9
; CHECK: str w[[C]], [sp]
  %0 = load i128, i128* bitcast (%struct.s41* @g41 to i128*), align 16
  %1 = load i128, i128* bitcast (%struct.s41* @g41_2 to i128*), align 16
  %call = tail call i32 @f41_stack(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6,
                            i32 7, i32 8, i32 9, i128 %0, i128 %1) #5
  ret i32 %call
}

; structs with size of 22 bytes, passed indirectly in x1 and x2
define i32 @f42(i32 %i, %struct.s42* nocapture %s1, %struct.s42* nocapture %s2) #2 {
entry:
; CHECK-LABEL: f42
; CHECK: ldr w[[A:[0-9]+]], [x1]
; CHECK: ldr w[[B:[0-9]+]], [x2]
; CHECK: add w[[C:[0-9]+]], w[[A]], w0
; CHECK: add {{w[0-9]+}}, w[[C]], w[[B]]
; FAST: f42
; FAST: ldr w[[A:[0-9]+]], [x1]
; FAST: ldr w[[B:[0-9]+]], [x2]
; FAST: add w[[C:[0-9]+]], w[[A]], w0
; FAST: add {{w[0-9]+}}, w[[C]], w[[B]]
  %i1 = getelementptr inbounds %struct.s42, %struct.s42* %s1, i64 0, i32 0
  %0 = load i32, i32* %i1, align 4, !tbaa !0
  %i2 = getelementptr inbounds %struct.s42, %struct.s42* %s2, i64 0, i32 0
  %1 = load i32, i32* %i2, align 4, !tbaa !0
  %s = getelementptr inbounds %struct.s42, %struct.s42* %s1, i64 0, i32 1
  %2 = load i16, i16* %s, align 2, !tbaa !3
  %conv = sext i16 %2 to i32
  %s5 = getelementptr inbounds %struct.s42, %struct.s42* %s2, i64 0, i32 1
  %3 = load i16, i16* %s5, align 2, !tbaa !3
  %conv6 = sext i16 %3 to i32
  %add = add i32 %0, %i
  %add3 = add i32 %add, %1
  %add4 = add i32 %add3, %conv
  %add7 = add i32 %add4, %conv6
  ret i32 %add7
}

; For s1, we allocate a 22-byte space, pass its address via x1
define i32 @caller42() #3 {
entry:
; CHECK-LABEL: caller42
; CHECK: str {{x[0-9]+}}, [sp, #48]
; CHECK: str {{q[0-9]+}}, [sp, #32]
; CHECK: str {{x[0-9]+}}, [sp, #16]
; CHECK: str {{q[0-9]+}}, [sp]
; CHECK: add x1, sp, #32
; CHECK: mov x2, sp
; Space for s1 is allocated at sp+32
; Space for s2 is allocated at sp

; FAST-LABEL: caller42
; FAST: sub sp, sp, #112
; Space for s1 is allocated at fp-24 = sp+72
; Space for s2 is allocated at sp+48
; FAST: sub x[[A:[0-9]+]], x29, #24
; FAST: add x[[A:[0-9]+]], sp, #48
; Call memcpy with size = 24 (0x18)
; FAST: orr {{x[0-9]+}}, xzr, #0x18
  %tmp = alloca %struct.s42, align 4
  %tmp1 = alloca %struct.s42, align 4
  %0 = bitcast %struct.s42* %tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* bitcast (%struct.s42* @g42 to i8*), i64 24, i32 4, i1 false), !tbaa.struct !4
  %1 = bitcast %struct.s42* %tmp1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* bitcast (%struct.s42* @g42_2 to i8*), i64 24, i32 4, i1 false), !tbaa.struct !4
  %call = call i32 @f42(i32 3, %struct.s42* %tmp, %struct.s42* %tmp1) #5
  ret i32 %call
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) #4

declare i32 @f42_stack(i32 %i, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6,
                       i32 %i7, i32 %i8, i32 %i9, %struct.s42* nocapture %s1,
                       %struct.s42* nocapture %s2) #2

define i32 @caller42_stack() #3 {
entry:
; CHECK-LABEL: caller42_stack
; CHECK: sub sp, sp, #112
; CHECK: add x29, sp, #96
; CHECK: stur {{x[0-9]+}}, [x29, #-16]
; CHECK: stur {{q[0-9]+}}, [x29, #-32]
; CHECK: str {{x[0-9]+}}, [sp, #48]
; CHECK: str {{q[0-9]+}}, [sp, #32]
; Space for s1 is allocated at x29-32 = sp+64
; Space for s2 is allocated at sp+32
; CHECK: add x[[B:[0-9]+]], sp, #32
; CHECK: str x[[B]], [sp, #16]
; CHECK: sub x[[A:[0-9]+]], x29, #32
; Address of s1 is passed on stack at sp+8
; CHECK: str x[[A]], [sp, #8]
; CHECK: mov w[[C:[0-9]+]], #9
; CHECK: str w[[C]], [sp]

; FAST-LABEL: caller42_stack
; Space for s1 is allocated at fp-24
; Space for s2 is allocated at fp-48
; FAST: sub x[[A:[0-9]+]], x29, #24
; FAST: sub x[[B:[0-9]+]], x29, #48
; Call memcpy with size = 24 (0x18)
; FAST: orr {{x[0-9]+}}, xzr, #0x18
; FAST: str {{w[0-9]+}}, [sp]
; Address of s1 is passed on stack at sp+8
; FAST: str {{x[0-9]+}}, [sp, #8]
; FAST: str {{x[0-9]+}}, [sp, #16]
  %tmp = alloca %struct.s42, align 4
  %tmp1 = alloca %struct.s42, align 4
  %0 = bitcast %struct.s42* %tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* bitcast (%struct.s42* @g42 to i8*), i64 24, i32 4, i1 false), !tbaa.struct !4
  %1 = bitcast %struct.s42* %tmp1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* bitcast (%struct.s42* @g42_2 to i8*), i64 24, i32 4, i1 false), !tbaa.struct !4
  %call = call i32 @f42_stack(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                       i32 8, i32 9, %struct.s42* %tmp, %struct.s42* %tmp1) #5
  ret i32 %call
}

; structs with size of 22 bytes, alignment of 16
; passed indirectly in x1 and x2
define i32 @f43(i32 %i, %struct.s43* nocapture %s1, %struct.s43* nocapture %s2) #2 {
entry:
; CHECK-LABEL: f43
; CHECK: ldr w[[A:[0-9]+]], [x1]
; CHECK: ldr w[[B:[0-9]+]], [x2]
; CHECK: add w[[C:[0-9]+]], w[[A]], w0
; CHECK: add {{w[0-9]+}}, w[[C]], w[[B]]
; FAST-LABEL: f43
; FAST: ldr w[[A:[0-9]+]], [x1]
; FAST: ldr w[[B:[0-9]+]], [x2]
; FAST: add w[[C:[0-9]+]], w[[A]], w0
; FAST: add {{w[0-9]+}}, w[[C]], w[[B]]
  %i1 = getelementptr inbounds %struct.s43, %struct.s43* %s1, i64 0, i32 0
  %0 = load i32, i32* %i1, align 4, !tbaa !0
  %i2 = getelementptr inbounds %struct.s43, %struct.s43* %s2, i64 0, i32 0
  %1 = load i32, i32* %i2, align 4, !tbaa !0
  %s = getelementptr inbounds %struct.s43, %struct.s43* %s1, i64 0, i32 1
  %2 = load i16, i16* %s, align 2, !tbaa !3
  %conv = sext i16 %2 to i32
  %s5 = getelementptr inbounds %struct.s43, %struct.s43* %s2, i64 0, i32 1
  %3 = load i16, i16* %s5, align 2, !tbaa !3
  %conv6 = sext i16 %3 to i32
  %add = add i32 %0, %i
  %add3 = add i32 %add, %1
  %add4 = add i32 %add3, %conv
  %add7 = add i32 %add4, %conv6
  ret i32 %add7
}

define i32 @caller43() #3 {
entry:
; CHECK-LABEL: caller43
; CHECK: str {{q[0-9]+}}, [sp, #48]
; CHECK: str {{q[0-9]+}}, [sp, #32]
; CHECK: str {{q[0-9]+}}, [sp, #16]
; CHECK: str {{q[0-9]+}}, [sp]
; CHECK: add x1, sp, #32
; CHECK: mov x2, sp
; Space for s1 is allocated at sp+32
; Space for s2 is allocated at sp

; FAST-LABEL: caller43
; FAST: add x29, sp, #64
; Space for s1 is allocated at sp+32
; Space for s2 is allocated at sp
; FAST: add x1, sp, #32
; FAST: mov x2, sp
; FAST: str {{x[0-9]+}}, [sp, #32]
; FAST: str {{x[0-9]+}}, [sp, #40]
; FAST: str {{x[0-9]+}}, [sp, #48]
; FAST: str {{x[0-9]+}}, [sp, #56]
; FAST: str {{x[0-9]+}}, [sp]
; FAST: str {{x[0-9]+}}, [sp, #8]
; FAST: str {{x[0-9]+}}, [sp, #16]
; FAST: str {{x[0-9]+}}, [sp, #24]
  %tmp = alloca %struct.s43, align 16
  %tmp1 = alloca %struct.s43, align 16
  %0 = bitcast %struct.s43* %tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* bitcast (%struct.s43* @g43 to i8*), i64 32, i32 16, i1 false), !tbaa.struct !4
  %1 = bitcast %struct.s43* %tmp1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* bitcast (%struct.s43* @g43_2 to i8*), i64 32, i32 16, i1 false), !tbaa.struct !4
  %call = call i32 @f43(i32 3, %struct.s43* %tmp, %struct.s43* %tmp1) #5
  ret i32 %call
}

declare i32 @f43_stack(i32 %i, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6,
                       i32 %i7, i32 %i8, i32 %i9, %struct.s43* nocapture %s1,
                       %struct.s43* nocapture %s2) #2

define i32 @caller43_stack() #3 {
entry:
; CHECK-LABEL: caller43_stack
; CHECK: sub sp, sp, #112
; CHECK: add x29, sp, #96
; CHECK: stur {{q[0-9]+}}, [x29, #-16]
; CHECK: stur {{q[0-9]+}}, [x29, #-32]
; CHECK: str {{q[0-9]+}}, [sp, #48]
; CHECK: str {{q[0-9]+}}, [sp, #32]
; Space for s1 is allocated at x29-32 = sp+64
; Space for s2 is allocated at sp+32
; CHECK: add x[[B:[0-9]+]], sp, #32
; CHECK: str x[[B]], [sp, #16]
; CHECK: sub x[[A:[0-9]+]], x29, #32
; Address of s1 is passed on stack at sp+8
; CHECK: str x[[A]], [sp, #8]
; CHECK: mov w[[C:[0-9]+]], #9
; CHECK: str w[[C]], [sp]

; FAST-LABEL: caller43_stack
; FAST: sub sp, sp, #112
; Space for s1 is allocated at fp-32 = sp+64
; Space for s2 is allocated at sp+32
; FAST: sub x[[A:[0-9]+]], x29, #32
; FAST: add x[[B:[0-9]+]], sp, #32
; FAST: stur {{x[0-9]+}}, [x29, #-32]
; FAST: stur {{x[0-9]+}}, [x29, #-24]
; FAST: stur {{x[0-9]+}}, [x29, #-16]
; FAST: stur {{x[0-9]+}}, [x29, #-8]
; FAST: str {{x[0-9]+}}, [sp, #32]
; FAST: str {{x[0-9]+}}, [sp, #40]
; FAST: str {{x[0-9]+}}, [sp, #48]
; FAST: str {{x[0-9]+}}, [sp, #56]
; FAST: str {{w[0-9]+}}, [sp]
; Address of s1 is passed on stack at sp+8
; FAST: str {{x[0-9]+}}, [sp, #8]
; FAST: str {{x[0-9]+}}, [sp, #16]
  %tmp = alloca %struct.s43, align 16
  %tmp1 = alloca %struct.s43, align 16
  %0 = bitcast %struct.s43* %tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* bitcast (%struct.s43* @g43 to i8*), i64 32, i32 16, i1 false), !tbaa.struct !4
  %1 = bitcast %struct.s43* %tmp1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* bitcast (%struct.s43* @g43_2 to i8*), i64 32, i32 16, i1 false), !tbaa.struct !4
  %call = call i32 @f43_stack(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                       i32 8, i32 9, %struct.s43* %tmp, %struct.s43* %tmp1) #5
  ret i32 %call
}

; rdar://13668927
; Check that we don't split an i128.
declare i32 @callee_i128_split(i32 %i, i32 %i2, i32 %i3, i32 %i4, i32 %i5,
                               i32 %i6, i32 %i7, i128 %s1, i32 %i8)

define i32 @i128_split() {
entry:
; CHECK-LABEL: i128_split
; "i128 %0" should be on stack at [sp].
; "i32 8" should be on stack at [sp, #16].
; CHECK: str {{w[0-9]+}}, [sp, #16]
; CHECK: stp {{x[0-9]+}}, {{x[0-9]+}}, [sp]
; FAST-LABEL: i128_split
; FAST: sub sp, sp
; FAST: mov x[[ADDR:[0-9]+]], sp
; FAST: str {{w[0-9]+}}, [x[[ADDR]], #16]
; Load/Store opt is disabled with -O0, so the i128 is split.
; FAST: str {{x[0-9]+}}, [x[[ADDR]], #8]
; FAST: str {{x[0-9]+}}, [x[[ADDR]]]
  %0 = load i128, i128* bitcast (%struct.s41* @g41 to i128*), align 16
  %call = tail call i32 @callee_i128_split(i32 1, i32 2, i32 3, i32 4, i32 5,
                                           i32 6, i32 7, i128 %0, i32 8) #5
  ret i32 %call
}

declare i32 @callee_i64(i32 %i, i32 %i2, i32 %i3, i32 %i4, i32 %i5,
                               i32 %i6, i32 %i7, i64 %s1, i32 %i8)

define i32 @i64_split() {
entry:
; CHECK-LABEL: i64_split
; "i64 %0" should be in register x7.
; "i32 8" should be on stack at [sp].
; CHECK: ldr x7, [{{x[0-9]+}}]
; CHECK: str {{w[0-9]+}}, [sp]
; FAST-LABEL: i64_split
; FAST: ldr x7, [{{x[0-9]+}}]
; FAST: mov x[[R0:[0-9]+]], sp
; FAST: orr w[[R1:[0-9]+]], wzr, #0x8
; FAST: str w[[R1]], {{\[}}x[[R0]]{{\]}}
  %0 = load i64, i64* bitcast (%struct.s41* @g41 to i64*), align 16
  %call = tail call i32 @callee_i64(i32 1, i32 2, i32 3, i32 4, i32 5,
                                    i32 6, i32 7, i64 %0, i32 8) #5
  ret i32 %call
}

attributes #0 = { noinline nounwind readnone "fp-contract-model"="standard" "relocation-model"="pic" "ssp-buffers-size"="8" }
attributes #1 = { nounwind readonly "fp-contract-model"="standard" "relocation-model"="pic" "ssp-buffers-size"="8" }
attributes #2 = { noinline nounwind readonly "fp-contract-model"="standard" "relocation-model"="pic" "ssp-buffers-size"="8" }
attributes #3 = { nounwind "fp-contract-model"="standard" "relocation-model"="pic" "ssp-buffers-size"="8" }
attributes #4 = { nounwind }
attributes #5 = { nobuiltin }

!0 = !{!"int", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!"short", !1}
!4 = !{i64 0, i64 4, !0, i64 4, i64 2, !3, i64 8, i64 4, !0, i64 12, i64 2, !3, i64 16, i64 4, !0, i64 20, i64 2, !3}

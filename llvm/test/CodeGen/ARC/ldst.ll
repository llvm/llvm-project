; RUN: llc -march=arc < %s | FileCheck %s

; CHECK-LABEL: load32
; CHECK: ld %r0, [%r0,16000]

define i32 @load32(i32* %bp) nounwind {
entry:
  %gep = getelementptr i32, i32* %bp, i32 4000
  %v = load i32, i32* %gep, align 4
  ret i32 %v
}

; CHECK-LABEL: load16
; CHECK: ldh %r0, [%r0,8000]

define i16 @load16(i16* %bp) nounwind {
entry:
  %gep = getelementptr i16, i16* %bp, i32 4000
  %v = load i16, i16* %gep, align 2
  ret i16 %v
}

; CHECK-LABEL: load8
; CHECK: ldb %r0, [%r0,4000]

define i8 @load8(i8* %bp) nounwind {
entry:
  %gep = getelementptr i8, i8* %bp, i32 4000
  %v = load i8, i8* %gep, align 1
  ret i8 %v
}

; CHECK-LABEL: sextload16
; CHECK: ldh.x %r0, [%r0,8000]

define i32 @sextload16(i16* %bp) nounwind {
entry:
  %gep = getelementptr i16, i16* %bp, i32 4000
  %vl = load i16, i16* %gep, align 2
  %v = sext i16 %vl to i32
  ret i32 %v
}

; CHECK-LABEL: sextload8
; CHECK: ldb.x %r0, [%r0,4000]

define i32 @sextload8(i8* %bp) nounwind {
entry:
  %gep = getelementptr i8, i8* %bp, i32 4000
  %vl = load i8, i8* %gep, align 1
  %v = sext i8 %vl to i32
  ret i32 %v
}

; CHECK-LABEL: s_sextload16
; CHECK: ldh.x %r0, [%r0,32]

define i32 @s_sextload16(i16* %bp) nounwind {
entry:
  %gep = getelementptr i16, i16* %bp, i32 16
  %vl = load i16, i16* %gep, align 2
  %v = sext i16 %vl to i32
  ret i32 %v
}

; CHECK-LABEL: s_sextload8
; CHECK: ldb.x %r0, [%r0,16]

define i32 @s_sextload8(i8* %bp) nounwind {
entry:
  %gep = getelementptr i8, i8* %bp, i32 16
  %vl = load i8, i8* %gep, align 1
  %v = sext i8 %vl to i32
  ret i32 %v
}

; CHECK-LABEL: store32
; CHECK: add %r[[REG:[0-9]+]], %r1, 16000
; CHECK: st %r0, [%r[[REG]],0]

; Long range stores (offset does not fit in s9) must be add followed by st.
define void @store32(i32 %val, i32* %bp) nounwind {
entry:
  %gep = getelementptr i32, i32* %bp, i32 4000
  store i32 %val, i32* %gep, align 4
  ret void
}

; CHECK-LABEL: store16
; CHECK: add %r[[REG:[0-9]+]], %r1, 8000
; CHECK: sth %r0, [%r[[REG]],0]

define void @store16(i16 zeroext %val, i16* %bp) nounwind {
entry:
  %gep = getelementptr i16, i16* %bp, i32 4000
  store i16 %val, i16* %gep, align 2
  ret void
}

; CHECK-LABEL: store8
; CHECK: add %r[[REG:[0-9]+]], %r1, 4000
; CHECK: stb %r0, [%r[[REG]],0]

define void @store8(i8 zeroext %val, i8* %bp) nounwind {
entry:
  %gep = getelementptr i8, i8* %bp, i32 4000
  store i8 %val, i8* %gep, align 1
  ret void
}

; Short range stores can be done with [reg, s9].
; CHECK-LABEL: s_store32
; CHECK-NOT: add
; CHECK: st %r0, [%r1,64]
define void @s_store32(i32 %val, i32* %bp) nounwind {
entry:
  %gep = getelementptr i32, i32* %bp, i32 16
  store i32 %val, i32* %gep, align 4
  ret void
}

; CHECK-LABEL: s_store16
; CHECK-NOT: add
; CHECK: sth %r0, [%r1,32]
define void @s_store16(i16 zeroext %val, i16* %bp) nounwind {
entry:
  %gep = getelementptr i16, i16* %bp, i32 16
  store i16 %val, i16* %gep, align 2
  ret void
}

; CHECK-LABEL: s_store8
; CHECK-NOT: add
; CHECK: stb %r0, [%r1,16]
define void @s_store8(i8 zeroext %val, i8* %bp) nounwind {
entry:
  %gep = getelementptr i8, i8* %bp, i32 16
  store i8 %val, i8* %gep, align 1
  ret void
}


@aaaa = internal global [128 x i32] zeroinitializer
@bbbb = internal global [128 x i16] zeroinitializer
@cccc = internal global [128 x i8]  zeroinitializer

; CHECK-LABEL: g_store32
; CHECK-NOT: add
; CHECK: st %r0, [@aaaa+64]
define void @g_store32(i32 %val) nounwind {
entry:
  store i32 %val, i32* getelementptr inbounds ([128 x i32], [128 x i32]* @aaaa, i32 0, i32 16), align 4
  ret void
}

; CHECK-LABEL: g_load32
; CHECK-NOT: add
; CHECK: ld %r0, [@aaaa+64]
define i32 @g_load32() nounwind {
  %gep = getelementptr inbounds [128 x i32], [128 x i32]* @aaaa, i32 0, i32 16
  %v = load i32, i32* %gep, align 4
  ret i32 %v
}

; CHECK-LABEL: g_store16
; CHECK-NOT: add
; CHECK: sth %r0, [@bbbb+32]
define void @g_store16(i16 %val) nounwind {
entry:
  store i16 %val, i16* getelementptr inbounds ([128 x i16], [128 x i16]* @bbbb, i16 0, i16 16), align 2
  ret void
}

; CHECK-LABEL: g_load16
; CHECK-NOT: add
; CHECK: ldh %r0, [@bbbb+32]
define i16 @g_load16() nounwind {
  %gep = getelementptr inbounds [128 x i16], [128 x i16]* @bbbb, i16 0, i16 16
  %v = load i16, i16* %gep, align 2
  ret i16 %v
}

; CHECK-LABEL: g_store8
; CHECK-NOT: add
; CHECK: stb %r0, [@cccc+16]
define void @g_store8(i8 %val) nounwind {
entry:
  store i8 %val, i8* getelementptr inbounds ([128 x i8], [128 x i8]* @cccc, i8 0, i8 16), align 1
  ret void
}

; CHECK-LABEL: g_load8
; CHECK-NOT: add
; CHECK: ldb %r0, [@cccc+16]
define i8 @g_load8() nounwind {
  %gep = getelementptr inbounds [128 x i8], [128 x i8]* @cccc, i8 0, i8 16
  %v = load i8, i8* %gep, align 1
  ret i8 %v
}

; CHECK-LABEL: align2_load32
; CHECK-DAG: ldh %r[[REG0:[0-9]+]], [%r0,0]
; CHECK-DAG: ldh %r[[REG1:[0-9]+]], [%r0,2]
; CHECK-DAG: asl %r[[REG2:[0-9]+]], %r[[REG1]], 16
define i32 @align2_load32(i8* %p) nounwind {
entry:
  %bp = bitcast i8* %p to i32*
  %v = load i32, i32* %bp, align 2
  ret i32 %v
}

; CHECK-LABEL: align1_load32
; CHECK-DAG: ldb %r[[REG0:[0-9]+]], [%r0,0]
; CHECK-DAG: ldb %r[[REG1:[0-9]+]], [%r0,1]
; CHECK-DAG: ldb %r[[REG2:[0-9]+]], [%r0,2]
; CHECK-DAG: ldb %r[[REG3:[0-9]+]], [%r0,3]
; CHECK-DAG: asl %r[[AREG1:[0-9]+]], %r[[REG1]], 8
; CHECK-DAG: asl %r[[AREG3:[0-9]+]], %r[[REG3]], 8
define i32 @align1_load32(i8* %p) nounwind {
entry:
  %bp = bitcast i8* %p to i32*
  %v = load i32, i32* %bp, align 1
  ret i32 %v
}

; CHECK-LABEL: align1_load16
; CHECK-DAG: ldb %r[[REG0:[0-9]+]], [%r0,0]
; CHECK-DAG: ldb %r[[REG1:[0-9]+]], [%r0,1]
; CHECK-DAG: asl %r[[REG2:[0-9]+]], %r[[REG1]], 8
define i16 @align1_load16(i8* %p) nounwind {
entry:
  %bp = bitcast i8* %p to i16*
  %v = load i16, i16* %bp, align 1
  ret i16 %v
}

; CHECK-LABEL: align2_store32
; CHECK-DAG: lsr %r[[REG:[0-9]+]], %r1, 16
; CHECK-DAG: sth %r1, [%r0,0]
; CHECK-DAG: sth %r[[REG:[0-9]+]], [%r0,2]
define void @align2_store32(i8* %p, i32 %v) nounwind {
entry:
  %bp = bitcast i8* %p to i32*
  store i32 %v, i32* %bp, align 2
  ret void
}

; CHECK-LABEL: align1_store16
; CHECK-DAG: lsr %r[[REG:[0-9]+]], %r1, 8
; CHECK-DAG: stb %r1, [%r0,0]
; CHECK-DAG: stb %r[[REG:[0-9]+]], [%r0,1]
define void @align1_store16(i8* %p, i16 %v) nounwind {
entry:
  %bp = bitcast i8* %p to i16*
  store i16 %v, i16* %bp, align 1
  ret void
}

; CHECK-LABEL: align1_store32
; CHECK-DAG: lsr %r[[REG0:[0-9]+]], %r1, 8
; CHECK-DAG: lsr %r[[REG1:[0-9]+]], %r1, 16
; CHECK-DAG: lsr %r[[REG2:[0-9]+]], %r1, 24
; CHECK-DAG: stb %r1, [%r0,0]
; CHECK-DAG: stb %r[[REG0]], [%r0,1]
; CHECK-DAG: stb %r[[REG1]], [%r0,2]
; CHECK-DAG: stb %r[[REG2]], [%r0,3]
define void @align1_store32(i8* %p, i32 %v) nounwind {
entry:
  %bp = bitcast i8* %p to i32*
  store i32 %v, i32* %bp, align 1
  ret void
}

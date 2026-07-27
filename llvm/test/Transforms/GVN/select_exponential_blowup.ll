; RUN: opt < %s -passes=gvn -disable-output

; Repeated logical-or selects used to cause GVN equality propagation to
; rediscover the same equality facts through exponentially many paths.

define void @logical_select_or_worklist_dedup(i1 %c0, ptr %p) {
entry:
  br label %bb0

bb0:
  %p0_0 = getelementptr i1, ptr %p, i64 0
  %p0_1 = getelementptr i1, ptr %p, i64 1
  %p0_2 = getelementptr i1, ptr %p, i64 2
  %x0_0 = load i1, ptr %p0_0
  %x0_1 = load i1, ptr %p0_1
  %x0_2 = load i1, ptr %p0_2
  %s0_0 = select i1 %c0, i1 true, i1 %x0_0
  %s0_1 = select i1 %c0, i1 true, i1 %x0_1
  %s0_2 = select i1 %c0, i1 true, i1 %x0_2
  %o0 = or i1 %s0_0, %s0_1
  %b0 = or i1 %o0, %s0_2
  br label %bb1

bb1:
  %p1_0 = getelementptr i1, ptr %p, i64 3
  %p1_1 = getelementptr i1, ptr %p, i64 4
  %p1_2 = getelementptr i1, ptr %p, i64 5
  %x1_0 = load i1, ptr %p1_0
  %x1_1 = load i1, ptr %p1_1
  %x1_2 = load i1, ptr %p1_2
  %s1_0 = select i1 %b0, i1 true, i1 %x1_0
  %s1_1 = select i1 %b0, i1 true, i1 %x1_1
  %s1_2 = select i1 %b0, i1 true, i1 %x1_2
  %o1 = or i1 %s1_0, %s1_1
  %b1 = or i1 %o1, %s1_2
  br label %bb2

bb2:
  %p2_0 = getelementptr i1, ptr %p, i64 6
  %p2_1 = getelementptr i1, ptr %p, i64 7
  %p2_2 = getelementptr i1, ptr %p, i64 8
  %x2_0 = load i1, ptr %p2_0
  %x2_1 = load i1, ptr %p2_1
  %x2_2 = load i1, ptr %p2_2
  %s2_0 = select i1 %b1, i1 true, i1 %x2_0
  %s2_1 = select i1 %b1, i1 true, i1 %x2_1
  %s2_2 = select i1 %b1, i1 true, i1 %x2_2
  %o2 = or i1 %s2_0, %s2_1
  %b2 = or i1 %o2, %s2_2
  br label %bb3

bb3:
  %p3_0 = getelementptr i1, ptr %p, i64 9
  %p3_1 = getelementptr i1, ptr %p, i64 10
  %p3_2 = getelementptr i1, ptr %p, i64 11
  %x3_0 = load i1, ptr %p3_0
  %x3_1 = load i1, ptr %p3_1
  %x3_2 = load i1, ptr %p3_2
  %s3_0 = select i1 %b2, i1 true, i1 %x3_0
  %s3_1 = select i1 %b2, i1 true, i1 %x3_1
  %s3_2 = select i1 %b2, i1 true, i1 %x3_2
  %o3 = or i1 %s3_0, %s3_1
  %b3 = or i1 %o3, %s3_2
  br label %bb4

bb4:
  %p4_0 = getelementptr i1, ptr %p, i64 12
  %p4_1 = getelementptr i1, ptr %p, i64 13
  %p4_2 = getelementptr i1, ptr %p, i64 14
  %x4_0 = load i1, ptr %p4_0
  %x4_1 = load i1, ptr %p4_1
  %x4_2 = load i1, ptr %p4_2
  %s4_0 = select i1 %b3, i1 true, i1 %x4_0
  %s4_1 = select i1 %b3, i1 true, i1 %x4_1
  %s4_2 = select i1 %b3, i1 true, i1 %x4_2
  %o4 = or i1 %s4_0, %s4_1
  %b4 = or i1 %o4, %s4_2
  br label %bb5

bb5:
  %p5_0 = getelementptr i1, ptr %p, i64 15
  %p5_1 = getelementptr i1, ptr %p, i64 16
  %p5_2 = getelementptr i1, ptr %p, i64 17
  %x5_0 = load i1, ptr %p5_0
  %x5_1 = load i1, ptr %p5_1
  %x5_2 = load i1, ptr %p5_2
  %s5_0 = select i1 %b4, i1 true, i1 %x5_0
  %s5_1 = select i1 %b4, i1 true, i1 %x5_1
  %s5_2 = select i1 %b4, i1 true, i1 %x5_2
  %o5 = or i1 %s5_0, %s5_1
  %b5 = or i1 %o5, %s5_2
  br label %bb6

bb6:
  %p6_0 = getelementptr i1, ptr %p, i64 18
  %p6_1 = getelementptr i1, ptr %p, i64 19
  %p6_2 = getelementptr i1, ptr %p, i64 20
  %x6_0 = load i1, ptr %p6_0
  %x6_1 = load i1, ptr %p6_1
  %x6_2 = load i1, ptr %p6_2
  %s6_0 = select i1 %b5, i1 true, i1 %x6_0
  %s6_1 = select i1 %b5, i1 true, i1 %x6_1
  %s6_2 = select i1 %b5, i1 true, i1 %x6_2
  %o6 = or i1 %s6_0, %s6_1
  %b6 = or i1 %o6, %s6_2
  br label %bb7

bb7:
  %p7_0 = getelementptr i1, ptr %p, i64 21
  %p7_1 = getelementptr i1, ptr %p, i64 22
  %p7_2 = getelementptr i1, ptr %p, i64 23
  %x7_0 = load i1, ptr %p7_0
  %x7_1 = load i1, ptr %p7_1
  %x7_2 = load i1, ptr %p7_2
  %s7_0 = select i1 %b6, i1 true, i1 %x7_0
  %s7_1 = select i1 %b6, i1 true, i1 %x7_1
  %s7_2 = select i1 %b6, i1 true, i1 %x7_2
  %o7 = or i1 %s7_0, %s7_1
  %b7 = or i1 %o7, %s7_2
  br label %bb8

bb8:
  %p8_0 = getelementptr i1, ptr %p, i64 24
  %p8_1 = getelementptr i1, ptr %p, i64 25
  %p8_2 = getelementptr i1, ptr %p, i64 26
  %x8_0 = load i1, ptr %p8_0
  %x8_1 = load i1, ptr %p8_1
  %x8_2 = load i1, ptr %p8_2
  %s8_0 = select i1 %b7, i1 true, i1 %x8_0
  %s8_1 = select i1 %b7, i1 true, i1 %x8_1
  %s8_2 = select i1 %b7, i1 true, i1 %x8_2
  %o8 = or i1 %s8_0, %s8_1
  %b8 = or i1 %o8, %s8_2
  br label %bb9

bb9:
  %p9_0 = getelementptr i1, ptr %p, i64 27
  %p9_1 = getelementptr i1, ptr %p, i64 28
  %p9_2 = getelementptr i1, ptr %p, i64 29
  %x9_0 = load i1, ptr %p9_0
  %x9_1 = load i1, ptr %p9_1
  %x9_2 = load i1, ptr %p9_2
  %s9_0 = select i1 %b8, i1 true, i1 %x9_0
  %s9_1 = select i1 %b8, i1 true, i1 %x9_1
  %s9_2 = select i1 %b8, i1 true, i1 %x9_2
  %o9 = or i1 %s9_0, %s9_1
  %b9 = or i1 %o9, %s9_2
  br label %bb10

bb10:
  %p10_0 = getelementptr i1, ptr %p, i64 30
  %p10_1 = getelementptr i1, ptr %p, i64 31
  %p10_2 = getelementptr i1, ptr %p, i64 32
  %x10_0 = load i1, ptr %p10_0
  %x10_1 = load i1, ptr %p10_1
  %x10_2 = load i1, ptr %p10_2
  %s10_0 = select i1 %b9, i1 true, i1 %x10_0
  %s10_1 = select i1 %b9, i1 true, i1 %x10_1
  %s10_2 = select i1 %b9, i1 true, i1 %x10_2
  %o10 = or i1 %s10_0, %s10_1
  %b10 = or i1 %o10, %s10_2
  br label %bb11

bb11:
  %p11_0 = getelementptr i1, ptr %p, i64 33
  %p11_1 = getelementptr i1, ptr %p, i64 34
  %p11_2 = getelementptr i1, ptr %p, i64 35
  %x11_0 = load i1, ptr %p11_0
  %x11_1 = load i1, ptr %p11_1
  %x11_2 = load i1, ptr %p11_2
  %s11_0 = select i1 %b10, i1 true, i1 %x11_0
  %s11_1 = select i1 %b10, i1 true, i1 %x11_1
  %s11_2 = select i1 %b10, i1 true, i1 %x11_2
  %o11 = or i1 %s11_0, %s11_1
  %b11 = or i1 %o11, %s11_2
  br label %bb12

bb12:
  %p12_0 = getelementptr i1, ptr %p, i64 36
  %p12_1 = getelementptr i1, ptr %p, i64 37
  %p12_2 = getelementptr i1, ptr %p, i64 38
  %x12_0 = load i1, ptr %p12_0
  %x12_1 = load i1, ptr %p12_1
  %x12_2 = load i1, ptr %p12_2
  %s12_0 = select i1 %b11, i1 true, i1 %x12_0
  %s12_1 = select i1 %b11, i1 true, i1 %x12_1
  %s12_2 = select i1 %b11, i1 true, i1 %x12_2
  %o12 = or i1 %s12_0, %s12_1
  %b12 = or i1 %o12, %s12_2
  br label %bb13

bb13:
  %p13_0 = getelementptr i1, ptr %p, i64 39
  %p13_1 = getelementptr i1, ptr %p, i64 40
  %p13_2 = getelementptr i1, ptr %p, i64 41
  %x13_0 = load i1, ptr %p13_0
  %x13_1 = load i1, ptr %p13_1
  %x13_2 = load i1, ptr %p13_2
  %s13_0 = select i1 %b12, i1 true, i1 %x13_0
  %s13_1 = select i1 %b12, i1 true, i1 %x13_1
  %s13_2 = select i1 %b12, i1 true, i1 %x13_2
  %o13 = or i1 %s13_0, %s13_1
  %b13 = or i1 %o13, %s13_2
  br label %bb14

bb14:
  %p14_0 = getelementptr i1, ptr %p, i64 42
  %p14_1 = getelementptr i1, ptr %p, i64 43
  %p14_2 = getelementptr i1, ptr %p, i64 44
  %x14_0 = load i1, ptr %p14_0
  %x14_1 = load i1, ptr %p14_1
  %x14_2 = load i1, ptr %p14_2
  %s14_0 = select i1 %b13, i1 true, i1 %x14_0
  %s14_1 = select i1 %b13, i1 true, i1 %x14_1
  %s14_2 = select i1 %b13, i1 true, i1 %x14_2
  %o14 = or i1 %s14_0, %s14_1
  %b14 = or i1 %o14, %s14_2
  br label %bb15

bb15:
  %p15_0 = getelementptr i1, ptr %p, i64 45
  %p15_1 = getelementptr i1, ptr %p, i64 46
  %p15_2 = getelementptr i1, ptr %p, i64 47
  %x15_0 = load i1, ptr %p15_0
  %x15_1 = load i1, ptr %p15_1
  %x15_2 = load i1, ptr %p15_2
  %s15_0 = select i1 %b14, i1 true, i1 %x15_0
  %s15_1 = select i1 %b14, i1 true, i1 %x15_1
  %s15_2 = select i1 %b14, i1 true, i1 %x15_2
  %o15 = or i1 %s15_0, %s15_1
  %b15 = or i1 %o15, %s15_2
  br label %bb16

bb16:
  %p16_0 = getelementptr i1, ptr %p, i64 48
  %p16_1 = getelementptr i1, ptr %p, i64 49
  %p16_2 = getelementptr i1, ptr %p, i64 50
  %x16_0 = load i1, ptr %p16_0
  %x16_1 = load i1, ptr %p16_1
  %x16_2 = load i1, ptr %p16_2
  %s16_0 = select i1 %b15, i1 true, i1 %x16_0
  %s16_1 = select i1 %b15, i1 true, i1 %x16_1
  %s16_2 = select i1 %b15, i1 true, i1 %x16_2
  %o16 = or i1 %s16_0, %s16_1
  %b16 = or i1 %o16, %s16_2
  br label %bb17

bb17:
  %p17_0 = getelementptr i1, ptr %p, i64 51
  %p17_1 = getelementptr i1, ptr %p, i64 52
  %p17_2 = getelementptr i1, ptr %p, i64 53
  %x17_0 = load i1, ptr %p17_0
  %x17_1 = load i1, ptr %p17_1
  %x17_2 = load i1, ptr %p17_2
  %s17_0 = select i1 %b16, i1 true, i1 %x17_0
  %s17_1 = select i1 %b16, i1 true, i1 %x17_1
  %s17_2 = select i1 %b16, i1 true, i1 %x17_2
  %o17 = or i1 %s17_0, %s17_1
  %b17 = or i1 %o17, %s17_2
  br label %bb18

bb18:
  %p18_0 = getelementptr i1, ptr %p, i64 54
  %p18_1 = getelementptr i1, ptr %p, i64 55
  %p18_2 = getelementptr i1, ptr %p, i64 56
  %x18_0 = load i1, ptr %p18_0
  %x18_1 = load i1, ptr %p18_1
  %x18_2 = load i1, ptr %p18_2
  %s18_0 = select i1 %b17, i1 true, i1 %x18_0
  %s18_1 = select i1 %b17, i1 true, i1 %x18_1
  %s18_2 = select i1 %b17, i1 true, i1 %x18_2
  %o18 = or i1 %s18_0, %s18_1
  %b18 = or i1 %o18, %s18_2
  br label %bb19

bb19:
  %p19_0 = getelementptr i1, ptr %p, i64 57
  %p19_1 = getelementptr i1, ptr %p, i64 58
  %p19_2 = getelementptr i1, ptr %p, i64 59
  %x19_0 = load i1, ptr %p19_0
  %x19_1 = load i1, ptr %p19_1
  %x19_2 = load i1, ptr %p19_2
  %s19_0 = select i1 %b18, i1 true, i1 %x19_0
  %s19_1 = select i1 %b18, i1 true, i1 %x19_1
  %s19_2 = select i1 %b18, i1 true, i1 %x19_2
  %o19 = or i1 %s19_0, %s19_1
  %b19 = or i1 %o19, %s19_2
  br i1 %b19, label %true_block, label %false_block

true_block:
  ret void

false_block:
  ret void
}

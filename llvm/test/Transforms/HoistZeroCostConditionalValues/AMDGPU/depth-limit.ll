; RUN: opt -passes=verify -mtriple=amdgpu9.42 -verify-each -S %s -o %t.reference
; RUN: opt -passes=hoist-zero-cost-conditional-values -mtriple=amdgpu9.42 -verify-each -S %s -o %t.actual
; RUN: diff %t.reference %t.actual
;
; This acyclic chain exceeds the recursive PHI depth limit. The checker finds
; poison inputs before reaching the limit. It must reject the entire candidate
; without hoisting the Else value or filling any of those inputs.

define i64 @deep_phi_chain(i1 %bypass, i32 %flags, i32 %a, i64 %x) {
entry:
  br i1 %bypass, label %flow, label %check0

check0:
  %bit0 = and i32 %flags, 1
  %continue0 = icmp ne i32 %bit0, 0
  br i1 %continue0, label %check1, label %join0

check1:
  %bit1 = and i32 %flags, 2
  %continue1 = icmp ne i32 %bit1, 0
  br i1 %continue1, label %check2, label %join1

check2:
  %bit2 = and i32 %flags, 4
  %continue2 = icmp ne i32 %bit2, 0
  br i1 %continue2, label %check3, label %join2

check3:
  %bit3 = and i32 %flags, 8
  %continue3 = icmp ne i32 %bit3, 0
  br i1 %continue3, label %check4, label %join3

check4:
  %bit4 = and i32 %flags, 16
  %continue4 = icmp ne i32 %bit4, 0
  br i1 %continue4, label %check5, label %join4

check5:
  %bit5 = and i32 %flags, 32
  %continue5 = icmp ne i32 %bit5, 0
  br i1 %continue5, label %check6, label %join5

check6:
  %bit6 = and i32 %flags, 64
  %continue6 = icmp ne i32 %bit6, 0
  br i1 %continue6, label %check7, label %join6

check7:
  %bit7 = and i32 %flags, 128
  %continue7 = icmp ne i32 %bit7, 0
  br i1 %continue7, label %check8, label %join7

check8:
  %bit8 = and i32 %flags, 256
  %continue8 = icmp ne i32 %bit8, 0
  br i1 %continue8, label %check9, label %join8

check9:
  %bit9 = and i32 %flags, 512
  %continue9 = icmp ne i32 %bit9, 0
  br i1 %continue9, label %check10, label %join9

check10:
  %bit10 = and i32 %flags, 1024
  %continue10 = icmp ne i32 %bit10, 0
  br i1 %continue10, label %check11, label %join10

check11:
  %bit11 = and i32 %flags, 2048
  %continue11 = icmp ne i32 %bit11, 0
  br i1 %continue11, label %check12, label %join11

check12:
  %bit12 = and i32 %flags, 4096
  %continue12 = icmp ne i32 %bit12, 0
  br i1 %continue12, label %check13, label %join12

check13:
  %bit13 = and i32 %flags, 8192
  %continue13 = icmp ne i32 %bit13, 0
  br i1 %continue13, label %check14, label %join13

check14:
  %bit14 = and i32 %flags, 16384
  %continue14 = icmp ne i32 %bit14, 0
  br i1 %continue14, label %check15, label %join14

check15:
  %bit15 = and i32 %flags, 32768
  %continue15 = icmp ne i32 %bit15, 0
  br i1 %continue15, label %check16, label %join15

check16:
  %bit16 = and i32 %flags, 65536
  %continue16 = icmp ne i32 %bit16, 0
  br i1 %continue16, label %check17, label %join16

check17:
  %bit17 = and i32 %flags, 131072
  %continue17 = icmp ne i32 %bit17, 0
  br i1 %continue17, label %check18, label %join17

check18:
  %bit18 = and i32 %flags, 262144
  %continue18 = icmp ne i32 %bit18, 0
  br i1 %continue18, label %check19, label %join18

check19:
  %bit19 = and i32 %flags, 524288
  %continue19 = icmp ne i32 %bit19, 0
  br i1 %continue19, label %check20, label %join19

check20:
  %bit20 = and i32 %flags, 1048576
  %continue20 = icmp ne i32 %bit20, 0
  br i1 %continue20, label %check21, label %join20

check21:
  %bit21 = and i32 %flags, 2097152
  %continue21 = icmp ne i32 %bit21, 0
  br i1 %continue21, label %check22, label %join21

check22:
  %bit22 = and i32 %flags, 4194304
  %continue22 = icmp ne i32 %bit22, 0
  br i1 %continue22, label %check23, label %join22

check23:
  %bit23 = and i32 %flags, 8388608
  %continue23 = icmp ne i32 %bit23, 0
  br i1 %continue23, label %check24, label %join23

check24:
  %bit24 = and i32 %flags, 16777216
  %continue24 = icmp ne i32 %bit24, 0
  br i1 %continue24, label %check25, label %join24

check25:
  %bit25 = and i32 %flags, 33554432
  %continue25 = icmp ne i32 %bit25, 0
  br i1 %continue25, label %check26, label %join25

check26:
  %bit26 = and i32 %flags, 67108864
  %continue26 = icmp ne i32 %bit26, 0
  br i1 %continue26, label %check27, label %join26

check27:
  %bit27 = and i32 %flags, 134217728
  %continue27 = icmp ne i32 %bit27, 0
  br i1 %continue27, label %check28, label %join27

check28:
  %bit28 = and i32 %flags, 268435456
  %continue28 = icmp ne i32 %bit28, 0
  br i1 %continue28, label %check29, label %join28

check29:
  %bit29 = and i32 %flags, 536870912
  %continue29 = icmp ne i32 %bit29, 0
  br i1 %continue29, label %check30, label %join29

check30:
  %bit30 = and i32 %flags, 1073741824
  %continue30 = icmp ne i32 %bit30, 0
  br i1 %continue30, label %check31, label %join30

check31:
  %bit31 = and i32 %flags, 2147483648
  %continue31 = icmp ne i32 %bit31, 0
  br i1 %continue31, label %then, label %join31

then:
  %modified = add i64 %x, 1
  br label %join31

join31:
  %v31 = phi i64 [ poison, %check31 ], [ %modified, %then ]
  %c31 = phi i1 [ true, %check31 ], [ false, %then ]
  br label %join30

join30:
  %v30 = phi i64 [ poison, %check30 ], [ %v31, %join31 ]
  %c30 = phi i1 [ true, %check30 ], [ %c31, %join31 ]
  br label %join29

join29:
  %v29 = phi i64 [ poison, %check29 ], [ %v30, %join30 ]
  %c29 = phi i1 [ true, %check29 ], [ %c30, %join30 ]
  br label %join28

join28:
  %v28 = phi i64 [ poison, %check28 ], [ %v29, %join29 ]
  %c28 = phi i1 [ true, %check28 ], [ %c29, %join29 ]
  br label %join27

join27:
  %v27 = phi i64 [ poison, %check27 ], [ %v28, %join28 ]
  %c27 = phi i1 [ true, %check27 ], [ %c28, %join28 ]
  br label %join26

join26:
  %v26 = phi i64 [ poison, %check26 ], [ %v27, %join27 ]
  %c26 = phi i1 [ true, %check26 ], [ %c27, %join27 ]
  br label %join25

join25:
  %v25 = phi i64 [ poison, %check25 ], [ %v26, %join26 ]
  %c25 = phi i1 [ true, %check25 ], [ %c26, %join26 ]
  br label %join24

join24:
  %v24 = phi i64 [ poison, %check24 ], [ %v25, %join25 ]
  %c24 = phi i1 [ true, %check24 ], [ %c25, %join25 ]
  br label %join23

join23:
  %v23 = phi i64 [ poison, %check23 ], [ %v24, %join24 ]
  %c23 = phi i1 [ true, %check23 ], [ %c24, %join24 ]
  br label %join22

join22:
  %v22 = phi i64 [ poison, %check22 ], [ %v23, %join23 ]
  %c22 = phi i1 [ true, %check22 ], [ %c23, %join23 ]
  br label %join21

join21:
  %v21 = phi i64 [ poison, %check21 ], [ %v22, %join22 ]
  %c21 = phi i1 [ true, %check21 ], [ %c22, %join22 ]
  br label %join20

join20:
  %v20 = phi i64 [ poison, %check20 ], [ %v21, %join21 ]
  %c20 = phi i1 [ true, %check20 ], [ %c21, %join21 ]
  br label %join19

join19:
  %v19 = phi i64 [ poison, %check19 ], [ %v20, %join20 ]
  %c19 = phi i1 [ true, %check19 ], [ %c20, %join20 ]
  br label %join18

join18:
  %v18 = phi i64 [ poison, %check18 ], [ %v19, %join19 ]
  %c18 = phi i1 [ true, %check18 ], [ %c19, %join19 ]
  br label %join17

join17:
  %v17 = phi i64 [ poison, %check17 ], [ %v18, %join18 ]
  %c17 = phi i1 [ true, %check17 ], [ %c18, %join18 ]
  br label %join16

join16:
  %v16 = phi i64 [ poison, %check16 ], [ %v17, %join17 ]
  %c16 = phi i1 [ true, %check16 ], [ %c17, %join17 ]
  br label %join15

join15:
  %v15 = phi i64 [ poison, %check15 ], [ %v16, %join16 ]
  %c15 = phi i1 [ true, %check15 ], [ %c16, %join16 ]
  br label %join14

join14:
  %v14 = phi i64 [ poison, %check14 ], [ %v15, %join15 ]
  %c14 = phi i1 [ true, %check14 ], [ %c15, %join15 ]
  br label %join13

join13:
  %v13 = phi i64 [ poison, %check13 ], [ %v14, %join14 ]
  %c13 = phi i1 [ true, %check13 ], [ %c14, %join14 ]
  br label %join12

join12:
  %v12 = phi i64 [ poison, %check12 ], [ %v13, %join13 ]
  %c12 = phi i1 [ true, %check12 ], [ %c13, %join13 ]
  br label %join11

join11:
  %v11 = phi i64 [ poison, %check11 ], [ %v12, %join12 ]
  %c11 = phi i1 [ true, %check11 ], [ %c12, %join12 ]
  br label %join10

join10:
  %v10 = phi i64 [ poison, %check10 ], [ %v11, %join11 ]
  %c10 = phi i1 [ true, %check10 ], [ %c11, %join11 ]
  br label %join9

join9:
  %v9 = phi i64 [ poison, %check9 ], [ %v10, %join10 ]
  %c9 = phi i1 [ true, %check9 ], [ %c10, %join10 ]
  br label %join8

join8:
  %v8 = phi i64 [ poison, %check8 ], [ %v9, %join9 ]
  %c8 = phi i1 [ true, %check8 ], [ %c9, %join9 ]
  br label %join7

join7:
  %v7 = phi i64 [ poison, %check7 ], [ %v8, %join8 ]
  %c7 = phi i1 [ true, %check7 ], [ %c8, %join8 ]
  br label %join6

join6:
  %v6 = phi i64 [ poison, %check6 ], [ %v7, %join7 ]
  %c6 = phi i1 [ true, %check6 ], [ %c7, %join7 ]
  br label %join5

join5:
  %v5 = phi i64 [ poison, %check5 ], [ %v6, %join6 ]
  %c5 = phi i1 [ true, %check5 ], [ %c6, %join6 ]
  br label %join4

join4:
  %v4 = phi i64 [ poison, %check4 ], [ %v5, %join5 ]
  %c4 = phi i1 [ true, %check4 ], [ %c5, %join5 ]
  br label %join3

join3:
  %v3 = phi i64 [ poison, %check3 ], [ %v4, %join4 ]
  %c3 = phi i1 [ true, %check3 ], [ %c4, %join4 ]
  br label %join2

join2:
  %v2 = phi i64 [ poison, %check2 ], [ %v3, %join3 ]
  %c2 = phi i1 [ true, %check2 ], [ %c3, %join3 ]
  br label %join1

join1:
  %v1 = phi i64 [ poison, %check1 ], [ %v2, %join2 ]
  %c1 = phi i1 [ true, %check1 ], [ %c2, %join2 ]
  br label %join0

join0:
  %v0 = phi i64 [ poison, %check0 ], [ %v1, %join1 ]
  %c0 = phi i1 [ true, %check0 ], [ %c1, %join1 ]
  br label %flow

flow:
  %v = phi i64 [ poison, %entry ], [ %v0, %join0 ]
  %take_else = phi i1 [ true, %entry ], [ %c0, %join0 ]
  br i1 %take_else, label %else, label %merge

else:
  %e = zext i32 %a to i64
  br label %merge

merge:
  %r = phi i64 [ %v, %flow ], [ %e, %else ]
  ret i64 %r
}

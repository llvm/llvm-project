; RUN: opt < %s -passes=correlated-propagation

; PR8161
define void @test1() nounwind ssp {
entry:
  br label %for.end

for.cond.us.us:                                   ; preds = %for.cond.us.us
  %cmp6.i.us.us = icmp sgt i32 1, 0
  %lor.ext.i.us.us = zext i1 %cmp6.i.us.us to i32
  %lor.ext.add.i.us.us = select i1 %cmp6.i.us.us, i32 %lor.ext.i.us.us, i32 undef
  %conv.i.us.us = trunc i32 %lor.ext.add.i.us.us to i16
  %sext.us.us = shl i16 %conv.i.us.us, 8
  %conv6.us.us = ashr i16 %sext.us.us, 8
  %and.us.us = and i16 %conv6.us.us, %and.us.us
  br i1 false, label %for.end, label %for.cond.us.us

for.end:                                          ; preds = %for.cond.us, %for.cond.us.us, %entry
  ret void
}

; PR 8790
define void @test2() nounwind ssp {
entry:
  br label %func_29.exit

sdf.exit.i:
  %l_44.1.mux.i = select i1 %tobool5.not.i, i8 %l_44.1.mux.i, i8 1
  br label %srf.exit.i

srf.exit.i:
  %tobool5.not.i = icmp ne i8 undef, 0
  br i1 %tobool5.not.i, label %sdf.exit.i, label %func_29.exit

func_29.exit:
  ret void
}

; PR13972
define void @test3(i1 %arg) nounwind {
for.body:
  br label %return

for.cond.i:                                       ; preds = %if.else.i, %for.body.i
  %e.2.i = phi i32 [ %e.2.i, %if.else.i ], [ -8, %for.body.i ]
  br i1 %arg, label %return, label %for.body.i

for.body.i:                                       ; preds = %for.cond.i
  switch i32 %e.2.i, label %for.cond3.i [
    i32 -3, label %if.else.i
    i32 0, label %for.cond.i
  ]

for.cond3.i:                                      ; preds = %for.cond3.i, %for.body.i
  br label %for.cond3.i

if.else.i:                                        ; preds = %for.body.i
  br label %for.cond.i

return:                                           ; preds = %for.cond.i, %for.body
  ret void
}

define i1 @test4(i32 %int) {
  %a0 = icmp ult i32 %int, 100
  %a1 = and i1 %a0, %a0
  %a2 = and i1 %a1, %a1
  %a3 = and i1 %a2, %a2
  %a4 = and i1 %a3, %a3
  %a5 = and i1 %a4, %a4
  %a6 = and i1 %a5, %a5
  %a7 = and i1 %a6, %a6
  %a8 = and i1 %a7, %a7
  %a9 = and i1 %a8, %a8
  %a10 = and i1 %a9, %a9
  %a11 = and i1 %a10, %a10
  %a12 = and i1 %a11, %a11
  %a13 = and i1 %a12, %a12
  %a14 = and i1 %a13, %a13
  %a15 = and i1 %a14, %a14
  %a16 = and i1 %a15, %a15
  %a17 = and i1 %a16, %a16
  %a18 = and i1 %a17, %a17
  %a19 = and i1 %a18, %a18
  %a20 = and i1 %a19, %a19
  %a21 = and i1 %a20, %a20
  %a22 = and i1 %a21, %a21
  %a23 = and i1 %a22, %a22
  %a24 = and i1 %a23, %a23
  %a25 = and i1 %a24, %a24
  %a26 = and i1 %a25, %a25
  %a27 = and i1 %a26, %a26
  %a28 = and i1 %a27, %a27
  %a29 = and i1 %a28, %a28
  %a30 = and i1 %a29, %a29
  %a31 = and i1 %a30, %a30
  %a32 = and i1 %a31, %a31
  %a33 = and i1 %a32, %a32
  %a34 = and i1 %a33, %a33
  %a35 = and i1 %a34, %a34
  %a36 = and i1 %a35, %a35
  %a37 = and i1 %a36, %a36
  %a38 = and i1 %a37, %a37
  %a39 = and i1 %a38, %a38
  %a40 = and i1 %a39, %a39
  %a41 = and i1 %a40, %a40
  %a42 = and i1 %a41, %a41
  %a43 = and i1 %a42, %a42
  %a44 = and i1 %a43, %a43
  %a45 = and i1 %a44, %a44
  %a46 = and i1 %a45, %a45
  %a47 = and i1 %a46, %a46
  %a48 = and i1 %a47, %a47
  %a49 = and i1 %a48, %a48
  %a50 = and i1 %a49, %a49
  %a51 = and i1 %a50, %a50
  %a52 = and i1 %a51, %a51
  %a53 = and i1 %a52, %a52
  %a54 = and i1 %a53, %a53
  %a55 = and i1 %a54, %a54
  %a56 = and i1 %a55, %a55
  %a57 = and i1 %a56, %a56
  %a58 = and i1 %a57, %a57
  %a59 = and i1 %a58, %a58
  %a60 = and i1 %a59, %a59
  %a61 = and i1 %a60, %a60
  %a62 = and i1 %a61, %a61
  %a63 = and i1 %a62, %a62
  %a64 = and i1 %a63, %a63
  %a65 = and i1 %a64, %a64
  %a66 = and i1 %a65, %a65
  %a67 = and i1 %a66, %a66
  %a68 = and i1 %a67, %a67
  %a69 = and i1 %a68, %a68
  %a70 = and i1 %a69, %a69
  %a71 = and i1 %a70, %a70
  %a72 = and i1 %a71, %a71
  %a73 = and i1 %a72, %a72
  %a74 = and i1 %a73, %a73
  %a75 = and i1 %a74, %a74
  %a76 = and i1 %a75, %a75
  %a77 = and i1 %a76, %a76
  %a78 = and i1 %a77, %a77
  %a79 = and i1 %a78, %a78
  %a80 = and i1 %a79, %a79
  %a81 = and i1 %a80, %a80
  %a82 = and i1 %a81, %a81
  %a83 = and i1 %a82, %a82
  %a84 = and i1 %a83, %a83
  %a85 = and i1 %a84, %a84
  %a86 = and i1 %a85, %a85
  %a87 = and i1 %a86, %a86
  %a88 = and i1 %a87, %a87
  %a89 = and i1 %a88, %a88
  %a90 = and i1 %a89, %a89
  %a91 = and i1 %a90, %a90
  %a92 = and i1 %a91, %a91
  %a93 = and i1 %a92, %a92
  %a94 = and i1 %a93, %a93
  %a95 = and i1 %a94, %a94
  %a96 = and i1 %a95, %a95
  %a97 = and i1 %a96, %a96
  %a98 = and i1 %a97, %a97
  %a99 = and i1 %a98, %a98
  %a100 = and i1 %a99, %a99
  %a101 = and i1 %a100, %a100
  %a102 = and i1 %a101, %a101
  %a103 = and i1 %a102, %a102
  %a104 = and i1 %a103, %a103
  %a105 = and i1 %a104, %a104
  %a106 = and i1 %a105, %a105
  %a107 = and i1 %a106, %a106
  %a108 = and i1 %a107, %a107
  %a109 = and i1 %a108, %a108
  %a110 = and i1 %a109, %a109
  %a111 = and i1 %a110, %a110
  %a112 = and i1 %a111, %a111
  %a113 = and i1 %a112, %a112
  %a114 = and i1 %a113, %a113
  %a115 = and i1 %a114, %a114
  %a116 = and i1 %a115, %a115
  %a117 = and i1 %a116, %a116
  %a118 = and i1 %a117, %a117
  %a119 = and i1 %a118, %a118
  %a120 = and i1 %a119, %a119
  %a121 = and i1 %a120, %a120
  %a122 = and i1 %a121, %a121
  %a123 = and i1 %a122, %a122
  %a124 = and i1 %a123, %a123
  %a125 = and i1 %a124, %a124
  %a126 = and i1 %a125, %a125
  %a127 = and i1 %a126, %a126
  %cond = and i1 %a127, %a127
  br i1 %cond, label %then, label %else

then:
  %result = icmp eq i32 %int, 255
  ret i1 %result

else:
  ret i1 false
}

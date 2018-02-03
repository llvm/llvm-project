; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-scops -analyze < %s | FileCheck %s

; CHECK:      Statements {
; CHECK-NEXT:  	Stmt_bb9
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb9[] };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb9[] -> [0] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb9[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb12
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb12[] : 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb12[] -> [1] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb12[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb15
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb15[] : -268435455 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb15[] -> [2] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb15[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb18
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb18[] : -134217727 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb18[] -> [3] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb18[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb21
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb21[] : -67108863 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb21[] -> [4] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb21[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb24
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb24[] : -33554431 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb24[] -> [5] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb24[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb27
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb27[] : -16777215 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb27[] -> [6] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb27[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb30
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb30[] : -8388607 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb30[] -> [7] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb30[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb33
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb33[] : -4194303 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb33[] -> [8] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb33[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb36
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb36[] : -2097151 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb36[] -> [9] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb36[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb39
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb39[] : -1048575 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb39[] -> [10] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb39[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb42
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb42[] : -524287 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb42[] -> [11] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb42[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb45
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb45[] : -262143 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb45[] -> [12] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb45[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb48
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb48[] : -131071 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb48[] -> [13] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb48[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb51
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb51[] : -65535 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb51[] -> [14] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb51[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb54
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb54[] : -32767 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb54[] -> [15] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb54[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb57
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb57[] : -16383 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb57[] -> [16] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb57[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb60
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb60[] : -8191 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb60[] -> [17] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb60[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb63
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb63[] : -4095 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb63[] -> [18] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb63[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb66
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb66[] : -2047 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb66[] -> [19] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb66[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb69
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb69[] : -1023 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb69[] -> [20] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb69[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb72
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb72[] : -511 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb72[] -> [21] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb72[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb75
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb75[] : -255 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb75[] -> [22] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb75[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb78
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb78[] : -127 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb78[] -> [23] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb78[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb81
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb81[] : -63 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb81[] -> [24] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb81[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb84
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb84[] : -31 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb84[] -> [25] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb84[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb87
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb87[] : -15 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb87[] -> [26] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb87[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb90
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb90[] : -7 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb90[] -> [27] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb90[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb93
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb93[] : -3 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb93[] -> [28] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb93[] -> MemRef_tmp8[] };
; CHECK-NEXT:  	Stmt_bb96
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb96[] : -1 + tmp <= 1073741824*floor((536870912 + tmp)/1073741824) <= tmp };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              [tmp] -> { Stmt_bb96[] -> [29] };
; CHECK-NEXT:          MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              [tmp] -> { Stmt_bb96[] -> MemRef_tmp8[] };
; CHECK-NEXT:  }

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7--linux-android"

define i32 @f(i32* nocapture readonly %arg5) {
bb:
  %tmp = load i32, i32* %arg5, align 4
  %tmp6 = icmp eq i32 %tmp, 0
  br i1 %tmp6, label %bb9, label %bb7

bb7:                                              ; preds = %bb96, %bb93, %bb90, %bb87, %bb84, %bb81, %bb78, %bb75, %bb72, %bb69, %bb66, %bb63, %bb60, %bb57, %bb54, %bb51, %bb48, %bb45, %bb42, %bb39, %bb36, %bb33, %bb30, %bb27, %bb24, %bb21, %bb18, %bb15, %bb12, %bb9, %bb
  %tmp8 = phi i32 [ 30, %bb93 ], [ 29, %bb90 ], [ 28, %bb87 ], [ 27, %bb84 ], [ 26, %bb81 ], [ 25, %bb78 ], [ 24, %bb75 ], [ 23, %bb72 ], [ 22, %bb69 ], [ 21, %bb66 ], [ 20, %bb63 ], [ 19, %bb60 ], [ 18, %bb57 ], [ 17, %bb54 ], [ 16, %bb51 ], [ 15, %bb48 ], [ 14, %bb45 ], [ 13, %bb42 ], [ 12, %bb39 ], [ 11, %bb36 ], [ 10, %bb33 ], [ 9, %bb30 ], [ 8, %bb27 ], [ 7, %bb24 ], [ 6, %bb21 ], [ 5, %bb18 ], [ 4, %bb15 ], [ 3, %bb12 ], [ 2, %bb9 ], [ 1, %bb ], [ %tmp98, %bb96 ]
  ret i32 %tmp8

bb9:                                              ; preds = %bb
  %tmp10 = and i32 %tmp, 536870912
  %tmp11 = icmp eq i32 %tmp10, 0
  br i1 %tmp11, label %bb12, label %bb7

bb12:                                             ; preds = %bb9
  %tmp13 = and i32 %tmp, 268435456
  %tmp14 = icmp eq i32 %tmp13, 0
  br i1 %tmp14, label %bb15, label %bb7

bb15:                                             ; preds = %bb12
  %tmp16 = and i32 %tmp, 134217728
  %tmp17 = icmp eq i32 %tmp16, 0
  br i1 %tmp17, label %bb18, label %bb7

bb18:                                             ; preds = %bb15
  %tmp19 = and i32 %tmp, 67108864
  %tmp20 = icmp eq i32 %tmp19, 0
  br i1 %tmp20, label %bb21, label %bb7

bb21:                                             ; preds = %bb18
  %tmp22 = and i32 %tmp, 33554432
  %tmp23 = icmp eq i32 %tmp22, 0
  br i1 %tmp23, label %bb24, label %bb7

bb24:                                             ; preds = %bb21
  %tmp25 = and i32 %tmp, 16777216
  %tmp26 = icmp eq i32 %tmp25, 0
  br i1 %tmp26, label %bb27, label %bb7

bb27:                                             ; preds = %bb24
  %tmp28 = and i32 %tmp, 8388608
  %tmp29 = icmp eq i32 %tmp28, 0
  br i1 %tmp29, label %bb30, label %bb7

bb30:                                             ; preds = %bb27
  %tmp31 = and i32 %tmp, 4194304
  %tmp32 = icmp eq i32 %tmp31, 0
  br i1 %tmp32, label %bb33, label %bb7

bb33:                                             ; preds = %bb30
  %tmp34 = and i32 %tmp, 2097152
  %tmp35 = icmp eq i32 %tmp34, 0
  br i1 %tmp35, label %bb36, label %bb7

bb36:                                             ; preds = %bb33
  %tmp37 = and i32 %tmp, 1048576
  %tmp38 = icmp eq i32 %tmp37, 0
  br i1 %tmp38, label %bb39, label %bb7

bb39:                                             ; preds = %bb36
  %tmp40 = and i32 %tmp, 524288
  %tmp41 = icmp eq i32 %tmp40, 0
  br i1 %tmp41, label %bb42, label %bb7

bb42:                                             ; preds = %bb39
  %tmp43 = and i32 %tmp, 262144
  %tmp44 = icmp eq i32 %tmp43, 0
  br i1 %tmp44, label %bb45, label %bb7

bb45:                                             ; preds = %bb42
  %tmp46 = and i32 %tmp, 131072
  %tmp47 = icmp eq i32 %tmp46, 0
  br i1 %tmp47, label %bb48, label %bb7

bb48:                                             ; preds = %bb45
  %tmp49 = and i32 %tmp, 65536
  %tmp50 = icmp eq i32 %tmp49, 0
  br i1 %tmp50, label %bb51, label %bb7

bb51:                                             ; preds = %bb48
  %tmp52 = and i32 %tmp, 32768
  %tmp53 = icmp eq i32 %tmp52, 0
  br i1 %tmp53, label %bb54, label %bb7

bb54:                                             ; preds = %bb51
  %tmp55 = and i32 %tmp, 16384
  %tmp56 = icmp eq i32 %tmp55, 0
  br i1 %tmp56, label %bb57, label %bb7

bb57:                                             ; preds = %bb54
  %tmp58 = and i32 %tmp, 8192
  %tmp59 = icmp eq i32 %tmp58, 0
  br i1 %tmp59, label %bb60, label %bb7

bb60:                                             ; preds = %bb57
  %tmp61 = and i32 %tmp, 4096
  %tmp62 = icmp eq i32 %tmp61, 0
  br i1 %tmp62, label %bb63, label %bb7

bb63:                                             ; preds = %bb60
  %tmp64 = and i32 %tmp, 2048
  %tmp65 = icmp eq i32 %tmp64, 0
  br i1 %tmp65, label %bb66, label %bb7

bb66:                                             ; preds = %bb63
  %tmp67 = and i32 %tmp, 1024
  %tmp68 = icmp eq i32 %tmp67, 0
  br i1 %tmp68, label %bb69, label %bb7

bb69:                                             ; preds = %bb66
  %tmp70 = and i32 %tmp, 512
  %tmp71 = icmp eq i32 %tmp70, 0
  br i1 %tmp71, label %bb72, label %bb7

bb72:                                             ; preds = %bb69
  %tmp73 = and i32 %tmp, 256
  %tmp74 = icmp eq i32 %tmp73, 0
  br i1 %tmp74, label %bb75, label %bb7

bb75:                                             ; preds = %bb72
  %tmp76 = and i32 %tmp, 128
  %tmp77 = icmp eq i32 %tmp76, 0
  br i1 %tmp77, label %bb78, label %bb7

bb78:                                             ; preds = %bb75
  %tmp79 = and i32 %tmp, 64
  %tmp80 = icmp eq i32 %tmp79, 0
  br i1 %tmp80, label %bb81, label %bb7

bb81:                                             ; preds = %bb78
  %tmp82 = and i32 %tmp, 32
  %tmp83 = icmp eq i32 %tmp82, 0
  br i1 %tmp83, label %bb84, label %bb7

bb84:                                             ; preds = %bb81
  %tmp85 = and i32 %tmp, 16
  %tmp86 = icmp eq i32 %tmp85, 0
  br i1 %tmp86, label %bb87, label %bb7

bb87:                                             ; preds = %bb84
  %tmp88 = and i32 %tmp, 8
  %tmp89 = icmp eq i32 %tmp88, 0
  br i1 %tmp89, label %bb90, label %bb7

bb90:                                             ; preds = %bb87
  %tmp91 = and i32 %tmp, 4
  %tmp92 = icmp eq i32 %tmp91, 0
  br i1 %tmp92, label %bb93, label %bb7

bb93:                                             ; preds = %bb90
  %tmp94 = and i32 %tmp, 2
  %tmp95 = icmp eq i32 %tmp94, 0
  br i1 %tmp95, label %bb96, label %bb7

bb96:                                             ; preds = %bb93
  %tmp97 = and i32 %tmp, 1
  %tmp98 = sub nsw i32 32, %tmp97
  br label %bb7
}

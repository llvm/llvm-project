; RUN: llc -relocation-model=pic -o /dev/null < %s

; Check that it doesn't crash.
; Crash was in ARMConstantIslands with error "underestimated function size".

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv5e-none-linux-gnueabi"

@.str = external dso_local constant [26 x i8]
@.str.1 = external dso_local constant [29 x i8]
@.str.2 = external dso_local constant [30 x i8]
@.str.3 = external dso_local constant [38 x i8]
@.str.4 = external dso_local constant [24 x i8]
@.str.5 = external dso_local constant [30 x i8]
@.str.6 = external dso_local constant [30 x i8]
@.str.7 = external dso_local constant [29 x i8]
@.str.8 = external dso_local constant [29 x i8]
@.str.9 = external dso_local constant [30 x i8]
@.str.10 = external dso_local constant [30 x i8]
@.str.11 = external dso_local constant [27 x i8]
@.str.12 = external dso_local constant [27 x i8]
@.str.13 = external dso_local constant [26 x i8]
@.str.14 = external dso_local constant [26 x i8]
@.str.15 = external dso_local constant [32 x i8]
@.str.16 = external dso_local constant [32 x i8]
@.str.17 = external dso_local constant [30 x i8]
@.str.18 = external dso_local constant [30 x i8]
@.str.19 = external dso_local constant [33 x i8]
@.str.20 = external dso_local constant [33 x i8]
@.str.21 = external dso_local constant [33 x i8]
@.str.22 = external dso_local constant [33 x i8]
@.str.23 = external dso_local constant [27 x i8]
@.str.24 = external dso_local constant [27 x i8]
@.str.25 = external dso_local constant [29 x i8]
@.str.26 = external dso_local constant [29 x i8]
@.str.27 = external dso_local constant [26 x i8]
@.str.28 = external dso_local constant [26 x i8]
@.str.29 = external dso_local constant [26 x i8]
@.str.30 = external dso_local constant [27 x i8]
@.str.31 = external dso_local constant [27 x i8]
@.str.32 = external dso_local constant [31 x i8]
@.str.33 = external dso_local constant [31 x i8]
@.str.34 = external dso_local constant [28 x i8]
@.str.35 = external dso_local constant [28 x i8]
@.str.37 = external dso_local constant [29 x i8]
@.str.38 = external dso_local constant [29 x i8]
@.str.39 = external dso_local constant [30 x i8]
@.str.40 = external dso_local constant [30 x i8]
@.str.41 = external dso_local constant [34 x i8]
@.str.42 = external dso_local constant [34 x i8]
@.str.43 = external dso_local constant [34 x i8]
@.str.44 = external dso_local constant [34 x i8]
@.str.45 = external dso_local constant [35 x i8]
@.str.46 = external dso_local constant [35 x i8]
@.str.47 = external dso_local constant [27 x i8]
@.str.48 = external dso_local constant [27 x i8]
@.str.49 = external dso_local constant [26 x i8]
@.str.50 = external dso_local constant [26 x i8]
@.str.51 = external dso_local constant [32 x i8]
@.str.52 = external dso_local constant [32 x i8]
@.str.53 = external dso_local constant [33 x i8]
@.str.54 = external dso_local constant [33 x i8]
@.str.55 = external dso_local constant [40 x i8]
@.str.56 = external dso_local constant [40 x i8]
@.str.57 = external dso_local constant [35 x i8]
@.str.58 = external dso_local constant [35 x i8]
@.str.59 = external dso_local constant [25 x i8]
@.str.60 = external dso_local constant [25 x i8]
@.str.61 = external dso_local constant [33 x i8]
@.str.65 = external dso_local constant [34 x i8]
@.str.66 = external dso_local constant [34 x i8]
@.str.67 = external dso_local constant [33 x i8]
@.str.68 = external dso_local constant [33 x i8]
@.str.69 = external dso_local constant [33 x i8]
@.str.70 = external dso_local constant [33 x i8]
@.str.71 = external dso_local constant [23 x i8]
@.str.72 = external dso_local constant [23 x i8]
@.str.73 = external dso_local constant [32 x i8]
@.str.74 = external dso_local constant [32 x i8]
@.str.75 = external dso_local constant [22 x i8]
@.str.79 = external dso_local constant [26 x i8]
@.str.80 = external dso_local constant [26 x i8]
@.str.81 = external dso_local constant [28 x i8]
@.str.82 = external dso_local constant [28 x i8]
@.str.83 = external dso_local constant [28 x i8]
@.str.84 = external dso_local constant [27 x i8]
@.str.85 = external dso_local constant [27 x i8]
@.str.86 = external dso_local constant [26 x i8]
@.str.87 = external dso_local constant [26 x i8]
@.str.88 = external dso_local constant [27 x i8]
@.str.89 = external dso_local constant [27 x i8]
@.str.90 = external dso_local constant [34 x i8]
@.str.91 = external dso_local constant [34 x i8]
@.str.92 = external dso_local constant [29 x i8]
@.str.93 = external dso_local constant [29 x i8]
@.str.94 = external dso_local constant [26 x i8]
@.str.95 = external dso_local constant [26 x i8]
@.str.96 = external dso_local constant [32 x i8]
@.str.97 = external dso_local constant [32 x i8]
@.str.98 = external dso_local constant [33 x i8]
@.str.99 = external dso_local constant [33 x i8]
@.str.111 = external dso_local constant [35 x i8]
@.str.112 = external dso_local constant [14 x i8]

; Function Attrs: optsize
define ptr @f(ptr %s) #0 {
entry:
  %0 = load i32, ptr %s, align 4
  switch i32 %0, label %sw.default [
    i32 16384, label %sw.epilog
    i32 8192, label %sw.bb1
    i32 4096, label %sw.bb2
    i32 3, label %sw.bb3
    i32 12292, label %sw.bb4
    i32 20480, label %sw.bb5
    i32 4099, label %sw.bb6
    i32 24576, label %sw.bb7
    i32 8195, label %sw.bb8
    i32 4224, label %sw.bb9
    i32 8320, label %sw.bb10
    i32 4112, label %sw.bb11
    i32 4113, label %sw.bb12
    i32 4128, label %sw.bb13
    i32 4129, label %sw.bb14
    i32 4144, label %sw.bb15
    i32 4145, label %sw.bb16
    i32 4160, label %sw.bb17
    i32 4161, label %sw.bb18
    i32 4176, label %sw.bb19
    i32 4177, label %sw.bb20
    i32 4178, label %sw.bb21
    i32 4179, label %sw.bb22
    i32 4192, label %sw.bb23
    i32 4193, label %sw.bb24
    i32 4208, label %sw.bb25
    i32 4209, label %sw.bb26
    i32 8208, label %sw.bb27
    i32 8209, label %sw.bb28
    i32 8210, label %sw.bb29
    i32 8224, label %sw.bb30
    i32 8225, label %sw.bb31
    i32 8240, label %sw.bb32
    i32 8241, label %sw.bb33
    i32 8256, label %sw.bb34
    i32 8257, label %sw.bb35
    i32 8468, label %sw.bb111
    i32 8272, label %sw.bb37
    i32 8273, label %sw.bb38
    i32 8288, label %sw.bb39
    i32 8289, label %sw.bb40
    i32 8304, label %sw.bb41
    i32 8305, label %sw.bb42
    i32 8306, label %sw.bb43
    i32 8307, label %sw.bb44
    i32 8336, label %sw.bb45
    i32 4240, label %sw.bb46
    i32 4368, label %sw.bb47
    i32 4369, label %sw.bb48
    i32 4384, label %sw.bb49
    i32 4385, label %sw.bb50
    i32 4400, label %sw.bb51
    i32 4401, label %sw.bb52
    i32 4416, label %sw.bb53
    i32 4417, label %sw.bb54
    i32 4432, label %sw.bb55
    i32 4433, label %sw.bb56
    i32 4576, label %sw.bb57
    i32 4577, label %sw.bb58
    i32 4448, label %sw.bb59
    i32 4449, label %sw.bb60
    i32 4464, label %sw.bb61
    i32 8593, label %sw.bb99
    i32 8592, label %sw.bb98
    i32 8577, label %sw.bb97
    i32 4480, label %sw.bb65
    i32 4481, label %sw.bb66
    i32 4496, label %sw.bb67
    i32 4497, label %sw.bb68
    i32 4512, label %sw.bb69
    i32 8656, label %sw.bb69
    i32 4513, label %sw.bb70
    i32 8657, label %sw.bb70
    i32 4528, label %sw.bb71
    i32 8672, label %sw.bb71
    i32 4529, label %sw.bb72
    i32 8673, label %sw.bb72
    i32 4544, label %sw.bb73
    i32 8624, label %sw.bb73
    i32 4545, label %sw.bb74
    i32 8625, label %sw.bb74
    i32 4560, label %sw.bb75
    i32 8640, label %sw.bb75
    i32 8576, label %sw.bb96
    i32 8561, label %sw.bb95
    i32 8560, label %sw.bb94
    i32 8689, label %sw.bb93
    i32 8688, label %sw.bb92
    i32 8465, label %sw.bb79
    i32 8466, label %sw.bb80
    i32 8480, label %sw.bb81
    i32 8481, label %sw.bb82
    i32 8482, label %sw.bb83
    i32 8496, label %sw.bb84
    i32 8497, label %sw.bb85
    i32 8512, label %sw.bb86
    i32 8513, label %sw.bb87
    i32 8528, label %sw.bb88
    i32 8529, label %sw.bb89
    i32 8544, label %sw.bb90
    i32 8545, label %sw.bb91
  ]

sw.bb1:                                           ; preds = %entry
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  br label %sw.epilog

sw.bb4:                                           ; preds = %entry
  br label %sw.epilog

sw.bb5:                                           ; preds = %entry
  br label %sw.epilog

sw.bb6:                                           ; preds = %entry
  br label %sw.epilog

sw.bb7:                                           ; preds = %entry
  br label %sw.epilog

sw.bb8:                                           ; preds = %entry
  br label %sw.epilog

sw.bb9:                                           ; preds = %entry
  br label %sw.epilog

sw.bb10:                                          ; preds = %entry
  br label %sw.epilog

sw.bb11:                                          ; preds = %entry
  br label %sw.epilog

sw.bb12:                                          ; preds = %entry
  br label %sw.epilog

sw.bb13:                                          ; preds = %entry
  br label %sw.epilog

sw.bb14:                                          ; preds = %entry
  br label %sw.epilog

sw.bb15:                                          ; preds = %entry
  br label %sw.epilog

sw.bb16:                                          ; preds = %entry
  br label %sw.epilog

sw.bb17:                                          ; preds = %entry
  br label %sw.epilog

sw.bb18:                                          ; preds = %entry
  br label %sw.epilog

sw.bb19:                                          ; preds = %entry
  br label %sw.epilog

sw.bb20:                                          ; preds = %entry
  br label %sw.epilog

sw.bb21:                                          ; preds = %entry
  br label %sw.epilog

sw.bb22:                                          ; preds = %entry
  br label %sw.epilog

sw.bb23:                                          ; preds = %entry
  br label %sw.epilog

sw.bb24:                                          ; preds = %entry
  br label %sw.epilog

sw.bb25:                                          ; preds = %entry
  br label %sw.epilog

sw.bb26:                                          ; preds = %entry
  br label %sw.epilog

sw.bb27:                                          ; preds = %entry
  br label %sw.epilog

sw.bb28:                                          ; preds = %entry
  br label %sw.epilog

sw.bb29:                                          ; preds = %entry
  br label %sw.epilog

sw.bb30:                                          ; preds = %entry
  br label %sw.epilog

sw.bb31:                                          ; preds = %entry
  br label %sw.epilog

sw.bb32:                                          ; preds = %entry
  br label %sw.epilog

sw.bb33:                                          ; preds = %entry
  br label %sw.epilog

sw.bb34:                                          ; preds = %entry
  br label %sw.epilog

sw.bb35:                                          ; preds = %entry
  br label %sw.epilog

sw.bb37:                                          ; preds = %entry
  br label %sw.epilog

sw.bb38:                                          ; preds = %entry
  br label %sw.epilog

sw.bb39:                                          ; preds = %entry
  br label %sw.epilog

sw.bb40:                                          ; preds = %entry
  br label %sw.epilog

sw.bb41:                                          ; preds = %entry
  br label %sw.epilog

sw.bb42:                                          ; preds = %entry
  br label %sw.epilog

sw.bb43:                                          ; preds = %entry
  br label %sw.epilog

sw.bb44:                                          ; preds = %entry
  br label %sw.epilog

sw.bb45:                                          ; preds = %entry
  br label %sw.epilog

sw.bb46:                                          ; preds = %entry
  br label %sw.epilog

sw.bb47:                                          ; preds = %entry
  br label %sw.epilog

sw.bb48:                                          ; preds = %entry
  br label %sw.epilog

sw.bb49:                                          ; preds = %entry
  br label %sw.epilog

sw.bb50:                                          ; preds = %entry
  br label %sw.epilog

sw.bb51:                                          ; preds = %entry
  br label %sw.epilog

sw.bb52:                                          ; preds = %entry
  br label %sw.epilog

sw.bb53:                                          ; preds = %entry
  br label %sw.epilog

sw.bb54:                                          ; preds = %entry
  br label %sw.epilog

sw.bb55:                                          ; preds = %entry
  br label %sw.epilog

sw.bb56:                                          ; preds = %entry
  br label %sw.epilog

sw.bb57:                                          ; preds = %entry
  br label %sw.epilog

sw.bb58:                                          ; preds = %entry
  br label %sw.epilog

sw.bb59:                                          ; preds = %entry
  br label %sw.epilog

sw.bb60:                                          ; preds = %entry
  br label %sw.epilog

sw.bb61:                                          ; preds = %entry
  br label %sw.epilog

sw.bb65:                                          ; preds = %entry
  br label %sw.epilog

sw.bb66:                                          ; preds = %entry
  br label %sw.epilog

sw.bb67:                                          ; preds = %entry
  br label %sw.epilog

sw.bb68:                                          ; preds = %entry
  br label %sw.epilog

sw.bb69:                                          ; preds = %entry, %entry
  br label %sw.epilog

sw.bb70:                                          ; preds = %entry, %entry
  br label %sw.epilog

sw.bb71:                                          ; preds = %entry, %entry
  br label %sw.epilog

sw.bb72:                                          ; preds = %entry, %entry
  br label %sw.epilog

sw.bb73:                                          ; preds = %entry, %entry
  br label %sw.epilog

sw.bb74:                                          ; preds = %entry, %entry
  br label %sw.epilog

sw.bb75:                                          ; preds = %entry, %entry
  br label %sw.epilog

sw.bb79:                                          ; preds = %entry
  br label %sw.epilog

sw.bb80:                                          ; preds = %entry
  br label %sw.epilog

sw.bb81:                                          ; preds = %entry
  br label %sw.epilog

sw.bb82:                                          ; preds = %entry
  br label %sw.epilog

sw.bb83:                                          ; preds = %entry
  br label %sw.epilog

sw.bb84:                                          ; preds = %entry
  br label %sw.epilog

sw.bb85:                                          ; preds = %entry
  br label %sw.epilog

sw.bb86:                                          ; preds = %entry
  br label %sw.epilog

sw.bb87:                                          ; preds = %entry
  br label %sw.epilog

sw.bb88:                                          ; preds = %entry
  br label %sw.epilog

sw.bb89:                                          ; preds = %entry
  br label %sw.epilog

sw.bb90:                                          ; preds = %entry
  br label %sw.epilog

sw.bb91:                                          ; preds = %entry
  br label %sw.epilog

sw.bb92:                                          ; preds = %entry
  br label %sw.epilog

sw.bb93:                                          ; preds = %entry
  br label %sw.epilog

sw.bb94:                                          ; preds = %entry
  br label %sw.epilog

sw.bb95:                                          ; preds = %entry
  br label %sw.epilog

sw.bb96:                                          ; preds = %entry
  br label %sw.epilog

sw.bb97:                                          ; preds = %entry
  br label %sw.epilog

sw.bb98:                                          ; preds = %entry
  br label %sw.epilog

sw.bb99:                                          ; preds = %entry
  br label %sw.epilog

sw.bb111:                                         ; preds = %entry
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sw.bb111, %sw.bb99, %sw.bb98, %sw.bb97, %sw.bb96, %sw.bb95, %sw.bb94, %sw.bb93, %sw.bb92, %sw.bb91, %sw.bb90, %sw.bb89, %sw.bb88, %sw.bb87, %sw.bb86, %sw.bb85, %sw.bb84, %sw.bb83, %sw.bb82, %sw.bb81, %sw.bb80, %sw.bb79, %sw.bb75, %sw.bb74, %sw.bb73, %sw.bb72, %sw.bb71, %sw.bb70, %sw.bb69, %sw.bb68, %sw.bb67, %sw.bb66, %sw.bb65, %sw.bb61, %sw.bb60, %sw.bb59, %sw.bb58, %sw.bb57, %sw.bb56, %sw.bb55, %sw.bb54, %sw.bb53, %sw.bb52, %sw.bb51, %sw.bb50, %sw.bb49, %sw.bb48, %sw.bb47, %sw.bb46, %sw.bb45, %sw.bb44, %sw.bb43, %sw.bb42, %sw.bb41, %sw.bb40, %sw.bb39, %sw.bb38, %sw.bb37, %sw.bb35, %sw.bb34, %sw.bb33, %sw.bb32, %sw.bb31, %sw.bb30, %sw.bb29, %sw.bb28, %sw.bb27, %sw.bb26, %sw.bb25, %sw.bb24, %sw.bb23, %sw.bb22, %sw.bb21, %sw.bb20, %sw.bb19, %sw.bb18, %sw.bb17, %sw.bb16, %sw.bb15, %sw.bb14, %sw.bb13, %sw.bb12, %sw.bb11, %sw.bb10, %sw.bb9, %sw.bb8, %sw.bb7, %sw.bb6, %sw.bb5, %sw.bb4, %sw.bb3, %sw.bb2, %sw.bb1, %entry
  %str.0 = phi ptr [ @.str.112, %sw.default ], [ @.str.111, %sw.bb111 ], [ @.str.1, %sw.bb1 ], [ @.str.2, %sw.bb2 ], [ @.str.3, %sw.bb3 ], [ @.str.4, %sw.bb4 ], [ @.str.5, %sw.bb5 ], [ @.str.6, %sw.bb6 ], [ @.str.7, %sw.bb7 ], [ @.str.8, %sw.bb8 ], [ @.str.9, %sw.bb9 ], [ @.str.10, %sw.bb10 ], [ @.str.11, %sw.bb11 ], [ @.str.12, %sw.bb12 ], [ @.str.13, %sw.bb13 ], [ @.str.14, %sw.bb14 ], [ @.str.15, %sw.bb15 ], [ @.str.16, %sw.bb16 ], [ @.str.17, %sw.bb17 ], [ @.str.18, %sw.bb18 ], [ @.str.19, %sw.bb19 ], [ @.str.20, %sw.bb20 ], [ @.str.21, %sw.bb21 ], [ @.str.22, %sw.bb22 ], [ @.str.23, %sw.bb23 ], [ @.str.24, %sw.bb24 ], [ @.str.25, %sw.bb25 ], [ @.str.26, %sw.bb26 ], [ @.str.27, %sw.bb27 ], [ @.str.28, %sw.bb28 ], [ @.str.29, %sw.bb29 ], [ @.str.30, %sw.bb30 ], [ @.str.31, %sw.bb31 ], [ @.str.32, %sw.bb32 ], [ @.str.33, %sw.bb33 ], [ @.str.34, %sw.bb34 ], [ @.str.35, %sw.bb35 ], [ @.str, %entry ], [ @.str.37, %sw.bb37 ], [ @.str.38, %sw.bb38 ], [ @.str.39, %sw.bb39 ], [ @.str.40, %sw.bb40 ], [ @.str.41, %sw.bb41 ], [ @.str.42, %sw.bb42 ], [ @.str.43, %sw.bb43 ], [ @.str.44, %sw.bb44 ], [ @.str.45, %sw.bb45 ], [ @.str.46, %sw.bb46 ], [ @.str.47, %sw.bb47 ], [ @.str.48, %sw.bb48 ], [ @.str.49, %sw.bb49 ], [ @.str.50, %sw.bb50 ], [ @.str.51, %sw.bb51 ], [ @.str.52, %sw.bb52 ], [ @.str.53, %sw.bb53 ], [ @.str.54, %sw.bb54 ], [ @.str.55, %sw.bb55 ], [ @.str.56, %sw.bb56 ], [ @.str.57, %sw.bb57 ], [ @.str.58, %sw.bb58 ], [ @.str.59, %sw.bb59 ], [ @.str.60, %sw.bb60 ], [ @.str.61, %sw.bb61 ], [ @.str.94, %sw.bb94 ], [ @.str.95, %sw.bb95 ], [ @.str.96, %sw.bb96 ], [ @.str.65, %sw.bb65 ], [ @.str.66, %sw.bb66 ], [ @.str.67, %sw.bb67 ], [ @.str.68, %sw.bb68 ], [ @.str.69, %sw.bb69 ], [ @.str.70, %sw.bb70 ], [ @.str.71, %sw.bb71 ], [ @.str.72, %sw.bb72 ], [ @.str.73, %sw.bb73 ], [ @.str.74, %sw.bb74 ], [ @.str.75, %sw.bb75 ], [ @.str.97, %sw.bb97 ], [ @.str.98, %sw.bb98 ], [ @.str.99, %sw.bb99 ], [ @.str.79, %sw.bb79 ], [ @.str.80, %sw.bb80 ], [ @.str.81, %sw.bb81 ], [ @.str.82, %sw.bb82 ], [ @.str.83, %sw.bb83 ], [ @.str.84, %sw.bb84 ], [ @.str.85, %sw.bb85 ], [ @.str.86, %sw.bb86 ], [ @.str.87, %sw.bb87 ], [ @.str.88, %sw.bb88 ], [ @.str.89, %sw.bb89 ], [ @.str.90, %sw.bb90 ], [ @.str.91, %sw.bb91 ], [ @.str.92, %sw.bb92 ], [ @.str.93, %sw.bb93 ]
  ret ptr %str.0
}

attributes #0 = { optsize }

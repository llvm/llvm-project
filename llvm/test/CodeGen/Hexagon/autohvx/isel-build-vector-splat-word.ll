; RUN: llc -mtriple=hexagon < %s | FileCheck %s

target triple = "hexagon"

; buildHvxVectorReg packs i8 elements into i32 words before initializing
; an HVX vector from the most frequent word. That initialization must
; use a word splat even though the final vector type is i8; a byte
; splat would broadcast only the low byte of %x.
define void @splat_i32_word_into_i8_vector(ptr %out, i32 %x, i8 %y) #0 {
; CHECK-LABEL: splat_i32_word_into_i8_vector:
; CHECK: v{{[0-9]+}} = vsplat(r{{[0-9]+}})
; CHECK-NOT: .b = vsplat
; CHECK: vmem
entry:
  %x0 = trunc i32 %x to i8
  %s1 = lshr i32 %x, 8
  %x1 = trunc i32 %s1 to i8
  %s2 = lshr i32 %x, 16
  %x2 = trunc i32 %s2 to i8
  %s3 = lshr i32 %x, 24
  %x3 = trunc i32 %s3 to i8

  %v000 = insertelement <128 x i8> poison, i8 %x0, i32 0
  %v001 = insertelement <128 x i8> %v000, i8 %x1, i32 1
  %v002 = insertelement <128 x i8> %v001, i8 %x2, i32 2
  %v003 = insertelement <128 x i8> %v002, i8 %x3, i32 3
  %v004 = insertelement <128 x i8> %v003, i8 %x0, i32 4
  %v005 = insertelement <128 x i8> %v004, i8 %x1, i32 5
  %v006 = insertelement <128 x i8> %v005, i8 %x2, i32 6
  %v007 = insertelement <128 x i8> %v006, i8 %x3, i32 7
  %v008 = insertelement <128 x i8> %v007, i8 %x0, i32 8
  %v009 = insertelement <128 x i8> %v008, i8 %x1, i32 9
  %v010 = insertelement <128 x i8> %v009, i8 %x2, i32 10
  %v011 = insertelement <128 x i8> %v010, i8 %x3, i32 11
  %v012 = insertelement <128 x i8> %v011, i8 %x0, i32 12
  %v013 = insertelement <128 x i8> %v012, i8 %x1, i32 13
  %v014 = insertelement <128 x i8> %v013, i8 %x2, i32 14
  %v015 = insertelement <128 x i8> %v014, i8 %x3, i32 15
  %v016 = insertelement <128 x i8> %v015, i8 %x0, i32 16
  %v017 = insertelement <128 x i8> %v016, i8 %x1, i32 17
  %v018 = insertelement <128 x i8> %v017, i8 %x2, i32 18
  %v019 = insertelement <128 x i8> %v018, i8 %x3, i32 19
  %v020 = insertelement <128 x i8> %v019, i8 %x0, i32 20
  %v021 = insertelement <128 x i8> %v020, i8 %x1, i32 21
  %v022 = insertelement <128 x i8> %v021, i8 %x2, i32 22
  %v023 = insertelement <128 x i8> %v022, i8 %x3, i32 23
  %v024 = insertelement <128 x i8> %v023, i8 %x0, i32 24
  %v025 = insertelement <128 x i8> %v024, i8 %x1, i32 25
  %v026 = insertelement <128 x i8> %v025, i8 %x2, i32 26
  %v027 = insertelement <128 x i8> %v026, i8 %x3, i32 27
  %v028 = insertelement <128 x i8> %v027, i8 %x0, i32 28
  %v029 = insertelement <128 x i8> %v028, i8 %x1, i32 29
  %v030 = insertelement <128 x i8> %v029, i8 %x2, i32 30
  %v031 = insertelement <128 x i8> %v030, i8 %x3, i32 31
  %v032 = insertelement <128 x i8> %v031, i8 %x0, i32 32
  %v033 = insertelement <128 x i8> %v032, i8 %x1, i32 33
  %v034 = insertelement <128 x i8> %v033, i8 %x2, i32 34
  %v035 = insertelement <128 x i8> %v034, i8 %x3, i32 35
  %v036 = insertelement <128 x i8> %v035, i8 %x0, i32 36
  %v037 = insertelement <128 x i8> %v036, i8 %x1, i32 37
  %v038 = insertelement <128 x i8> %v037, i8 %x2, i32 38
  %v039 = insertelement <128 x i8> %v038, i8 %x3, i32 39
  %v040 = insertelement <128 x i8> %v039, i8 %x0, i32 40
  %v041 = insertelement <128 x i8> %v040, i8 %x1, i32 41
  %v042 = insertelement <128 x i8> %v041, i8 %x2, i32 42
  %v043 = insertelement <128 x i8> %v042, i8 %x3, i32 43
  %v044 = insertelement <128 x i8> %v043, i8 %x0, i32 44
  %v045 = insertelement <128 x i8> %v044, i8 %x1, i32 45
  %v046 = insertelement <128 x i8> %v045, i8 %x2, i32 46
  %v047 = insertelement <128 x i8> %v046, i8 %x3, i32 47
  %v048 = insertelement <128 x i8> %v047, i8 %x0, i32 48
  %v049 = insertelement <128 x i8> %v048, i8 %x1, i32 49
  %v050 = insertelement <128 x i8> %v049, i8 %x2, i32 50
  %v051 = insertelement <128 x i8> %v050, i8 %x3, i32 51
  %v052 = insertelement <128 x i8> %v051, i8 %x0, i32 52
  %v053 = insertelement <128 x i8> %v052, i8 %x1, i32 53
  %v054 = insertelement <128 x i8> %v053, i8 %x2, i32 54
  %v055 = insertelement <128 x i8> %v054, i8 %x3, i32 55
  %v056 = insertelement <128 x i8> %v055, i8 %x0, i32 56
  %v057 = insertelement <128 x i8> %v056, i8 %x1, i32 57
  %v058 = insertelement <128 x i8> %v057, i8 %x2, i32 58
  %v059 = insertelement <128 x i8> %v058, i8 %x3, i32 59
  %v060 = insertelement <128 x i8> %v059, i8 %x0, i32 60
  %v061 = insertelement <128 x i8> %v060, i8 %x1, i32 61
  %v062 = insertelement <128 x i8> %v061, i8 %x2, i32 62
  %v063 = insertelement <128 x i8> %v062, i8 %x3, i32 63
  %v064 = insertelement <128 x i8> %v063, i8 %y, i32 64
  %v065 = insertelement <128 x i8> %v064, i8 %x1, i32 65
  %v066 = insertelement <128 x i8> %v065, i8 %x2, i32 66
  %v067 = insertelement <128 x i8> %v066, i8 %x3, i32 67
  %v068 = insertelement <128 x i8> %v067, i8 %x0, i32 68
  %v069 = insertelement <128 x i8> %v068, i8 %y, i32 69
  %v070 = insertelement <128 x i8> %v069, i8 %x2, i32 70
  %v071 = insertelement <128 x i8> %v070, i8 %x3, i32 71
  %v072 = insertelement <128 x i8> %v071, i8 %x0, i32 72
  %v073 = insertelement <128 x i8> %v072, i8 %x1, i32 73
  %v074 = insertelement <128 x i8> %v073, i8 %y, i32 74
  %v075 = insertelement <128 x i8> %v074, i8 %x3, i32 75
  %v076 = insertelement <128 x i8> %v075, i8 %x0, i32 76
  %v077 = insertelement <128 x i8> %v076, i8 %x1, i32 77
  %v078 = insertelement <128 x i8> %v077, i8 %x2, i32 78
  %v079 = insertelement <128 x i8> %v078, i8 %y, i32 79
  %v080 = insertelement <128 x i8> %v079, i8 %x0, i32 80
  %v081 = insertelement <128 x i8> %v080, i8 %x1, i32 81
  %v082 = insertelement <128 x i8> %v081, i8 %x2, i32 82
  %v083 = insertelement <128 x i8> %v082, i8 %x3, i32 83
  %v084 = insertelement <128 x i8> %v083, i8 %x0, i32 84
  %v085 = insertelement <128 x i8> %v084, i8 %x1, i32 85
  %v086 = insertelement <128 x i8> %v085, i8 %x2, i32 86
  %v087 = insertelement <128 x i8> %v086, i8 %x3, i32 87
  %v088 = insertelement <128 x i8> %v087, i8 %x0, i32 88
  %v089 = insertelement <128 x i8> %v088, i8 %x1, i32 89
  %v090 = insertelement <128 x i8> %v089, i8 %x2, i32 90
  %v091 = insertelement <128 x i8> %v090, i8 %x3, i32 91
  %v092 = insertelement <128 x i8> %v091, i8 %x0, i32 92
  %v093 = insertelement <128 x i8> %v092, i8 %x1, i32 93
  %v094 = insertelement <128 x i8> %v093, i8 %x2, i32 94
  %v095 = insertelement <128 x i8> %v094, i8 %x3, i32 95
  %v096 = insertelement <128 x i8> %v095, i8 %x0, i32 96
  %v097 = insertelement <128 x i8> %v096, i8 %x1, i32 97
  %v098 = insertelement <128 x i8> %v097, i8 %x2, i32 98
  %v099 = insertelement <128 x i8> %v098, i8 %x3, i32 99
  %v100 = insertelement <128 x i8> %v099, i8 %x0, i32 100
  %v101 = insertelement <128 x i8> %v100, i8 %x1, i32 101
  %v102 = insertelement <128 x i8> %v101, i8 %x2, i32 102
  %v103 = insertelement <128 x i8> %v102, i8 %x3, i32 103
  %v104 = insertelement <128 x i8> %v103, i8 %x0, i32 104
  %v105 = insertelement <128 x i8> %v104, i8 %x1, i32 105
  %v106 = insertelement <128 x i8> %v105, i8 %x2, i32 106
  %v107 = insertelement <128 x i8> %v106, i8 %x3, i32 107
  %v108 = insertelement <128 x i8> %v107, i8 %x0, i32 108
  %v109 = insertelement <128 x i8> %v108, i8 %x1, i32 109
  %v110 = insertelement <128 x i8> %v109, i8 %x2, i32 110
  %v111 = insertelement <128 x i8> %v110, i8 %x3, i32 111
  %v112 = insertelement <128 x i8> %v111, i8 %x0, i32 112
  %v113 = insertelement <128 x i8> %v112, i8 %x1, i32 113
  %v114 = insertelement <128 x i8> %v113, i8 %x2, i32 114
  %v115 = insertelement <128 x i8> %v114, i8 %x3, i32 115
  %v116 = insertelement <128 x i8> %v115, i8 %x0, i32 116
  %v117 = insertelement <128 x i8> %v116, i8 %x1, i32 117
  %v118 = insertelement <128 x i8> %v117, i8 %x2, i32 118
  %v119 = insertelement <128 x i8> %v118, i8 %x3, i32 119
  %v120 = insertelement <128 x i8> %v119, i8 %x0, i32 120
  %v121 = insertelement <128 x i8> %v120, i8 %x1, i32 121
  %v122 = insertelement <128 x i8> %v121, i8 %x2, i32 122
  %v123 = insertelement <128 x i8> %v122, i8 %x3, i32 123
  %v124 = insertelement <128 x i8> %v123, i8 %x0, i32 124
  %v125 = insertelement <128 x i8> %v124, i8 %x1, i32 125
  %v126 = insertelement <128 x i8> %v125, i8 %x2, i32 126
  %v127 = insertelement <128 x i8> %v126, i8 %x3, i32 127
  store <128 x i8> %v127, ptr %out, align 128
  ret void
}

define void @splat_i8_into_i8_vector(ptr %out, i8 %x) #0 {
; CHECK-LABEL: splat_i8_into_i8_vector:
; CHECK: v{{[0-9]+}}.b = vsplat(r{{[0-9]+}})
; CHECK: vmem
entry:
  %v0 = insertelement <128 x i8> poison, i8 %x, i32 0
  %v1 = shufflevector <128 x i8> %v0, <128 x i8> poison, <128 x i32> zeroinitializer
  store <128 x i8> %v1, ptr %out, align 128
  ret void
}

; The same word-splat requirement applies when the final vector type is
; i16 and the repeated value is a packed i32 containing two halfwords.
define void @splat_i32_word_into_i16_vector(ptr %out, i32 %x, i16 %y) #0 {
; CHECK-LABEL: splat_i32_word_into_i16_vector:
; CHECK: v{{[0-9]+}} = vsplat(r{{[0-9]+}})
; CHECK-NOT: .h = vsplat
; CHECK: vmem
entry:
  %x0 = trunc i32 %x to i16
  %s1 = lshr i32 %x, 16
  %x1 = trunc i32 %s1 to i16
  %v0 = insertelement <64 x i16> poison, i16 %x0, i32 0
  %v1 = insertelement <64 x i16> %v0, i16 %x1, i32 1
  %v2 = insertelement <64 x i16> %v1, i16 %x0, i32 2
  %v3 = insertelement <64 x i16> %v2, i16 %x1, i32 3
  %v4 = insertelement <64 x i16> %v3, i16 %x0, i32 4
  %v5 = insertelement <64 x i16> %v4, i16 %x1, i32 5
  %v6 = insertelement <64 x i16> %v5, i16 %x0, i32 6
  %v7 = insertelement <64 x i16> %v6, i16 %x1, i32 7
  %v8 = insertelement <64 x i16> %v7, i16 %x0, i32 8
  %v9 = insertelement <64 x i16> %v8, i16 %x1, i32 9
  %v10 = insertelement <64 x i16> %v9, i16 %x0, i32 10
  %v11 = insertelement <64 x i16> %v10, i16 %x1, i32 11
  %v12 = insertelement <64 x i16> %v11, i16 %x0, i32 12
  %v13 = insertelement <64 x i16> %v12, i16 %x1, i32 13
  %v14 = insertelement <64 x i16> %v13, i16 %x0, i32 14
  %v15 = insertelement <64 x i16> %v14, i16 %x1, i32 15
  %v16 = insertelement <64 x i16> %v15, i16 %x0, i32 16
  %v17 = insertelement <64 x i16> %v16, i16 %x1, i32 17
  %v18 = insertelement <64 x i16> %v17, i16 %x0, i32 18
  %v19 = insertelement <64 x i16> %v18, i16 %x1, i32 19
  %v20 = insertelement <64 x i16> %v19, i16 %x0, i32 20
  %v21 = insertelement <64 x i16> %v20, i16 %x1, i32 21
  %v22 = insertelement <64 x i16> %v21, i16 %x0, i32 22
  %v23 = insertelement <64 x i16> %v22, i16 %x1, i32 23
  %v24 = insertelement <64 x i16> %v23, i16 %x0, i32 24
  %v25 = insertelement <64 x i16> %v24, i16 %x1, i32 25
  %v26 = insertelement <64 x i16> %v25, i16 %x0, i32 26
  %v27 = insertelement <64 x i16> %v26, i16 %x1, i32 27
  %v28 = insertelement <64 x i16> %v27, i16 %x0, i32 28
  %v29 = insertelement <64 x i16> %v28, i16 %x1, i32 29
  %v30 = insertelement <64 x i16> %v29, i16 %x0, i32 30
  %v31 = insertelement <64 x i16> %v30, i16 %x1, i32 31
  %v32 = insertelement <64 x i16> %v31, i16 %y, i32 32
  %v33 = insertelement <64 x i16> %v32, i16 %x1, i32 33
  %v34 = insertelement <64 x i16> %v33, i16 %x0, i32 34
  %v35 = insertelement <64 x i16> %v34, i16 %x1, i32 35
  %v36 = insertelement <64 x i16> %v35, i16 %x0, i32 36
  %v37 = insertelement <64 x i16> %v36, i16 %y, i32 37
  %v38 = insertelement <64 x i16> %v37, i16 %x0, i32 38
  %v39 = insertelement <64 x i16> %v38, i16 %x1, i32 39
  %v40 = insertelement <64 x i16> %v39, i16 %x0, i32 40
  %v41 = insertelement <64 x i16> %v40, i16 %x1, i32 41
  %v42 = insertelement <64 x i16> %v41, i16 %y, i32 42
  %v43 = insertelement <64 x i16> %v42, i16 %x1, i32 43
  %v44 = insertelement <64 x i16> %v43, i16 %x0, i32 44
  %v45 = insertelement <64 x i16> %v44, i16 %x1, i32 45
  %v46 = insertelement <64 x i16> %v45, i16 %x0, i32 46
  %v47 = insertelement <64 x i16> %v46, i16 %y, i32 47
  %v48 = insertelement <64 x i16> %v47, i16 %x0, i32 48
  %v49 = insertelement <64 x i16> %v48, i16 %x1, i32 49
  %v50 = insertelement <64 x i16> %v49, i16 %x0, i32 50
  %v51 = insertelement <64 x i16> %v50, i16 %x1, i32 51
  %v52 = insertelement <64 x i16> %v51, i16 %x0, i32 52
  %v53 = insertelement <64 x i16> %v52, i16 %x1, i32 53
  %v54 = insertelement <64 x i16> %v53, i16 %x0, i32 54
  %v55 = insertelement <64 x i16> %v54, i16 %x1, i32 55
  %v56 = insertelement <64 x i16> %v55, i16 %x0, i32 56
  %v57 = insertelement <64 x i16> %v56, i16 %x1, i32 57
  %v58 = insertelement <64 x i16> %v57, i16 %x0, i32 58
  %v59 = insertelement <64 x i16> %v58, i16 %x1, i32 59
  %v60 = insertelement <64 x i16> %v59, i16 %x0, i32 60
  %v61 = insertelement <64 x i16> %v60, i16 %x1, i32 61
  %v62 = insertelement <64 x i16> %v61, i16 %x0, i32 62
  %v63 = insertelement <64 x i16> %v62, i16 %x1, i32 63
  store <64 x i16> %v63, ptr %out, align 128
  ret void
}

define void @splat_i16_into_i16_vector(ptr %out, i16 %x) #0 {
; CHECK-LABEL: splat_i16_into_i16_vector:
; CHECK: v{{[0-9]+}}.h = vsplat(r{{[0-9]+}})
; CHECK: vmem
entry:
  %v0 = insertelement <64 x i16> poison, i16 %x, i32 0
  %v1 = shufflevector <64 x i16> %v0, <64 x i16> poison, <64 x i32> zeroinitializer
  store <64 x i16> %v1, ptr %out, align 128
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv75" "target-features"="+hvxv75,+hvx-length128b" }

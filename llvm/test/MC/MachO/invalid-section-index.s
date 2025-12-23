// REQUIRES: aarch64-registered-target

/// Test that when there are more than 255 sections, error is shown specifying too many sections.
// REQUIRES: stability

// RUN: not llvm-mc -filetype=obj -triple arm64-apple-darwin %s -o - 2>&1 | FileCheck %s --check-prefix=MACHOERROR

// MACHOERROR: error: Too many sections!
// MACHOERROR-NEXT: error: Invalid section index!
// MACHOERROR-NEXT: error: Invalid section index!

	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main                           ; -- Begin function main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:                                ; %entry
	sub	sp, sp, #16
	.cfi_def_cfa_offset 16
	mov	w0, #0                          ; =0x0
	str	wzr, [sp, #12]
	add	sp, sp, #16
	ret
	.cfi_endproc
                                        ; -- End function
	.section	seg,sect0
	.globl	_var0                           ; @var0
	.p2align	2, 0x0
_var0:
	.long	0                               ; 0x0

	.section	seg,sect1
	.globl	_var1                           ; @var1
	.p2align	2, 0x0
_var1:
	.long	1                               ; 0x1

	.section	seg,sect2
	.globl	_var2                           ; @var2
	.p2align	2, 0x0
_var2:
	.long	2                               ; 0x2

	.section	seg,sect3
	.globl	_var3                           ; @var3
	.p2align	2, 0x0
_var3:
	.long	3                               ; 0x3

	.section	seg,sect4
	.globl	_var4                           ; @var4
	.p2align	2, 0x0
_var4:
	.long	4                               ; 0x4

	.section	seg,sect5
	.globl	_var5                           ; @var5
	.p2align	2, 0x0
_var5:
	.long	5                               ; 0x5

	.section	seg,sect6
	.globl	_var6                           ; @var6
	.p2align	2, 0x0
_var6:
	.long	6                               ; 0x6

	.section	seg,sect7
	.globl	_var7                           ; @var7
	.p2align	2, 0x0
_var7:
	.long	7                               ; 0x7

	.section	seg,sect8
	.globl	_var8                           ; @var8
	.p2align	2, 0x0
_var8:
	.long	8                               ; 0x8

	.section	seg,sect9
	.globl	_var9                           ; @var9
	.p2align	2, 0x0
_var9:
	.long	9                               ; 0x9

	.section	seg,sect10
	.globl	_var10                          ; @var10
	.p2align	2, 0x0
_var10:
	.long	10                              ; 0xa

	.section	seg,sect11
	.globl	_var11                          ; @var11
	.p2align	2, 0x0
_var11:
	.long	11                              ; 0xb

	.section	seg,sect12
	.globl	_var12                          ; @var12
	.p2align	2, 0x0
_var12:
	.long	12                              ; 0xc

	.section	seg,sect13
	.globl	_var13                          ; @var13
	.p2align	2, 0x0
_var13:
	.long	13                              ; 0xd

	.section	seg,sect14
	.globl	_var14                          ; @var14
	.p2align	2, 0x0
_var14:
	.long	14                              ; 0xe

	.section	seg,sect15
	.globl	_var15                          ; @var15
	.p2align	2, 0x0
_var15:
	.long	15                              ; 0xf

	.section	seg,sect16
	.globl	_var16                          ; @var16
	.p2align	2, 0x0
_var16:
	.long	16                              ; 0x10

	.section	seg,sect17
	.globl	_var17                          ; @var17
	.p2align	2, 0x0
_var17:
	.long	17                              ; 0x11

	.section	seg,sect18
	.globl	_var18                          ; @var18
	.p2align	2, 0x0
_var18:
	.long	18                              ; 0x12

	.section	seg,sect19
	.globl	_var19                          ; @var19
	.p2align	2, 0x0
_var19:
	.long	19                              ; 0x13

	.section	seg,sect20
	.globl	_var20                          ; @var20
	.p2align	2, 0x0
_var20:
	.long	20                              ; 0x14

	.section	seg,sect21
	.globl	_var21                          ; @var21
	.p2align	2, 0x0
_var21:
	.long	21                              ; 0x15

	.section	seg,sect22
	.globl	_var22                          ; @var22
	.p2align	2, 0x0
_var22:
	.long	22                              ; 0x16

	.section	seg,sect23
	.globl	_var23                          ; @var23
	.p2align	2, 0x0
_var23:
	.long	23                              ; 0x17

	.section	seg,sect24
	.globl	_var24                          ; @var24
	.p2align	2, 0x0
_var24:
	.long	24                              ; 0x18

	.section	seg,sect25
	.globl	_var25                          ; @var25
	.p2align	2, 0x0
_var25:
	.long	25                              ; 0x19

	.section	seg,sect26
	.globl	_var26                          ; @var26
	.p2align	2, 0x0
_var26:
	.long	26                              ; 0x1a

	.section	seg,sect27
	.globl	_var27                          ; @var27
	.p2align	2, 0x0
_var27:
	.long	27                              ; 0x1b

	.section	seg,sect28
	.globl	_var28                          ; @var28
	.p2align	2, 0x0
_var28:
	.long	28                              ; 0x1c

	.section	seg,sect29
	.globl	_var29                          ; @var29
	.p2align	2, 0x0
_var29:
	.long	29                              ; 0x1d

	.section	seg,sect30
	.globl	_var30                          ; @var30
	.p2align	2, 0x0
_var30:
	.long	30                              ; 0x1e

	.section	seg,sect31
	.globl	_var31                          ; @var31
	.p2align	2, 0x0
_var31:
	.long	31                              ; 0x1f

	.section	seg,sect32
	.globl	_var32                          ; @var32
	.p2align	2, 0x0
_var32:
	.long	32                              ; 0x20

	.section	seg,sect33
	.globl	_var33                          ; @var33
	.p2align	2, 0x0
_var33:
	.long	33                              ; 0x21

	.section	seg,sect34
	.globl	_var34                          ; @var34
	.p2align	2, 0x0
_var34:
	.long	34                              ; 0x22

	.section	seg,sect35
	.globl	_var35                          ; @var35
	.p2align	2, 0x0
_var35:
	.long	35                              ; 0x23

	.section	seg,sect36
	.globl	_var36                          ; @var36
	.p2align	2, 0x0
_var36:
	.long	36                              ; 0x24

	.section	seg,sect37
	.globl	_var37                          ; @var37
	.p2align	2, 0x0
_var37:
	.long	37                              ; 0x25

	.section	seg,sect38
	.globl	_var38                          ; @var38
	.p2align	2, 0x0
_var38:
	.long	38                              ; 0x26

	.section	seg,sect39
	.globl	_var39                          ; @var39
	.p2align	2, 0x0
_var39:
	.long	39                              ; 0x27

	.section	seg,sect40
	.globl	_var40                          ; @var40
	.p2align	2, 0x0
_var40:
	.long	40                              ; 0x28

	.section	seg,sect41
	.globl	_var41                          ; @var41
	.p2align	2, 0x0
_var41:
	.long	41                              ; 0x29

	.section	seg,sect42
	.globl	_var42                          ; @var42
	.p2align	2, 0x0
_var42:
	.long	42                              ; 0x2a

	.section	seg,sect43
	.globl	_var43                          ; @var43
	.p2align	2, 0x0
_var43:
	.long	43                              ; 0x2b

	.section	seg,sect44
	.globl	_var44                          ; @var44
	.p2align	2, 0x0
_var44:
	.long	44                              ; 0x2c

	.section	seg,sect45
	.globl	_var45                          ; @var45
	.p2align	2, 0x0
_var45:
	.long	45                              ; 0x2d

	.section	seg,sect46
	.globl	_var46                          ; @var46
	.p2align	2, 0x0
_var46:
	.long	46                              ; 0x2e

	.section	seg,sect47
	.globl	_var47                          ; @var47
	.p2align	2, 0x0
_var47:
	.long	47                              ; 0x2f

	.section	seg,sect48
	.globl	_var48                          ; @var48
	.p2align	2, 0x0
_var48:
	.long	48                              ; 0x30

	.section	seg,sect49
	.globl	_var49                          ; @var49
	.p2align	2, 0x0
_var49:
	.long	49                              ; 0x31

	.section	seg,sect50
	.globl	_var50                          ; @var50
	.p2align	2, 0x0
_var50:
	.long	50                              ; 0x32

	.section	seg,sect51
	.globl	_var51                          ; @var51
	.p2align	2, 0x0
_var51:
	.long	51                              ; 0x33

	.section	seg,sect52
	.globl	_var52                          ; @var52
	.p2align	2, 0x0
_var52:
	.long	52                              ; 0x34

	.section	seg,sect53
	.globl	_var53                          ; @var53
	.p2align	2, 0x0
_var53:
	.long	53                              ; 0x35

	.section	seg,sect54
	.globl	_var54                          ; @var54
	.p2align	2, 0x0
_var54:
	.long	54                              ; 0x36

	.section	seg,sect55
	.globl	_var55                          ; @var55
	.p2align	2, 0x0
_var55:
	.long	55                              ; 0x37

	.section	seg,sect56
	.globl	_var56                          ; @var56
	.p2align	2, 0x0
_var56:
	.long	56                              ; 0x38

	.section	seg,sect57
	.globl	_var57                          ; @var57
	.p2align	2, 0x0
_var57:
	.long	57                              ; 0x39

	.section	seg,sect58
	.globl	_var58                          ; @var58
	.p2align	2, 0x0
_var58:
	.long	58                              ; 0x3a

	.section	seg,sect59
	.globl	_var59                          ; @var59
	.p2align	2, 0x0
_var59:
	.long	59                              ; 0x3b

	.section	seg,sect60
	.globl	_var60                          ; @var60
	.p2align	2, 0x0
_var60:
	.long	60                              ; 0x3c

	.section	seg,sect61
	.globl	_var61                          ; @var61
	.p2align	2, 0x0
_var61:
	.long	61                              ; 0x3d

	.section	seg,sect62
	.globl	_var62                          ; @var62
	.p2align	2, 0x0
_var62:
	.long	62                              ; 0x3e

	.section	seg,sect63
	.globl	_var63                          ; @var63
	.p2align	2, 0x0
_var63:
	.long	63                              ; 0x3f

	.section	seg,sect64
	.globl	_var64                          ; @var64
	.p2align	2, 0x0
_var64:
	.long	64                              ; 0x40

	.section	seg,sect65
	.globl	_var65                          ; @var65
	.p2align	2, 0x0
_var65:
	.long	65                              ; 0x41

	.section	seg,sect66
	.globl	_var66                          ; @var66
	.p2align	2, 0x0
_var66:
	.long	66                              ; 0x42

	.section	seg,sect67
	.globl	_var67                          ; @var67
	.p2align	2, 0x0
_var67:
	.long	67                              ; 0x43

	.section	seg,sect68
	.globl	_var68                          ; @var68
	.p2align	2, 0x0
_var68:
	.long	68                              ; 0x44

	.section	seg,sect69
	.globl	_var69                          ; @var69
	.p2align	2, 0x0
_var69:
	.long	69                              ; 0x45

	.section	seg,sect70
	.globl	_var70                          ; @var70
	.p2align	2, 0x0
_var70:
	.long	70                              ; 0x46

	.section	seg,sect71
	.globl	_var71                          ; @var71
	.p2align	2, 0x0
_var71:
	.long	71                              ; 0x47

	.section	seg,sect72
	.globl	_var72                          ; @var72
	.p2align	2, 0x0
_var72:
	.long	72                              ; 0x48

	.section	seg,sect73
	.globl	_var73                          ; @var73
	.p2align	2, 0x0
_var73:
	.long	73                              ; 0x49

	.section	seg,sect74
	.globl	_var74                          ; @var74
	.p2align	2, 0x0
_var74:
	.long	74                              ; 0x4a

	.section	seg,sect75
	.globl	_var75                          ; @var75
	.p2align	2, 0x0
_var75:
	.long	75                              ; 0x4b

	.section	seg,sect76
	.globl	_var76                          ; @var76
	.p2align	2, 0x0
_var76:
	.long	76                              ; 0x4c

	.section	seg,sect77
	.globl	_var77                          ; @var77
	.p2align	2, 0x0
_var77:
	.long	77                              ; 0x4d

	.section	seg,sect78
	.globl	_var78                          ; @var78
	.p2align	2, 0x0
_var78:
	.long	78                              ; 0x4e

	.section	seg,sect79
	.globl	_var79                          ; @var79
	.p2align	2, 0x0
_var79:
	.long	79                              ; 0x4f

	.section	seg,sect80
	.globl	_var80                          ; @var80
	.p2align	2, 0x0
_var80:
	.long	80                              ; 0x50

	.section	seg,sect81
	.globl	_var81                          ; @var81
	.p2align	2, 0x0
_var81:
	.long	81                              ; 0x51

	.section	seg,sect82
	.globl	_var82                          ; @var82
	.p2align	2, 0x0
_var82:
	.long	82                              ; 0x52

	.section	seg,sect83
	.globl	_var83                          ; @var83
	.p2align	2, 0x0
_var83:
	.long	83                              ; 0x53

	.section	seg,sect84
	.globl	_var84                          ; @var84
	.p2align	2, 0x0
_var84:
	.long	84                              ; 0x54

	.section	seg,sect85
	.globl	_var85                          ; @var85
	.p2align	2, 0x0
_var85:
	.long	85                              ; 0x55

	.section	seg,sect86
	.globl	_var86                          ; @var86
	.p2align	2, 0x0
_var86:
	.long	86                              ; 0x56

	.section	seg,sect87
	.globl	_var87                          ; @var87
	.p2align	2, 0x0
_var87:
	.long	87                              ; 0x57

	.section	seg,sect88
	.globl	_var88                          ; @var88
	.p2align	2, 0x0
_var88:
	.long	88                              ; 0x58

	.section	seg,sect89
	.globl	_var89                          ; @var89
	.p2align	2, 0x0
_var89:
	.long	89                              ; 0x59

	.section	seg,sect90
	.globl	_var90                          ; @var90
	.p2align	2, 0x0
_var90:
	.long	90                              ; 0x5a

	.section	seg,sect91
	.globl	_var91                          ; @var91
	.p2align	2, 0x0
_var91:
	.long	91                              ; 0x5b

	.section	seg,sect92
	.globl	_var92                          ; @var92
	.p2align	2, 0x0
_var92:
	.long	92                              ; 0x5c

	.section	seg,sect93
	.globl	_var93                          ; @var93
	.p2align	2, 0x0
_var93:
	.long	93                              ; 0x5d

	.section	seg,sect94
	.globl	_var94                          ; @var94
	.p2align	2, 0x0
_var94:
	.long	94                              ; 0x5e

	.section	seg,sect95
	.globl	_var95                          ; @var95
	.p2align	2, 0x0
_var95:
	.long	95                              ; 0x5f

	.section	seg,sect96
	.globl	_var96                          ; @var96
	.p2align	2, 0x0
_var96:
	.long	96                              ; 0x60

	.section	seg,sect97
	.globl	_var97                          ; @var97
	.p2align	2, 0x0
_var97:
	.long	97                              ; 0x61

	.section	seg,sect98
	.globl	_var98                          ; @var98
	.p2align	2, 0x0
_var98:
	.long	98                              ; 0x62

	.section	seg,sect99
	.globl	_var99                          ; @var99
	.p2align	2, 0x0
_var99:
	.long	99                              ; 0x63

	.section	seg,sect100
	.globl	_var100                         ; @var100
	.p2align	2, 0x0
_var100:
	.long	100                             ; 0x64

	.section	seg,sect101
	.globl	_var101                         ; @var101
	.p2align	2, 0x0
_var101:
	.long	101                             ; 0x65

	.section	seg,sect102
	.globl	_var102                         ; @var102
	.p2align	2, 0x0
_var102:
	.long	102                             ; 0x66

	.section	seg,sect103
	.globl	_var103                         ; @var103
	.p2align	2, 0x0
_var103:
	.long	103                             ; 0x67

	.section	seg,sect104
	.globl	_var104                         ; @var104
	.p2align	2, 0x0
_var104:
	.long	104                             ; 0x68

	.section	seg,sect105
	.globl	_var105                         ; @var105
	.p2align	2, 0x0
_var105:
	.long	105                             ; 0x69

	.section	seg,sect106
	.globl	_var106                         ; @var106
	.p2align	2, 0x0
_var106:
	.long	106                             ; 0x6a

	.section	seg,sect107
	.globl	_var107                         ; @var107
	.p2align	2, 0x0
_var107:
	.long	107                             ; 0x6b

	.section	seg,sect108
	.globl	_var108                         ; @var108
	.p2align	2, 0x0
_var108:
	.long	108                             ; 0x6c

	.section	seg,sect109
	.globl	_var109                         ; @var109
	.p2align	2, 0x0
_var109:
	.long	109                             ; 0x6d

	.section	seg,sect110
	.globl	_var110                         ; @var110
	.p2align	2, 0x0
_var110:
	.long	110                             ; 0x6e

	.section	seg,sect111
	.globl	_var111                         ; @var111
	.p2align	2, 0x0
_var111:
	.long	111                             ; 0x6f

	.section	seg,sect112
	.globl	_var112                         ; @var112
	.p2align	2, 0x0
_var112:
	.long	112                             ; 0x70

	.section	seg,sect113
	.globl	_var113                         ; @var113
	.p2align	2, 0x0
_var113:
	.long	113                             ; 0x71

	.section	seg,sect114
	.globl	_var114                         ; @var114
	.p2align	2, 0x0
_var114:
	.long	114                             ; 0x72

	.section	seg,sect115
	.globl	_var115                         ; @var115
	.p2align	2, 0x0
_var115:
	.long	115                             ; 0x73

	.section	seg,sect116
	.globl	_var116                         ; @var116
	.p2align	2, 0x0
_var116:
	.long	116                             ; 0x74

	.section	seg,sect117
	.globl	_var117                         ; @var117
	.p2align	2, 0x0
_var117:
	.long	117                             ; 0x75

	.section	seg,sect118
	.globl	_var118                         ; @var118
	.p2align	2, 0x0
_var118:
	.long	118                             ; 0x76

	.section	seg,sect119
	.globl	_var119                         ; @var119
	.p2align	2, 0x0
_var119:
	.long	119                             ; 0x77

	.section	seg,sect120
	.globl	_var120                         ; @var120
	.p2align	2, 0x0
_var120:
	.long	120                             ; 0x78

	.section	seg,sect121
	.globl	_var121                         ; @var121
	.p2align	2, 0x0
_var121:
	.long	121                             ; 0x79

	.section	seg,sect122
	.globl	_var122                         ; @var122
	.p2align	2, 0x0
_var122:
	.long	122                             ; 0x7a

	.section	seg,sect123
	.globl	_var123                         ; @var123
	.p2align	2, 0x0
_var123:
	.long	123                             ; 0x7b

	.section	seg,sect124
	.globl	_var124                         ; @var124
	.p2align	2, 0x0
_var124:
	.long	124                             ; 0x7c

	.section	seg,sect125
	.globl	_var125                         ; @var125
	.p2align	2, 0x0
_var125:
	.long	125                             ; 0x7d

	.section	seg,sect126
	.globl	_var126                         ; @var126
	.p2align	2, 0x0
_var126:
	.long	126                             ; 0x7e

	.section	seg,sect127
	.globl	_var127                         ; @var127
	.p2align	2, 0x0
_var127:
	.long	127                             ; 0x7f

	.section	seg,sect128
	.globl	_var128                         ; @var128
	.p2align	2, 0x0
_var128:
	.long	128                             ; 0x80

	.section	seg,sect129
	.globl	_var129                         ; @var129
	.p2align	2, 0x0
_var129:
	.long	129                             ; 0x81

	.section	seg,sect130
	.globl	_var130                         ; @var130
	.p2align	2, 0x0
_var130:
	.long	130                             ; 0x82

	.section	seg,sect131
	.globl	_var131                         ; @var131
	.p2align	2, 0x0
_var131:
	.long	131                             ; 0x83

	.section	seg,sect132
	.globl	_var132                         ; @var132
	.p2align	2, 0x0
_var132:
	.long	132                             ; 0x84

	.section	seg,sect133
	.globl	_var133                         ; @var133
	.p2align	2, 0x0
_var133:
	.long	133                             ; 0x85

	.section	seg,sect134
	.globl	_var134                         ; @var134
	.p2align	2, 0x0
_var134:
	.long	134                             ; 0x86

	.section	seg,sect135
	.globl	_var135                         ; @var135
	.p2align	2, 0x0
_var135:
	.long	135                             ; 0x87

	.section	seg,sect136
	.globl	_var136                         ; @var136
	.p2align	2, 0x0
_var136:
	.long	136                             ; 0x88

	.section	seg,sect137
	.globl	_var137                         ; @var137
	.p2align	2, 0x0
_var137:
	.long	137                             ; 0x89

	.section	seg,sect138
	.globl	_var138                         ; @var138
	.p2align	2, 0x0
_var138:
	.long	138                             ; 0x8a

	.section	seg,sect139
	.globl	_var139                         ; @var139
	.p2align	2, 0x0
_var139:
	.long	139                             ; 0x8b

	.section	seg,sect140
	.globl	_var140                         ; @var140
	.p2align	2, 0x0
_var140:
	.long	140                             ; 0x8c

	.section	seg,sect141
	.globl	_var141                         ; @var141
	.p2align	2, 0x0
_var141:
	.long	141                             ; 0x8d

	.section	seg,sect142
	.globl	_var142                         ; @var142
	.p2align	2, 0x0
_var142:
	.long	142                             ; 0x8e

	.section	seg,sect143
	.globl	_var143                         ; @var143
	.p2align	2, 0x0
_var143:
	.long	143                             ; 0x8f

	.section	seg,sect144
	.globl	_var144                         ; @var144
	.p2align	2, 0x0
_var144:
	.long	144                             ; 0x90

	.section	seg,sect145
	.globl	_var145                         ; @var145
	.p2align	2, 0x0
_var145:
	.long	145                             ; 0x91

	.section	seg,sect146
	.globl	_var146                         ; @var146
	.p2align	2, 0x0
_var146:
	.long	146                             ; 0x92

	.section	seg,sect147
	.globl	_var147                         ; @var147
	.p2align	2, 0x0
_var147:
	.long	147                             ; 0x93

	.section	seg,sect148
	.globl	_var148                         ; @var148
	.p2align	2, 0x0
_var148:
	.long	148                             ; 0x94

	.section	seg,sect149
	.globl	_var149                         ; @var149
	.p2align	2, 0x0
_var149:
	.long	149                             ; 0x95

	.section	seg,sect150
	.globl	_var150                         ; @var150
	.p2align	2, 0x0
_var150:
	.long	150                             ; 0x96

	.section	seg,sect151
	.globl	_var151                         ; @var151
	.p2align	2, 0x0
_var151:
	.long	151                             ; 0x97

	.section	seg,sect152
	.globl	_var152                         ; @var152
	.p2align	2, 0x0
_var152:
	.long	152                             ; 0x98

	.section	seg,sect153
	.globl	_var153                         ; @var153
	.p2align	2, 0x0
_var153:
	.long	153                             ; 0x99

	.section	seg,sect154
	.globl	_var154                         ; @var154
	.p2align	2, 0x0
_var154:
	.long	154                             ; 0x9a

	.section	seg,sect155
	.globl	_var155                         ; @var155
	.p2align	2, 0x0
_var155:
	.long	155                             ; 0x9b

	.section	seg,sect156
	.globl	_var156                         ; @var156
	.p2align	2, 0x0
_var156:
	.long	156                             ; 0x9c

	.section	seg,sect157
	.globl	_var157                         ; @var157
	.p2align	2, 0x0
_var157:
	.long	157                             ; 0x9d

	.section	seg,sect158
	.globl	_var158                         ; @var158
	.p2align	2, 0x0
_var158:
	.long	158                             ; 0x9e

	.section	seg,sect159
	.globl	_var159                         ; @var159
	.p2align	2, 0x0
_var159:
	.long	159                             ; 0x9f

	.section	seg,sect160
	.globl	_var160                         ; @var160
	.p2align	2, 0x0
_var160:
	.long	160                             ; 0xa0

	.section	seg,sect161
	.globl	_var161                         ; @var161
	.p2align	2, 0x0
_var161:
	.long	161                             ; 0xa1

	.section	seg,sect162
	.globl	_var162                         ; @var162
	.p2align	2, 0x0
_var162:
	.long	162                             ; 0xa2

	.section	seg,sect163
	.globl	_var163                         ; @var163
	.p2align	2, 0x0
_var163:
	.long	163                             ; 0xa3

	.section	seg,sect164
	.globl	_var164                         ; @var164
	.p2align	2, 0x0
_var164:
	.long	164                             ; 0xa4

	.section	seg,sect165
	.globl	_var165                         ; @var165
	.p2align	2, 0x0
_var165:
	.long	165                             ; 0xa5

	.section	seg,sect166
	.globl	_var166                         ; @var166
	.p2align	2, 0x0
_var166:
	.long	166                             ; 0xa6

	.section	seg,sect167
	.globl	_var167                         ; @var167
	.p2align	2, 0x0
_var167:
	.long	167                             ; 0xa7

	.section	seg,sect168
	.globl	_var168                         ; @var168
	.p2align	2, 0x0
_var168:
	.long	168                             ; 0xa8

	.section	seg,sect169
	.globl	_var169                         ; @var169
	.p2align	2, 0x0
_var169:
	.long	169                             ; 0xa9

	.section	seg,sect170
	.globl	_var170                         ; @var170
	.p2align	2, 0x0
_var170:
	.long	170                             ; 0xaa

	.section	seg,sect171
	.globl	_var171                         ; @var171
	.p2align	2, 0x0
_var171:
	.long	171                             ; 0xab

	.section	seg,sect172
	.globl	_var172                         ; @var172
	.p2align	2, 0x0
_var172:
	.long	172                             ; 0xac

	.section	seg,sect173
	.globl	_var173                         ; @var173
	.p2align	2, 0x0
_var173:
	.long	173                             ; 0xad

	.section	seg,sect174
	.globl	_var174                         ; @var174
	.p2align	2, 0x0
_var174:
	.long	174                             ; 0xae

	.section	seg,sect175
	.globl	_var175                         ; @var175
	.p2align	2, 0x0
_var175:
	.long	175                             ; 0xaf

	.section	seg,sect176
	.globl	_var176                         ; @var176
	.p2align	2, 0x0
_var176:
	.long	176                             ; 0xb0

	.section	seg,sect177
	.globl	_var177                         ; @var177
	.p2align	2, 0x0
_var177:
	.long	177                             ; 0xb1

	.section	seg,sect178
	.globl	_var178                         ; @var178
	.p2align	2, 0x0
_var178:
	.long	178                             ; 0xb2

	.section	seg,sect179
	.globl	_var179                         ; @var179
	.p2align	2, 0x0
_var179:
	.long	179                             ; 0xb3

	.section	seg,sect180
	.globl	_var180                         ; @var180
	.p2align	2, 0x0
_var180:
	.long	180                             ; 0xb4

	.section	seg,sect181
	.globl	_var181                         ; @var181
	.p2align	2, 0x0
_var181:
	.long	181                             ; 0xb5

	.section	seg,sect182
	.globl	_var182                         ; @var182
	.p2align	2, 0x0
_var182:
	.long	182                             ; 0xb6

	.section	seg,sect183
	.globl	_var183                         ; @var183
	.p2align	2, 0x0
_var183:
	.long	183                             ; 0xb7

	.section	seg,sect184
	.globl	_var184                         ; @var184
	.p2align	2, 0x0
_var184:
	.long	184                             ; 0xb8

	.section	seg,sect185
	.globl	_var185                         ; @var185
	.p2align	2, 0x0
_var185:
	.long	185                             ; 0xb9

	.section	seg,sect186
	.globl	_var186                         ; @var186
	.p2align	2, 0x0
_var186:
	.long	186                             ; 0xba

	.section	seg,sect187
	.globl	_var187                         ; @var187
	.p2align	2, 0x0
_var187:
	.long	187                             ; 0xbb

	.section	seg,sect188
	.globl	_var188                         ; @var188
	.p2align	2, 0x0
_var188:
	.long	188                             ; 0xbc

	.section	seg,sect189
	.globl	_var189                         ; @var189
	.p2align	2, 0x0
_var189:
	.long	189                             ; 0xbd

	.section	seg,sect190
	.globl	_var190                         ; @var190
	.p2align	2, 0x0
_var190:
	.long	190                             ; 0xbe

	.section	seg,sect191
	.globl	_var191                         ; @var191
	.p2align	2, 0x0
_var191:
	.long	191                             ; 0xbf

	.section	seg,sect192
	.globl	_var192                         ; @var192
	.p2align	2, 0x0
_var192:
	.long	192                             ; 0xc0

	.section	seg,sect193
	.globl	_var193                         ; @var193
	.p2align	2, 0x0
_var193:
	.long	193                             ; 0xc1

	.section	seg,sect194
	.globl	_var194                         ; @var194
	.p2align	2, 0x0
_var194:
	.long	194                             ; 0xc2

	.section	seg,sect195
	.globl	_var195                         ; @var195
	.p2align	2, 0x0
_var195:
	.long	195                             ; 0xc3

	.section	seg,sect196
	.globl	_var196                         ; @var196
	.p2align	2, 0x0
_var196:
	.long	196                             ; 0xc4

	.section	seg,sect197
	.globl	_var197                         ; @var197
	.p2align	2, 0x0
_var197:
	.long	197                             ; 0xc5

	.section	seg,sect198
	.globl	_var198                         ; @var198
	.p2align	2, 0x0
_var198:
	.long	198                             ; 0xc6

	.section	seg,sect199
	.globl	_var199                         ; @var199
	.p2align	2, 0x0
_var199:
	.long	199                             ; 0xc7

	.section	seg,sect200
	.globl	_var200                         ; @var200
	.p2align	2, 0x0
_var200:
	.long	200                             ; 0xc8

	.section	seg,sect201
	.globl	_var201                         ; @var201
	.p2align	2, 0x0
_var201:
	.long	201                             ; 0xc9

	.section	seg,sect202
	.globl	_var202                         ; @var202
	.p2align	2, 0x0
_var202:
	.long	202                             ; 0xca

	.section	seg,sect203
	.globl	_var203                         ; @var203
	.p2align	2, 0x0
_var203:
	.long	203                             ; 0xcb

	.section	seg,sect204
	.globl	_var204                         ; @var204
	.p2align	2, 0x0
_var204:
	.long	204                             ; 0xcc

	.section	seg,sect205
	.globl	_var205                         ; @var205
	.p2align	2, 0x0
_var205:
	.long	205                             ; 0xcd

	.section	seg,sect206
	.globl	_var206                         ; @var206
	.p2align	2, 0x0
_var206:
	.long	206                             ; 0xce

	.section	seg,sect207
	.globl	_var207                         ; @var207
	.p2align	2, 0x0
_var207:
	.long	207                             ; 0xcf

	.section	seg,sect208
	.globl	_var208                         ; @var208
	.p2align	2, 0x0
_var208:
	.long	208                             ; 0xd0

	.section	seg,sect209
	.globl	_var209                         ; @var209
	.p2align	2, 0x0
_var209:
	.long	209                             ; 0xd1

	.section	seg,sect210
	.globl	_var210                         ; @var210
	.p2align	2, 0x0
_var210:
	.long	210                             ; 0xd2

	.section	seg,sect211
	.globl	_var211                         ; @var211
	.p2align	2, 0x0
_var211:
	.long	211                             ; 0xd3

	.section	seg,sect212
	.globl	_var212                         ; @var212
	.p2align	2, 0x0
_var212:
	.long	212                             ; 0xd4

	.section	seg,sect213
	.globl	_var213                         ; @var213
	.p2align	2, 0x0
_var213:
	.long	213                             ; 0xd5

	.section	seg,sect214
	.globl	_var214                         ; @var214
	.p2align	2, 0x0
_var214:
	.long	214                             ; 0xd6

	.section	seg,sect215
	.globl	_var215                         ; @var215
	.p2align	2, 0x0
_var215:
	.long	215                             ; 0xd7

	.section	seg,sect216
	.globl	_var216                         ; @var216
	.p2align	2, 0x0
_var216:
	.long	216                             ; 0xd8

	.section	seg,sect217
	.globl	_var217                         ; @var217
	.p2align	2, 0x0
_var217:
	.long	217                             ; 0xd9

	.section	seg,sect218
	.globl	_var218                         ; @var218
	.p2align	2, 0x0
_var218:
	.long	218                             ; 0xda

	.section	seg,sect219
	.globl	_var219                         ; @var219
	.p2align	2, 0x0
_var219:
	.long	219                             ; 0xdb

	.section	seg,sect220
	.globl	_var220                         ; @var220
	.p2align	2, 0x0
_var220:
	.long	220                             ; 0xdc

	.section	seg,sect221
	.globl	_var221                         ; @var221
	.p2align	2, 0x0
_var221:
	.long	221                             ; 0xdd

	.section	seg,sect222
	.globl	_var222                         ; @var222
	.p2align	2, 0x0
_var222:
	.long	222                             ; 0xde

	.section	seg,sect223
	.globl	_var223                         ; @var223
	.p2align	2, 0x0
_var223:
	.long	223                             ; 0xdf

	.section	seg,sect224
	.globl	_var224                         ; @var224
	.p2align	2, 0x0
_var224:
	.long	224                             ; 0xe0

	.section	seg,sect225
	.globl	_var225                         ; @var225
	.p2align	2, 0x0
_var225:
	.long	225                             ; 0xe1

	.section	seg,sect226
	.globl	_var226                         ; @var226
	.p2align	2, 0x0
_var226:
	.long	226                             ; 0xe2

	.section	seg,sect227
	.globl	_var227                         ; @var227
	.p2align	2, 0x0
_var227:
	.long	227                             ; 0xe3

	.section	seg,sect228
	.globl	_var228                         ; @var228
	.p2align	2, 0x0
_var228:
	.long	228                             ; 0xe4

	.section	seg,sect229
	.globl	_var229                         ; @var229
	.p2align	2, 0x0
_var229:
	.long	229                             ; 0xe5

	.section	seg,sect230
	.globl	_var230                         ; @var230
	.p2align	2, 0x0
_var230:
	.long	230                             ; 0xe6

	.section	seg,sect231
	.globl	_var231                         ; @var231
	.p2align	2, 0x0
_var231:
	.long	231                             ; 0xe7

	.section	seg,sect232
	.globl	_var232                         ; @var232
	.p2align	2, 0x0
_var232:
	.long	232                             ; 0xe8

	.section	seg,sect233
	.globl	_var233                         ; @var233
	.p2align	2, 0x0
_var233:
	.long	233                             ; 0xe9

	.section	seg,sect234
	.globl	_var234                         ; @var234
	.p2align	2, 0x0
_var234:
	.long	234                             ; 0xea

	.section	seg,sect235
	.globl	_var235                         ; @var235
	.p2align	2, 0x0
_var235:
	.long	235                             ; 0xeb

	.section	seg,sect236
	.globl	_var236                         ; @var236
	.p2align	2, 0x0
_var236:
	.long	236                             ; 0xec

	.section	seg,sect237
	.globl	_var237                         ; @var237
	.p2align	2, 0x0
_var237:
	.long	237                             ; 0xed

	.section	seg,sect238
	.globl	_var238                         ; @var238
	.p2align	2, 0x0
_var238:
	.long	238                             ; 0xee

	.section	seg,sect239
	.globl	_var239                         ; @var239
	.p2align	2, 0x0
_var239:
	.long	239                             ; 0xef

	.section	seg,sect240
	.globl	_var240                         ; @var240
	.p2align	2, 0x0
_var240:
	.long	240                             ; 0xf0

	.section	seg,sect241
	.globl	_var241                         ; @var241
	.p2align	2, 0x0
_var241:
	.long	241                             ; 0xf1

	.section	seg,sect242
	.globl	_var242                         ; @var242
	.p2align	2, 0x0
_var242:
	.long	242                             ; 0xf2

	.section	seg,sect243
	.globl	_var243                         ; @var243
	.p2align	2, 0x0
_var243:
	.long	243                             ; 0xf3

	.section	seg,sect244
	.globl	_var244                         ; @var244
	.p2align	2, 0x0
_var244:
	.long	244                             ; 0xf4

	.section	seg,sect245
	.globl	_var245                         ; @var245
	.p2align	2, 0x0
_var245:
	.long	245                             ; 0xf5

	.section	seg,sect246
	.globl	_var246                         ; @var246
	.p2align	2, 0x0
_var246:
	.long	246                             ; 0xf6

	.section	seg,sect247
	.globl	_var247                         ; @var247
	.p2align	2, 0x0
_var247:
	.long	247                             ; 0xf7

	.section	seg,sect248
	.globl	_var248                         ; @var248
	.p2align	2, 0x0
_var248:
	.long	248                             ; 0xf8

	.section	seg,sect249
	.globl	_var249                         ; @var249
	.p2align	2, 0x0
_var249:
	.long	249                             ; 0xf9

	.section	seg,sect250
	.globl	_var250                         ; @var250
	.p2align	2, 0x0
_var250:
	.long	250                             ; 0xfa

	.section	seg,sect251
	.globl	_var251                         ; @var251
	.p2align	2, 0x0
_var251:
	.long	251                             ; 0xfb

	.section	seg,sect252
	.globl	_var252                         ; @var252
	.p2align	2, 0x0
_var252:
	.long	252                             ; 0xfc

	.section	seg,sect253
	.globl	_var253                         ; @var253
	.p2align	2, 0x0
_var253:
	.long	253                             ; 0xfd

	.section	seg,sect254
	.globl	_var254                         ; @var254
	.p2align	2, 0x0
_var254:
	.long	254                             ; 0xfe

	.section	seg,sect255
	.globl	_var255                         ; @var255
	.p2align	2, 0x0
_var255:
	.long	255                             ; 0xff

	.section	seg,sect256
	.globl	_var256                         ; @var256
	.p2align	2, 0x0
_var256:
	.long	256                             ; 0x100

	.section	seg,sect257
	.globl	_var257                         ; @var257
	.p2align	2, 0x0
_var257:
	.long	257                             ; 0x101

.subsections_via_symbols

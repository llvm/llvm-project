; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=g5

define void @test(ptr, ptr) {
xOperationInitMasks.exit:
	%.sub7896 = getelementptr [4 x <4 x i32>], ptr null, i32 0, i32 0		; <ptr> [#uses=24]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 175, i32 3		; <ptr>:2 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 174, i32 2		; <ptr>:3 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 174, i32 3		; <ptr>:4 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 173, i32 1		; <ptr>:5 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 173, i32 2		; <ptr>:6 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 173, i32 3		; <ptr>:7 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 172, i32 1		; <ptr>:8 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 172, i32 2		; <ptr>:9 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 172, i32 3		; <ptr>:10 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 171, i32 1		; <ptr>:11 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 171, i32 2		; <ptr>:12 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 171, i32 3		; <ptr>:13 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 170, i32 1		; <ptr>:14 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 170, i32 2		; <ptr>:15 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 170, i32 3		; <ptr>:16 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 169, i32 1		; <ptr>:17 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 169, i32 2		; <ptr>:18 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 169, i32 3		; <ptr>:19 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 168, i32 1		; <ptr>:20 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 168, i32 2		; <ptr>:21 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 168, i32 3		; <ptr>:22 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 167, i32 1		; <ptr>:23 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 167, i32 2		; <ptr>:24 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 167, i32 3		; <ptr>:25 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 166, i32 1		; <ptr>:26 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 166, i32 2		; <ptr>:27 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 166, i32 3		; <ptr>:28 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 165, i32 1		; <ptr>:29 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 165, i32 2		; <ptr>:30 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 165, i32 3		; <ptr>:31 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 164, i32 1		; <ptr>:32 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 164, i32 2		; <ptr>:33 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 164, i32 3		; <ptr>:34 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 163, i32 1		; <ptr>:35 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 163, i32 2		; <ptr>:36 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 163, i32 3		; <ptr>:37 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 162, i32 1		; <ptr>:38 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 162, i32 2		; <ptr>:39 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 162, i32 3		; <ptr>:40 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 161, i32 1		; <ptr>:41 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 161, i32 2		; <ptr>:42 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 161, i32 3		; <ptr>:43 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 160, i32 1		; <ptr>:44 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 160, i32 2		; <ptr>:45 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 160, i32 3		; <ptr>:46 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 159, i32 1		; <ptr>:47 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 159, i32 2		; <ptr>:48 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 159, i32 3		; <ptr>:49 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 158, i32 1		; <ptr>:50 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 158, i32 2		; <ptr>:51 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 158, i32 3		; <ptr>:52 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 157, i32 1		; <ptr>:53 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 157, i32 2		; <ptr>:54 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 157, i32 3		; <ptr>:55 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 156, i32 1		; <ptr>:56 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 156, i32 2		; <ptr>:57 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 156, i32 3		; <ptr>:58 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 155, i32 1		; <ptr>:59 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 155, i32 2		; <ptr>:60 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 155, i32 3		; <ptr>:61 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 154, i32 1		; <ptr>:62 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 154, i32 2		; <ptr>:63 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 154, i32 3		; <ptr>:64 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 153, i32 1		; <ptr>:65 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 153, i32 2		; <ptr>:66 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 153, i32 3		; <ptr>:67 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 152, i32 1		; <ptr>:68 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 152, i32 2		; <ptr>:69 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 152, i32 3		; <ptr>:70 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 151, i32 1		; <ptr>:71 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 151, i32 2		; <ptr>:72 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 151, i32 3		; <ptr>:73 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 150, i32 1		; <ptr>:74 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 150, i32 2		; <ptr>:75 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 150, i32 3		; <ptr>:76 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 149, i32 1		; <ptr>:77 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 149, i32 2		; <ptr>:78 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 149, i32 3		; <ptr>:79 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 148, i32 1		; <ptr>:80 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 148, i32 2		; <ptr>:81 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 148, i32 3		; <ptr>:82 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 147, i32 1		; <ptr>:83 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 147, i32 2		; <ptr>:84 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 147, i32 3		; <ptr>:85 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 146, i32 1		; <ptr>:86 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 146, i32 2		; <ptr>:87 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 146, i32 3		; <ptr>:88 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 145, i32 1		; <ptr>:89 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 145, i32 2		; <ptr>:90 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 145, i32 3		; <ptr>:91 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 144, i32 1		; <ptr>:92 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 144, i32 2		; <ptr>:93 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 144, i32 3		; <ptr>:94 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 143, i32 1		; <ptr>:95 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 143, i32 2		; <ptr>:96 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 143, i32 3		; <ptr>:97 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 142, i32 1		; <ptr>:98 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 142, i32 2		; <ptr>:99 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 142, i32 3		; <ptr>:100 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 141, i32 1		; <ptr>:101 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 141, i32 2		; <ptr>:102 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 141, i32 3		; <ptr>:103 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 140, i32 1		; <ptr>:104 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 140, i32 2		; <ptr>:105 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 140, i32 3		; <ptr>:106 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 139, i32 1		; <ptr>:107 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 139, i32 2		; <ptr>:108 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 139, i32 3		; <ptr>:109 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 138, i32 1		; <ptr>:110 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 138, i32 2		; <ptr>:111 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 138, i32 3		; <ptr>:112 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 137, i32 1		; <ptr>:113 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 137, i32 2		; <ptr>:114 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 137, i32 3		; <ptr>:115 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 136, i32 1		; <ptr>:116 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 136, i32 2		; <ptr>:117 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 136, i32 3		; <ptr>:118 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 135, i32 1		; <ptr>:119 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 135, i32 2		; <ptr>:120 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 135, i32 3		; <ptr>:121 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 134, i32 1		; <ptr>:122 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 134, i32 2		; <ptr>:123 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 134, i32 3		; <ptr>:124 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 133, i32 1		; <ptr>:125 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 133, i32 2		; <ptr>:126 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 133, i32 3		; <ptr>:127 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 132, i32 1		; <ptr>:128 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 132, i32 2		; <ptr>:129 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 132, i32 3		; <ptr>:130 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 131, i32 1		; <ptr>:131 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 131, i32 2		; <ptr>:132 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 131, i32 3		; <ptr>:133 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 130, i32 1		; <ptr>:134 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 130, i32 2		; <ptr>:135 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 130, i32 3		; <ptr>:136 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 129, i32 1		; <ptr>:137 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 129, i32 2		; <ptr>:138 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 129, i32 3		; <ptr>:139 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 128, i32 1		; <ptr>:140 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 128, i32 2		; <ptr>:141 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 128, i32 3		; <ptr>:142 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 127, i32 1		; <ptr>:143 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 127, i32 2		; <ptr>:144 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 127, i32 3		; <ptr>:145 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 126, i32 1		; <ptr>:146 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 126, i32 2		; <ptr>:147 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 126, i32 3		; <ptr>:148 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 125, i32 1		; <ptr>:149 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 125, i32 2		; <ptr>:150 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 125, i32 3		; <ptr>:151 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 124, i32 1		; <ptr>:152 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 124, i32 2		; <ptr>:153 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 124, i32 3		; <ptr>:154 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 123, i32 1		; <ptr>:155 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 123, i32 2		; <ptr>:156 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 123, i32 3		; <ptr>:157 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 122, i32 1		; <ptr>:158 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 122, i32 2		; <ptr>:159 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 122, i32 3		; <ptr>:160 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 121, i32 1		; <ptr>:161 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 121, i32 2		; <ptr>:162 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 121, i32 3		; <ptr>:163 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 120, i32 1		; <ptr>:164 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 120, i32 2		; <ptr>:165 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 120, i32 3		; <ptr>:166 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 119, i32 1		; <ptr>:167 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 119, i32 2		; <ptr>:168 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 119, i32 3		; <ptr>:169 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 118, i32 1		; <ptr>:170 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 118, i32 2		; <ptr>:171 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 118, i32 3		; <ptr>:172 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 117, i32 1		; <ptr>:173 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 117, i32 2		; <ptr>:174 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 117, i32 3		; <ptr>:175 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 116, i32 1		; <ptr>:176 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 116, i32 2		; <ptr>:177 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 116, i32 3		; <ptr>:178 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 115, i32 1		; <ptr>:179 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 115, i32 2		; <ptr>:180 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 115, i32 3		; <ptr>:181 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 114, i32 1		; <ptr>:182 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 114, i32 2		; <ptr>:183 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 114, i32 3		; <ptr>:184 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 113, i32 1		; <ptr>:185 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 113, i32 2		; <ptr>:186 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 113, i32 3		; <ptr>:187 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 112, i32 1		; <ptr>:188 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 112, i32 2		; <ptr>:189 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 112, i32 3		; <ptr>:190 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 111, i32 1		; <ptr>:191 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 111, i32 2		; <ptr>:192 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 111, i32 3		; <ptr>:193 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 110, i32 1		; <ptr>:194 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 110, i32 2		; <ptr>:195 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 110, i32 3		; <ptr>:196 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 109, i32 1		; <ptr>:197 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 109, i32 2		; <ptr>:198 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 109, i32 3		; <ptr>:199 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 108, i32 1		; <ptr>:200 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 108, i32 2		; <ptr>:201 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 108, i32 3		; <ptr>:202 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 107, i32 1		; <ptr>:203 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 107, i32 2		; <ptr>:204 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 107, i32 3		; <ptr>:205 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 106, i32 1		; <ptr>:206 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 106, i32 2		; <ptr>:207 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 106, i32 3		; <ptr>:208 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 105, i32 1		; <ptr>:209 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 105, i32 2		; <ptr>:210 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 105, i32 3		; <ptr>:211 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 104, i32 1		; <ptr>:212 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 104, i32 2		; <ptr>:213 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 104, i32 3		; <ptr>:214 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 103, i32 1		; <ptr>:215 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 103, i32 2		; <ptr>:216 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 103, i32 3		; <ptr>:217 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 102, i32 1		; <ptr>:218 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 102, i32 2		; <ptr>:219 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 102, i32 3		; <ptr>:220 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 101, i32 1		; <ptr>:221 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 101, i32 2		; <ptr>:222 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 101, i32 3		; <ptr>:223 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 100, i32 1		; <ptr>:224 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 100, i32 2		; <ptr>:225 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 100, i32 3		; <ptr>:226 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 99, i32 1		; <ptr>:227 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 99, i32 2		; <ptr>:228 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 99, i32 3		; <ptr>:229 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 98, i32 1		; <ptr>:230 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 98, i32 2		; <ptr>:231 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 98, i32 3		; <ptr>:232 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 97, i32 1		; <ptr>:233 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 97, i32 2		; <ptr>:234 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 97, i32 3		; <ptr>:235 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 96, i32 1		; <ptr>:236 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 96, i32 2		; <ptr>:237 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 96, i32 3		; <ptr>:238 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 95, i32 1		; <ptr>:239 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 95, i32 2		; <ptr>:240 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 95, i32 3		; <ptr>:241 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 94, i32 1		; <ptr>:242 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 94, i32 2		; <ptr>:243 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 94, i32 3		; <ptr>:244 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 93, i32 1		; <ptr>:245 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 93, i32 2		; <ptr>:246 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 93, i32 3		; <ptr>:247 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 92, i32 1		; <ptr>:248 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 92, i32 2		; <ptr>:249 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 92, i32 3		; <ptr>:250 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 91, i32 1		; <ptr>:251 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 91, i32 2		; <ptr>:252 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 91, i32 3		; <ptr>:253 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 90, i32 1		; <ptr>:254 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 90, i32 2		; <ptr>:255 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 90, i32 3		; <ptr>:256 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 89, i32 1		; <ptr>:257 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 89, i32 2		; <ptr>:258 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 89, i32 3		; <ptr>:259 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 88, i32 1		; <ptr>:260 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 88, i32 2		; <ptr>:261 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 88, i32 3		; <ptr>:262 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 87, i32 1		; <ptr>:263 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 87, i32 2		; <ptr>:264 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 87, i32 3		; <ptr>:265 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 86, i32 1		; <ptr>:266 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 86, i32 2		; <ptr>:267 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 86, i32 3		; <ptr>:268 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 85, i32 1		; <ptr>:269 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 85, i32 2		; <ptr>:270 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 85, i32 3		; <ptr>:271 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 84, i32 1		; <ptr>:272 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 84, i32 2		; <ptr>:273 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 84, i32 3		; <ptr>:274 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 83, i32 1		; <ptr>:275 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 83, i32 2		; <ptr>:276 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 83, i32 3		; <ptr>:277 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 82, i32 1		; <ptr>:278 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 82, i32 2		; <ptr>:279 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 82, i32 3		; <ptr>:280 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 81, i32 1		; <ptr>:281 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 81, i32 2		; <ptr>:282 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 81, i32 3		; <ptr>:283 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 80, i32 1		; <ptr>:284 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 80, i32 2		; <ptr>:285 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 80, i32 3		; <ptr>:286 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 79, i32 1		; <ptr>:287 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 79, i32 2		; <ptr>:288 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 79, i32 3		; <ptr>:289 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 78, i32 1		; <ptr>:290 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 78, i32 2		; <ptr>:291 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 78, i32 3		; <ptr>:292 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 77, i32 1		; <ptr>:293 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 77, i32 2		; <ptr>:294 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 77, i32 3		; <ptr>:295 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 76, i32 1		; <ptr>:296 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 76, i32 2		; <ptr>:297 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 76, i32 3		; <ptr>:298 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 75, i32 1		; <ptr>:299 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 75, i32 2		; <ptr>:300 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 75, i32 3		; <ptr>:301 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 74, i32 1		; <ptr>:302 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 74, i32 2		; <ptr>:303 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 74, i32 3		; <ptr>:304 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 73, i32 1		; <ptr>:305 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 73, i32 2		; <ptr>:306 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 73, i32 3		; <ptr>:307 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 72, i32 1		; <ptr>:308 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 72, i32 2		; <ptr>:309 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 72, i32 3		; <ptr>:310 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 71, i32 1		; <ptr>:311 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 71, i32 2		; <ptr>:312 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 71, i32 3		; <ptr>:313 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 70, i32 1		; <ptr>:314 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 70, i32 2		; <ptr>:315 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 70, i32 3		; <ptr>:316 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 69, i32 1		; <ptr>:317 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 69, i32 2		; <ptr>:318 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 69, i32 3		; <ptr>:319 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 68, i32 1		; <ptr>:320 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 68, i32 2		; <ptr>:321 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 68, i32 3		; <ptr>:322 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 67, i32 1		; <ptr>:323 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 67, i32 2		; <ptr>:324 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 67, i32 3		; <ptr>:325 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 66, i32 1		; <ptr>:326 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 66, i32 2		; <ptr>:327 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 66, i32 3		; <ptr>:328 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 65, i32 1		; <ptr>:329 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 65, i32 2		; <ptr>:330 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 65, i32 3		; <ptr>:331 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 64, i32 1		; <ptr>:332 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 64, i32 2		; <ptr>:333 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 64, i32 3		; <ptr>:334 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 63, i32 1		; <ptr>:335 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 63, i32 2		; <ptr>:336 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 63, i32 3		; <ptr>:337 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 62, i32 1		; <ptr>:338 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 62, i32 2		; <ptr>:339 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 62, i32 3		; <ptr>:340 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 61, i32 1		; <ptr>:341 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 61, i32 2		; <ptr>:342 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 61, i32 3		; <ptr>:343 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 60, i32 1		; <ptr>:344 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 60, i32 2		; <ptr>:345 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 60, i32 3		; <ptr>:346 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 59, i32 1		; <ptr>:347 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 59, i32 2		; <ptr>:348 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 59, i32 3		; <ptr>:349 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 58, i32 1		; <ptr>:350 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 58, i32 2		; <ptr>:351 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 58, i32 3		; <ptr>:352 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 57, i32 1		; <ptr>:353 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 57, i32 2		; <ptr>:354 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 57, i32 3		; <ptr>:355 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 56, i32 1		; <ptr>:356 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 56, i32 2		; <ptr>:357 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 56, i32 3		; <ptr>:358 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 55, i32 1		; <ptr>:359 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 55, i32 2		; <ptr>:360 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 55, i32 3		; <ptr>:361 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 54, i32 1		; <ptr>:362 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 54, i32 2		; <ptr>:363 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 54, i32 3		; <ptr>:364 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 53, i32 1		; <ptr>:365 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 53, i32 2		; <ptr>:366 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 53, i32 3		; <ptr>:367 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 52, i32 1		; <ptr>:368 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 52, i32 2		; <ptr>:369 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 52, i32 3		; <ptr>:370 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 51, i32 1		; <ptr>:371 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 51, i32 2		; <ptr>:372 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 51, i32 3		; <ptr>:373 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 50, i32 1		; <ptr>:374 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 50, i32 2		; <ptr>:375 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 50, i32 3		; <ptr>:376 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 49, i32 1		; <ptr>:377 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 49, i32 2		; <ptr>:378 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 49, i32 3		; <ptr>:379 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 48, i32 1		; <ptr>:380 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 48, i32 2		; <ptr>:381 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 48, i32 3		; <ptr>:382 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 47, i32 1		; <ptr>:383 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 47, i32 2		; <ptr>:384 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 47, i32 3		; <ptr>:385 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 46, i32 1		; <ptr>:386 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 46, i32 2		; <ptr>:387 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 46, i32 3		; <ptr>:388 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 45, i32 1		; <ptr>:389 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 45, i32 2		; <ptr>:390 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 45, i32 3		; <ptr>:391 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 44, i32 1		; <ptr>:392 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 44, i32 2		; <ptr>:393 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 44, i32 3		; <ptr>:394 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 43, i32 1		; <ptr>:395 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 43, i32 2		; <ptr>:396 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 43, i32 3		; <ptr>:397 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 42, i32 1		; <ptr>:398 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 42, i32 2		; <ptr>:399 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 42, i32 3		; <ptr>:400 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 41, i32 1		; <ptr>:401 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 41, i32 2		; <ptr>:402 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 41, i32 3		; <ptr>:403 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 40, i32 1		; <ptr>:404 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 40, i32 2		; <ptr>:405 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 40, i32 3		; <ptr>:406 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 39, i32 1		; <ptr>:407 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 39, i32 2		; <ptr>:408 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 39, i32 3		; <ptr>:409 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 38, i32 1		; <ptr>:410 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 38, i32 2		; <ptr>:411 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 38, i32 3		; <ptr>:412 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 37, i32 1		; <ptr>:413 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 37, i32 2		; <ptr>:414 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 37, i32 3		; <ptr>:415 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 36, i32 1		; <ptr>:416 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 36, i32 2		; <ptr>:417 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 36, i32 3		; <ptr>:418 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 35, i32 1		; <ptr>:419 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 35, i32 2		; <ptr>:420 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 35, i32 3		; <ptr>:421 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 34, i32 1		; <ptr>:422 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 34, i32 2		; <ptr>:423 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 34, i32 3		; <ptr>:424 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 33, i32 1		; <ptr>:425 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 33, i32 2		; <ptr>:426 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 33, i32 3		; <ptr>:427 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 32, i32 1		; <ptr>:428 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 32, i32 2		; <ptr>:429 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 32, i32 3		; <ptr>:430 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 31, i32 1		; <ptr>:431 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 31, i32 2		; <ptr>:432 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 31, i32 3		; <ptr>:433 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 30, i32 1		; <ptr>:434 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 30, i32 2		; <ptr>:435 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 30, i32 3		; <ptr>:436 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 29, i32 1		; <ptr>:437 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 29, i32 2		; <ptr>:438 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 29, i32 3		; <ptr>:439 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 28, i32 1		; <ptr>:440 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 28, i32 2		; <ptr>:441 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 28, i32 3		; <ptr>:442 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 27, i32 1		; <ptr>:443 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 27, i32 2		; <ptr>:444 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 27, i32 3		; <ptr>:445 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 26, i32 1		; <ptr>:446 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 26, i32 2		; <ptr>:447 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 26, i32 3		; <ptr>:448 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 25, i32 1		; <ptr>:449 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 25, i32 2		; <ptr>:450 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 25, i32 3		; <ptr>:451 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 24, i32 1		; <ptr>:452 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 24, i32 2		; <ptr>:453 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 24, i32 3		; <ptr>:454 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 23, i32 1		; <ptr>:455 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 23, i32 2		; <ptr>:456 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 23, i32 3		; <ptr>:457 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 22, i32 1		; <ptr>:458 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 22, i32 2		; <ptr>:459 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 22, i32 3		; <ptr>:460 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 21, i32 1		; <ptr>:461 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 21, i32 2		; <ptr>:462 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 21, i32 3		; <ptr>:463 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 20, i32 1		; <ptr>:464 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 20, i32 2		; <ptr>:465 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 20, i32 3		; <ptr>:466 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 19, i32 1		; <ptr>:467 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 19, i32 2		; <ptr>:468 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 19, i32 3		; <ptr>:469 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 18, i32 1		; <ptr>:470 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 18, i32 2		; <ptr>:471 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 18, i32 3		; <ptr>:472 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 17, i32 1		; <ptr>:473 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 17, i32 2		; <ptr>:474 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 17, i32 3		; <ptr>:475 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 16, i32 1		; <ptr>:476 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 16, i32 2		; <ptr>:477 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 16, i32 3		; <ptr>:478 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 15, i32 1		; <ptr>:479 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 15, i32 2		; <ptr>:480 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 15, i32 3		; <ptr>:481 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 14, i32 1		; <ptr>:482 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 14, i32 2		; <ptr>:483 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 14, i32 3		; <ptr>:484 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 1		; <ptr>:485 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 2		; <ptr>:486 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 3		; <ptr>:487 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 3, i32 1		; <ptr>:488 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 3, i32 2		; <ptr>:489 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 3, i32 3		; <ptr>:490 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 180, i32 1		; <ptr>:491 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 180, i32 2		; <ptr>:492 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 180, i32 3		; <ptr>:493 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 181, i32 1		; <ptr>:494 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 181, i32 2		; <ptr>:495 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 181, i32 3		; <ptr>:496 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 182, i32 1		; <ptr>:497 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 182, i32 2		; <ptr>:498 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 182, i32 3		; <ptr>:499 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 183, i32 1		; <ptr>:500 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 183, i32 2		; <ptr>:501 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 183, i32 3		; <ptr>:502 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 184, i32 1		; <ptr>:503 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 184, i32 2		; <ptr>:504 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 184, i32 3		; <ptr>:505 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 185, i32 1		; <ptr>:506 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 185, i32 2		; <ptr>:507 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 185, i32 3		; <ptr>:508 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 190, i32 1		; <ptr>:509 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 190, i32 2		; <ptr>:510 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 190, i32 3		; <ptr>:511 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 9, i32 1		; <ptr>:512 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 9, i32 2		; <ptr>:513 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 9, i32 3		; <ptr>:514 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 10, i32 1		; <ptr>:515 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 10, i32 2		; <ptr>:516 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 10, i32 3		; <ptr>:517 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 11, i32 1		; <ptr>:518 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 11, i32 2		; <ptr>:519 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 11, i32 3		; <ptr>:520 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 12, i32 1		; <ptr>:521 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 12, i32 2		; <ptr>:522 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 12, i32 3		; <ptr>:523 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 13, i32 1		; <ptr>:524 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 13, i32 2		; <ptr>:525 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 13, i32 3		; <ptr>:526 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 5, i32 1		; <ptr>:527 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 5, i32 2		; <ptr>:528 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 5, i32 3		; <ptr>:529 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 1		; <ptr>:530 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 2		; <ptr>:531 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 3		; <ptr>:532 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 2, i32 1		; <ptr>:533 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 2, i32 2		; <ptr>:534 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 2, i32 3		; <ptr>:535 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 6, i32 1		; <ptr>:536 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 6, i32 2		; <ptr>:537 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 6, i32 3		; <ptr>:538 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 7, i32 1		; <ptr>:539 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 7, i32 2		; <ptr>:540 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 7, i32 3		; <ptr>:541 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 186, i32 1		; <ptr>:542 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 186, i32 2		; <ptr>:543 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 186, i32 3		; <ptr>:544 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 187, i32 1		; <ptr>:545 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 187, i32 2		; <ptr>:546 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 187, i32 3		; <ptr>:547 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 8, i32 1		; <ptr>:548 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 8, i32 2		; <ptr>:549 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 8, i32 3		; <ptr>:550 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:551 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 188, i32 1		; <ptr>:552 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 188, i32 2		; <ptr>:553 [#uses=1]
	load <4 x float>, ptr %553		; <<4 x float>>:554 [#uses=1]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 188, i32 3		; <ptr>:555 [#uses=0]
	shufflevector <4 x float> %554, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:556 [#uses=1]
	call <4 x i32> @llvm.ppc.altivec.vcmpgtfp( <4 x float> zeroinitializer, <4 x float> %556 )		; <<4 x i32>>:557 [#uses=0]
	bitcast <4 x i32> zeroinitializer to <4 x float>		; <<4 x float>>:558 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 0		; <ptr>:559 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 2		; <ptr>:560 [#uses=1]
	store <4 x float> zeroinitializer, ptr %560
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 3		; <ptr>:561 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 1		; <ptr>:562 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 2		; <ptr>:563 [#uses=0]
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:564 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:565 [#uses=1]
	store <4 x float> %565, ptr null
	icmp eq i32 0, 0		; <i1>:566 [#uses=1]
	br i1 %566, label %.critedge, label %xPIF.exit

.critedge:		; preds = %xOperationInitMasks.exit
	getelementptr [4 x <4 x i32>], ptr null, i32 0, i32 3		; <ptr>:567 [#uses=0]
	and <4 x i32> zeroinitializer, zeroinitializer		; <<4 x i32>>:568 [#uses=0]
	or <4 x i32> zeroinitializer, zeroinitializer		; <<4 x i32>>:569 [#uses=0]
	icmp eq i32 0, 0		; <i1>:570 [#uses=1]
	br i1 %570, label %.critedge7898, label %xPBRK.exit

.critedge7898:		; preds = %.critedge
	br label %xPIF.exit

xPIF.exit:		; preds = %.critedge7898, %xOperationInitMasks.exit
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 188, i32 1		; <ptr>:571 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:572 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:573 [#uses=0]
	icmp eq i32 0, 0		; <i1>:574 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 3, i32 1		; <ptr>:575 [#uses=0]
	load <4 x float>, ptr %0		; <<4 x float>>:576 [#uses=0]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:577 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 2, i32 0		; <ptr>:578 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 2, i32 1		; <ptr>:579 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 2, i32 2		; <ptr>:580 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 2, i32 3		; <ptr>:581 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 3		; <ptr>:582 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:583 [#uses=1]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 1		; <ptr>:584 [#uses=1]
	load <4 x float>, ptr %584		; <<4 x float>>:585 [#uses=1]
	load <4 x float>, ptr null		; <<4 x float>>:586 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 3		; <ptr>:587 [#uses=1]
	load <4 x float>, ptr %587		; <<4 x float>>:588 [#uses=1]
	shufflevector <4 x float> %583, <4 x float> undef, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >		; <<4 x float>>:589 [#uses=1]
	shufflevector <4 x float> %585, <4 x float> undef, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >		; <<4 x float>>:590 [#uses=1]
	shufflevector <4 x float> %588, <4 x float> undef, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >		; <<4 x float>>:591 [#uses=1]
	fmul <4 x float> zeroinitializer, %589		; <<4 x float>>:592 [#uses=0]
	fmul <4 x float> zeroinitializer, %590		; <<4 x float>>:593 [#uses=0]
	fmul <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:594 [#uses=1]
	fmul <4 x float> zeroinitializer, %591		; <<4 x float>>:595 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 0		; <ptr>:596 [#uses=2]
	load <4 x float>, ptr %596		; <<4 x float>>:597 [#uses=0]
	store <4 x float> zeroinitializer, ptr %596
	load <4 x float>, ptr null		; <<4 x float>>:598 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 2		; <ptr>:599 [#uses=0]
	shufflevector <4 x float> %594, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 1, i32 2, i32 7 >		; <<4 x float>>:600 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 3		; <ptr>:601 [#uses=2]
	load <4 x float>, ptr %601		; <<4 x float>>:602 [#uses=0]
	store <4 x float> zeroinitializer, ptr %601
	load <4 x float>, ptr null		; <<4 x float>>:603 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:604 [#uses=1]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 2		; <ptr>:605 [#uses=1]
	load <4 x float>, ptr %605		; <<4 x float>>:606 [#uses=1]
	fsub <4 x float> zeroinitializer, %604		; <<4 x float>>:607 [#uses=2]
	fsub <4 x float> zeroinitializer, %606		; <<4 x float>>:608 [#uses=2]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:609 [#uses=0]
	br i1 false, label %617, label %610

; <label>:610		; preds = %xPIF.exit
	load <4 x float>, ptr null		; <<4 x float>>:611 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 1		; <ptr>:612 [#uses=2]
	load <4 x float>, ptr %612		; <<4 x float>>:613 [#uses=1]
	shufflevector <4 x float> %607, <4 x float> %613, <4 x i32> < i32 0, i32 1, i32 2, i32 7 >		; <<4 x float>>:614 [#uses=1]
	store <4 x float> %614, ptr %612
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 3		; <ptr>:615 [#uses=2]
	load <4 x float>, ptr %615		; <<4 x float>>:616 [#uses=0]
	store <4 x float> zeroinitializer, ptr %615
	br label %xST.exit400

; <label>:617		; preds = %xPIF.exit
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:618 [#uses=0]
	shufflevector <4 x i32> zeroinitializer, <4 x i32> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >		; <<4 x i32>>:619 [#uses=1]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> %619, <4 x i32> zeroinitializer )		; <i32>:620 [#uses=1]
	icmp eq i32 %620, 0		; <i1>:621 [#uses=1]
	br i1 %621, label %625, label %622

; <label>:622		; preds = %617
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 1		; <ptr>:623 [#uses=0]
	shufflevector <4 x float> %607, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 1, i32 2, i32 7 >		; <<4 x float>>:624 [#uses=0]
	br label %625

; <label>:625		; preds = %622, %617
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:626 [#uses=0]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:627 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 2		; <ptr>:628 [#uses=1]
	load <4 x float>, ptr %628		; <<4 x float>>:629 [#uses=0]
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:630 [#uses=0]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:631 [#uses=1]
	icmp eq i32 %631, 0		; <i1>:632 [#uses=1]
	br i1 %632, label %xST.exit400, label %633

; <label>:633		; preds = %625
	load <4 x float>, ptr null		; <<4 x float>>:634 [#uses=1]
	shufflevector <4 x float> zeroinitializer, <4 x float> %634, <4 x i32> < i32 0, i32 1, i32 2, i32 7 >		; <<4 x float>>:635 [#uses=1]
	store <4 x float> %635, ptr null
	br label %xST.exit400

xST.exit400:		; preds = %633, %625, %610
	%.17218 = phi <4 x float> [ zeroinitializer, %610 ], [ %608, %633 ], [ %608, %625 ]		; <<4 x float>> [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 0		; <ptr>:636 [#uses=1]
	load <4 x float>, ptr %636		; <<4 x float>>:637 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:638 [#uses=2]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 2		; <ptr>:639 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:640 [#uses=2]
	fmul <4 x float> %638, %638		; <<4 x float>>:641 [#uses=1]
	fmul <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:642 [#uses=0]
	fmul <4 x float> %640, %640		; <<4 x float>>:643 [#uses=2]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >		; <<4 x float>>:644 [#uses=0]
	shufflevector <4 x float> %643, <4 x float> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >		; <<4 x float>>:645 [#uses=1]
	fadd <4 x float> %645, %643		; <<4 x float>>:646 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> < i32 2, i32 2, i32 2, i32 2 >		; <<4 x float>>:647 [#uses=1]
	shufflevector <4 x float> %641, <4 x float> undef, <4 x i32> < i32 2, i32 2, i32 2, i32 2 >		; <<4 x float>>:648 [#uses=1]
	fadd <4 x float> zeroinitializer, %647		; <<4 x float>>:649 [#uses=2]
	fadd <4 x float> zeroinitializer, %648		; <<4 x float>>:650 [#uses=0]
	fadd <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:651 [#uses=2]
	call <4 x float> @llvm.ppc.altivec.vrsqrtefp( <4 x float> %649 )		; <<4 x float>>:652 [#uses=1]
	fmul <4 x float> %652, %649		; <<4 x float>>:653 [#uses=1]
	call <4 x float> @llvm.ppc.altivec.vrsqrtefp( <4 x float> %651 )		; <<4 x float>>:654 [#uses=1]
	fmul <4 x float> %654, %651		; <<4 x float>>:655 [#uses=0]
	icmp eq i32 0, 0		; <i1>:656 [#uses=1]
	br i1 %656, label %665, label %657

; <label>:657		; preds = %xST.exit400
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 0		; <ptr>:658 [#uses=0]
	shufflevector <4 x float> %653, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:659 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 1		; <ptr>:660 [#uses=1]
	load <4 x float>, ptr %660		; <<4 x float>>:661 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 2		; <ptr>:662 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 3		; <ptr>:663 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:664 [#uses=0]
	br label %xST.exit402

; <label>:665		; preds = %xST.exit400
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:666 [#uses=0]
	br i1 false, label %669, label %667

; <label>:667		; preds = %665
	load <4 x float>, ptr null		; <<4 x float>>:668 [#uses=0]
	br label %669

; <label>:669		; preds = %667, %665
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:670 [#uses=0]
	br label %xST.exit402

xST.exit402:		; preds = %669, %657
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 0		; <ptr>:671 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:672 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 2		; <ptr>:673 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 1		; <ptr>:674 [#uses=1]
	load <4 x float>, ptr %674		; <<4 x float>>:675 [#uses=1]
	load <4 x float>, ptr null		; <<4 x float>>:676 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:677 [#uses=1]
	shufflevector <4 x float> %675, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:678 [#uses=1]
	fmul <4 x float> zeroinitializer, %677		; <<4 x float>>:679 [#uses=0]
	fmul <4 x float> zeroinitializer, %678		; <<4 x float>>:680 [#uses=0]
	fmul <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:681 [#uses=1]
	icmp eq i32 0, 0		; <i1>:682 [#uses=1]
	br i1 %682, label %689, label %683

; <label>:683		; preds = %xST.exit402
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 5, i32 1		; <ptr>:684 [#uses=1]
	load <4 x float>, ptr %684		; <<4 x float>>:685 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 5, i32 2		; <ptr>:686 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 5, i32 3		; <ptr>:687 [#uses=0]
	shufflevector <4 x float> %681, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:688 [#uses=0]
	br label %xST.exit405

; <label>:689		; preds = %xST.exit402
	shufflevector <4 x i32> zeroinitializer, <4 x i32> undef, <4 x i32> zeroinitializer		; <<4 x i32>>:690 [#uses=0]
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:691 [#uses=1]
	shufflevector <4 x i32> %691, <4 x i32> undef, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >		; <<4 x i32>>:692 [#uses=1]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> %692, <4 x i32> zeroinitializer )		; <i32>:693 [#uses=1]
	icmp eq i32 %693, 0		; <i1>:694 [#uses=0]
	br label %xST.exit405

xST.exit405:		; preds = %689, %683
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 3		; <ptr>:695 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:696 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:697 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:698 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 5, i32 2		; <ptr>:699 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:700 [#uses=1]
	fadd <4 x float> zeroinitializer, %700		; <<4 x float>>:701 [#uses=0]
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:702 [#uses=1]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> %702, <4 x i32> zeroinitializer )		; <i32>:703 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 1		; <ptr>:704 [#uses=2]
	load <4 x float>, ptr %704		; <<4 x float>>:705 [#uses=0]
	store <4 x float> zeroinitializer, ptr %704
	load <4 x float>, ptr null		; <<4 x float>>:706 [#uses=0]
	store <4 x float> zeroinitializer, ptr null
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 3		; <ptr>:707 [#uses=2]
	load <4 x float>, ptr %707		; <<4 x float>>:708 [#uses=0]
	store <4 x float> zeroinitializer, ptr %707
	load <4 x float>, ptr null		; <<4 x float>>:709 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:710 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:711 [#uses=1]
	shufflevector <4 x float> %711, <4 x float> undef, <4 x i32> < i32 2, i32 2, i32 2, i32 2 >		; <<4 x float>>:712 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 1		; <ptr>:713 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 2		; <ptr>:714 [#uses=1]
	load <4 x float>, ptr %714		; <<4 x float>>:715 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:716 [#uses=0]
	fmul <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:717 [#uses=1]
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:718 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 5, i32 0		; <ptr>:719 [#uses=1]
	store <4 x float> zeroinitializer, ptr %719
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 5, i32 1		; <ptr>:720 [#uses=1]
	shufflevector <4 x float> %717, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:721 [#uses=1]
	store <4 x float> %721, ptr %720
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 5, i32 2		; <ptr>:722 [#uses=1]
	load <4 x float>, ptr %722		; <<4 x float>>:723 [#uses=1]
	shufflevector <4 x float> zeroinitializer, <4 x float> %723, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:724 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 5, i32 3		; <ptr>:725 [#uses=1]
	store <4 x float> zeroinitializer, ptr %725
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 5, i32 2		; <ptr>:726 [#uses=1]
	load <4 x float>, ptr %726		; <<4 x float>>:727 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 5, i32 3		; <ptr>:728 [#uses=1]
	load <4 x float>, ptr %728		; <<4 x float>>:729 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 0		; <ptr>:730 [#uses=1]
	load <4 x float>, ptr %730		; <<4 x float>>:731 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 1		; <ptr>:732 [#uses=1]
	load <4 x float>, ptr %732		; <<4 x float>>:733 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 3		; <ptr>:734 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:735 [#uses=1]
	fmul <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:736 [#uses=1]
	fmul <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:737 [#uses=1]
	fmul <4 x float> zeroinitializer, %735		; <<4 x float>>:738 [#uses=1]
	fmul <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:739 [#uses=1]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:740 [#uses=1]
	icmp eq i32 %740, 0		; <i1>:741 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 0		; <ptr>:742 [#uses=2]
	load <4 x float>, ptr %742		; <<4 x float>>:743 [#uses=1]
	shufflevector <4 x float> %736, <4 x float> %743, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:744 [#uses=1]
	store <4 x float> %744, ptr %742
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 1		; <ptr>:745 [#uses=1]
	load <4 x float>, ptr %745		; <<4 x float>>:746 [#uses=1]
	shufflevector <4 x float> %737, <4 x float> %746, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:747 [#uses=0]
	shufflevector <4 x float> %738, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:748 [#uses=1]
	store <4 x float> %748, ptr null
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 3		; <ptr>:749 [#uses=1]
	load <4 x float>, ptr %749		; <<4 x float>>:750 [#uses=1]
	shufflevector <4 x float> %739, <4 x float> %750, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:751 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 0		; <ptr>:752 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 1		; <ptr>:753 [#uses=1]
	load <4 x float>, ptr %753		; <<4 x float>>:754 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 2		; <ptr>:755 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:756 [#uses=1]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:757 [#uses=1]
	shufflevector <4 x float> %756, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:758 [#uses=1]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 2		; <ptr>:759 [#uses=1]
	load <4 x float>, ptr %759		; <<4 x float>>:760 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 3		; <ptr>:761 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:762 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:763 [#uses=1]
	fadd <4 x float> %757, zeroinitializer		; <<4 x float>>:764 [#uses=0]
	fadd <4 x float> %758, %763		; <<4 x float>>:765 [#uses=0]
	fmul <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:766 [#uses=1]
	br i1 false, label %773, label %767

; <label>:767		; preds = %xST.exit405
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 1		; <ptr>:768 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:769 [#uses=1]
	shufflevector <4 x float> zeroinitializer, <4 x float> %769, <4 x i32> < i32 0, i32 1, i32 2, i32 7 >		; <<4 x float>>:770 [#uses=1]
	store <4 x float> %770, ptr null
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 3		; <ptr>:771 [#uses=1]
	load <4 x float>, ptr %771		; <<4 x float>>:772 [#uses=0]
	br label %xST.exit422

; <label>:773		; preds = %xST.exit405
	br label %xST.exit422

xST.exit422:		; preds = %773, %767
	%.07267 = phi <4 x float> [ %766, %767 ], [ undef, %773 ]		; <<4 x float>> [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 3		; <ptr>:774 [#uses=0]
	fmul <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:775 [#uses=0]
	icmp eq i32 0, 0		; <i1>:776 [#uses=1]
	br i1 %776, label %780, label %777

; <label>:777		; preds = %xST.exit422
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 2		; <ptr>:778 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 3		; <ptr>:779 [#uses=0]
	br label %xST.exit431

; <label>:780		; preds = %xST.exit422
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:781 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 2		; <ptr>:782 [#uses=2]
	load <4 x float>, ptr %782		; <<4 x float>>:783 [#uses=0]
	store <4 x float> zeroinitializer, ptr %782
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:784 [#uses=1]
	shufflevector <4 x i32> %784, <4 x i32> undef, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >		; <<4 x i32>>:785 [#uses=0]
	icmp eq i32 0, 0		; <i1>:786 [#uses=0]
	br label %xST.exit431

xST.exit431:		; preds = %780, %777
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 2		; <ptr>:787 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:788 [#uses=0]
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:789 [#uses=2]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> %789, <4 x i32> zeroinitializer )		; <i32>:790 [#uses=1]
	icmp eq i32 %790, 0		; <i1>:791 [#uses=0]
	shufflevector <4 x i32> %789, <4 x i32> undef, <4 x i32> zeroinitializer		; <<4 x i32>>:792 [#uses=1]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> %792, <4 x i32> zeroinitializer )		; <i32>:793 [#uses=1]
	icmp eq i32 %793, 0		; <i1>:794 [#uses=1]
	br i1 %794, label %797, label %795

; <label>:795		; preds = %xST.exit431
	load <4 x float>, ptr null		; <<4 x float>>:796 [#uses=0]
	store <4 x float> zeroinitializer, ptr null
	br label %797

; <label>:797		; preds = %795, %xST.exit431
	%.07332 = phi <4 x float> [ zeroinitializer, %795 ], [ undef, %xST.exit431 ]		; <<4 x float>> [#uses=0]
	shufflevector <4 x i32> zeroinitializer, <4 x i32> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >		; <<4 x i32>>:798 [#uses=0]
	br i1 false, label %xST.exit434, label %799

; <label>:799		; preds = %797
	load <4 x float>, ptr null		; <<4 x float>>:800 [#uses=0]
	store <4 x float> zeroinitializer, ptr null
	br label %xST.exit434

xST.exit434:		; preds = %799, %797
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:801 [#uses=1]
	shufflevector <4 x i32> %801, <4 x i32> undef, <4 x i32> < i32 2, i32 2, i32 2, i32 2 >		; <<4 x i32>>:802 [#uses=0]
	shufflevector <4 x i32> zeroinitializer, <4 x i32> undef, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >		; <<4 x i32>>:803 [#uses=0]
	icmp eq i32 0, 0		; <i1>:804 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 0		; <ptr>:805 [#uses=1]
	load <4 x float>, ptr %805		; <<4 x float>>:806 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 1		; <ptr>:807 [#uses=1]
	load <4 x float>, ptr %807		; <<4 x float>>:808 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:809 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:810 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 0		; <ptr>:811 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 2		; <ptr>:812 [#uses=1]
	load <4 x float>, ptr %812		; <<4 x float>>:813 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 3		; <ptr>:814 [#uses=1]
	load <4 x float>, ptr %814		; <<4 x float>>:815 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:816 [#uses=0]
	unreachable

xPBRK.exit:		; preds = %.critedge
	store <4 x i32> < i32 -1, i32 -1, i32 -1, i32 -1 >, ptr %.sub7896
	store <4 x i32> zeroinitializer, ptr null
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 186, i32 1		; <ptr>:817 [#uses=1]
	load <4 x float>, ptr %817		; <<4 x float>>:818 [#uses=1]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 186, i32 2		; <ptr>:819 [#uses=1]
	load <4 x float>, ptr %819		; <<4 x float>>:820 [#uses=1]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 186, i32 3		; <ptr>:821 [#uses=1]
	load <4 x float>, ptr %821		; <<4 x float>>:822 [#uses=1]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:823 [#uses=1]
	shufflevector <4 x float> %818, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:824 [#uses=1]
	shufflevector <4 x float> %820, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:825 [#uses=1]
	shufflevector <4 x float> %822, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:826 [#uses=1]
	shufflevector <4 x float> %823, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:827 [#uses=0]
	shufflevector <4 x float> %824, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:828 [#uses=1]
	store <4 x float> %828, ptr null
	load <4 x float>, ptr null		; <<4 x float>>:829 [#uses=1]
	shufflevector <4 x float> %825, <4 x float> %829, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:830 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 186, i32 3		; <ptr>:831 [#uses=2]
	load <4 x float>, ptr %831		; <<4 x float>>:832 [#uses=1]
	shufflevector <4 x float> %826, <4 x float> %832, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:833 [#uses=1]
	store <4 x float> %833, ptr %831
	br label %xLS.exit449

xLS.exit449:		; preds = %1215, %xPBRK.exit
	%.27464 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.17463, %1215 ]		; <<4 x float>> [#uses=2]
	%.27469 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.17468, %1215 ]		; <<4 x float>> [#uses=2]
	%.27474 = phi <4 x float> [ undef, %xPBRK.exit ], [ zeroinitializer, %1215 ]		; <<4 x float>> [#uses=1]
	%.17482 = phi <4 x float> [ undef, %xPBRK.exit ], [ zeroinitializer, %1215 ]		; <<4 x float>> [#uses=0]
	%.17486 = phi <4 x float> [ undef, %xPBRK.exit ], [ zeroinitializer, %1215 ]		; <<4 x float>> [#uses=0]
	%.17490 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.07489, %1215 ]		; <<4 x float>> [#uses=2]
	%.17494 = phi <4 x float> [ undef, %xPBRK.exit ], [ zeroinitializer, %1215 ]		; <<4 x float>> [#uses=0]
	%.27504 = phi <4 x float> [ undef, %xPBRK.exit ], [ zeroinitializer, %1215 ]		; <<4 x float>> [#uses=0]
	%.17513 = phi <4 x float> [ undef, %xPBRK.exit ], [ zeroinitializer, %1215 ]		; <<4 x float>> [#uses=0]
	%.17517 = phi <4 x float> [ undef, %xPBRK.exit ], [ zeroinitializer, %1215 ]		; <<4 x float>> [#uses=0]
	%.17552 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.07551, %1215 ]		; <<4 x float>> [#uses=2]
	%.17556 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.07555, %1215 ]		; <<4 x float>> [#uses=2]
	%.17560 = phi <4 x float> [ undef, %xPBRK.exit ], [ zeroinitializer, %1215 ]		; <<4 x float>> [#uses=0]
	%.17583 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.07582, %1215 ]		; <<4 x float>> [#uses=2]
	%.17591 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.07590, %1215 ]		; <<4 x float>> [#uses=2]
	%.17599 = phi <4 x float> [ undef, %xPBRK.exit ], [ zeroinitializer, %1215 ]		; <<4 x float>> [#uses=0]
	%.17618 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.07617, %1215 ]		; <<4 x float>> [#uses=2]
	%.17622 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.07621, %1215 ]		; <<4 x float>> [#uses=2]
	%.17626 = phi <4 x float> [ undef, %xPBRK.exit ], [ zeroinitializer, %1215 ]		; <<4 x float>> [#uses=0]
	%.17653 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.07652, %1215 ]		; <<4 x float>> [#uses=2]
	%.17657 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.07656, %1215 ]		; <<4 x float>> [#uses=2]
	%.17661 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.07660, %1215 ]		; <<4 x float>> [#uses=2]
	%.17665 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.07664, %1215 ]		; <<4 x float>> [#uses=2]
	%.17723 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.07722, %1215 ]		; <<4 x float>> [#uses=2]
	%.17727 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.07726, %1215 ]		; <<4 x float>> [#uses=2]
	%.17731 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.07730, %1215 ]		; <<4 x float>> [#uses=2]
	%.17735 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.07734, %1215 ]		; <<4 x float>> [#uses=2]
	%.17770 = phi <4 x float> [ undef, %xPBRK.exit ], [ %.07769, %1215 ]		; <<4 x float>> [#uses=2]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 186, i32 0		; <ptr>:834 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:835 [#uses=1]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 186, i32 2		; <ptr>:836 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 186, i32 3		; <ptr>:837 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:838 [#uses=0]
	shufflevector <4 x float> %835, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:839 [#uses=1]
	getelementptr <4 x float>, ptr null, i32 878		; <ptr>:840 [#uses=1]
	load <4 x float>, ptr %840		; <<4 x float>>:841 [#uses=0]
	call <4 x float> @llvm.ppc.altivec.vcfsx( <4 x i32> zeroinitializer, i32 0 )		; <<4 x float>>:842 [#uses=1]
	shufflevector <4 x float> %842, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:843 [#uses=2]
	call <4 x i32> @llvm.ppc.altivec.vcmpgtfp( <4 x float> %843, <4 x float> %839 )		; <<4 x i32>>:844 [#uses=1]
	bitcast <4 x i32> %844 to <4 x float>		; <<4 x float>>:845 [#uses=1]
	call <4 x i32> @llvm.ppc.altivec.vcmpgtfp( <4 x float> %843, <4 x float> zeroinitializer )		; <<4 x i32>>:846 [#uses=0]
	bitcast <4 x i32> zeroinitializer to <4 x float>		; <<4 x float>>:847 [#uses=1]
	icmp eq i32 0, 0		; <i1>:848 [#uses=1]
	br i1 %848, label %854, label %849

; <label>:849		; preds = %xLS.exit449
	shufflevector <4 x float> zeroinitializer, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:850 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 1		; <ptr>:851 [#uses=1]
	store <4 x float> zeroinitializer, ptr %851
	shufflevector <4 x float> zeroinitializer, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:852 [#uses=1]
	store <4 x float> %852, ptr null
	shufflevector <4 x float> %847, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:853 [#uses=0]
	br label %xST.exit451

; <label>:854		; preds = %xLS.exit449
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:855 [#uses=0]
	br i1 false, label %859, label %856

; <label>:856		; preds = %854
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 0		; <ptr>:857 [#uses=2]
	load <4 x float>, ptr %857		; <<4 x float>>:858 [#uses=0]
	store <4 x float> zeroinitializer, ptr %857
	br label %859

; <label>:859		; preds = %856, %854
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:860 [#uses=0]
	br i1 false, label %864, label %861

; <label>:861		; preds = %859
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 1		; <ptr>:862 [#uses=1]
	shufflevector <4 x float> %845, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:863 [#uses=1]
	store <4 x float> %863, ptr %862
	br label %864

; <label>:864		; preds = %861, %859
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:865 [#uses=1]
	shufflevector <4 x i32> %865, <4 x i32> undef, <4 x i32> < i32 2, i32 2, i32 2, i32 2 >		; <<4 x i32>>:866 [#uses=0]
	br i1 false, label %868, label %867

; <label>:867		; preds = %864
	store <4 x float> zeroinitializer, ptr null
	br label %868

; <label>:868		; preds = %867, %864
	shufflevector <4 x i32> zeroinitializer, <4 x i32> undef, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >		; <<4 x i32>>:869 [#uses=0]
	br label %xST.exit451

xST.exit451:		; preds = %868, %849
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 0		; <ptr>:870 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 1		; <ptr>:871 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:872 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:873 [#uses=1]
	bitcast <4 x float> zeroinitializer to <4 x i32>		; <<4 x i32>>:874 [#uses=1]
	xor <4 x i32> %874, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>>:875 [#uses=0]
	bitcast <4 x float> %873 to <4 x i32>		; <<4 x i32>>:876 [#uses=1]
	xor <4 x i32> %876, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>>:877 [#uses=0]
	bitcast <4 x float> zeroinitializer to <4 x i32>		; <<4 x i32>>:878 [#uses=1]
	xor <4 x i32> %878, < i32 -1, i32 -1, i32 -1, i32 -1 >		; <<4 x i32>>:879 [#uses=1]
	bitcast <4 x i32> %879 to <4 x float>		; <<4 x float>>:880 [#uses=0]
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:881 [#uses=1]
	icmp eq i32 0, 0		; <i1>:882 [#uses=1]
	br i1 %882, label %888, label %883

; <label>:883		; preds = %xST.exit451
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 0		; <ptr>:884 [#uses=1]
	store <4 x float> zeroinitializer, ptr %884
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 1		; <ptr>:885 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:886 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 3		; <ptr>:887 [#uses=0]
	br label %xST.exit453

; <label>:888		; preds = %xST.exit451
	shufflevector <4 x i32> %881, <4 x i32> undef, <4 x i32> zeroinitializer		; <<4 x i32>>:889 [#uses=0]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:890 [#uses=0]
	br i1 false, label %894, label %891

; <label>:891		; preds = %888
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 1		; <ptr>:892 [#uses=1]
	shufflevector <4 x float> zeroinitializer, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:893 [#uses=1]
	store <4 x float> %893, ptr %892
	br label %894

; <label>:894		; preds = %891, %888
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:895 [#uses=1]
	icmp eq i32 %895, 0		; <i1>:896 [#uses=1]
	br i1 %896, label %898, label %897

; <label>:897		; preds = %894
	br label %898

; <label>:898		; preds = %897, %894
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:899 [#uses=0]
	br i1 false, label %xST.exit453, label %900

; <label>:900		; preds = %898
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 3		; <ptr>:901 [#uses=1]
	load <4 x float>, ptr %901		; <<4 x float>>:902 [#uses=1]
	shufflevector <4 x float> zeroinitializer, <4 x float> %902, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:903 [#uses=0]
	br label %xST.exit453

xST.exit453:		; preds = %900, %898, %883
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 1		; <ptr>:904 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:905 [#uses=1]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 189, i32 3		; <ptr>:906 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:907 [#uses=1]
	shufflevector <4 x float> %905, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:908 [#uses=1]
	bitcast <4 x float> zeroinitializer to <4 x i32>		; <<4 x i32>>:909 [#uses=0]
	bitcast <4 x float> %908 to <4 x i32>		; <<4 x i32>>:910 [#uses=0]
	bitcast <4 x float> %907 to <4 x i32>		; <<4 x i32>>:911 [#uses=0]
	bitcast <4 x float> zeroinitializer to <4 x i32>		; <<4 x i32>>:912 [#uses=0]
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:913 [#uses=0]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 2, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:914 [#uses=0]
	br i1 false, label %915, label %xPIF.exit455

; <label>:915		; preds = %xST.exit453
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:916 [#uses=0]
	getelementptr [4 x <4 x i32>], ptr null, i32 0, i32 3		; <ptr>:917 [#uses=1]
	store <4 x i32> zeroinitializer, ptr %917
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:918 [#uses=1]
	and <4 x i32> %918, zeroinitializer		; <<4 x i32>>:919 [#uses=0]
	br label %.critedge7899

.critedge7899:		; preds = %.critedge7899, %915
	or <4 x i32> zeroinitializer, zeroinitializer		; <<4 x i32>>:920 [#uses=1]
	br i1 false, label %.critedge7899, label %xPBRK.exit456

xPBRK.exit456:		; preds = %.critedge7899
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 2, <4 x i32> %920, <4 x i32> zeroinitializer )		; <i32>:921 [#uses=0]
	unreachable

xPIF.exit455:		; preds = %xST.exit453
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 186, i32 0		; <ptr>:922 [#uses=1]
	load <4 x float>, ptr %922		; <<4 x float>>:923 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 186, i32 1		; <ptr>:924 [#uses=1]
	load <4 x float>, ptr %924		; <<4 x float>>:925 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 186, i32 2		; <ptr>:926 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 186, i32 3		; <ptr>:927 [#uses=0]
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:928 [#uses=0]
	bitcast ptr %1 to ptr		; <ptr>:929 [#uses=0]
	bitcast <4 x float> zeroinitializer to <4 x i32>		; <<4 x i32>>:930 [#uses=0]
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:931 [#uses=0]
	icmp eq i32 0, 0		; <i1>:932 [#uses=1]
	br i1 %932, label %934, label %933

; <label>:933		; preds = %xPIF.exit455
	store <4 x float> zeroinitializer, ptr null
	br label %934

; <label>:934		; preds = %933, %xPIF.exit455
	shufflevector <4 x i32> zeroinitializer, <4 x i32> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >		; <<4 x i32>>:935 [#uses=0]
	icmp eq i32 0, 0		; <i1>:936 [#uses=1]
	br i1 %936, label %xST.exit459, label %937

; <label>:937		; preds = %934
	br label %xST.exit459

xST.exit459:		; preds = %937, %934
	shufflevector <4 x i32> zeroinitializer, <4 x i32> undef, <4 x i32> < i32 2, i32 2, i32 2, i32 2 >		; <<4 x i32>>:938 [#uses=1]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> %938, <4 x i32> zeroinitializer )		; <i32>:939 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 188, i32 2		; <ptr>:940 [#uses=1]
	store <4 x float> zeroinitializer, ptr %940
	load <4 x float>, ptr null		; <<4 x float>>:941 [#uses=1]
	shufflevector <4 x float> zeroinitializer, <4 x float> %941, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:942 [#uses=1]
	store <4 x float> %942, ptr null
	shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:943 [#uses=0]
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:944 [#uses=0]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:945 [#uses=0]
	br i1 false, label %947, label %946

; <label>:946		; preds = %xST.exit459
	br label %947

; <label>:947		; preds = %946, %xST.exit459
	shufflevector <4 x i32> zeroinitializer, <4 x i32> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >		; <<4 x i32>>:948 [#uses=0]
	icmp eq i32 0, 0		; <i1>:949 [#uses=1]
	br i1 %949, label %952, label %950

; <label>:950		; preds = %947
	shufflevector <4 x i32> zeroinitializer, <4 x i32> undef, <4 x i32> zeroinitializer		; <<4 x i32>>:951 [#uses=1]
	call void @llvm.ppc.altivec.stvewx( <4 x i32> %951, ptr null )
	br label %952

; <label>:952		; preds = %950, %947
	br i1 false, label %955, label %953

; <label>:953		; preds = %952
	getelementptr [4 x <4 x i32>], ptr null, i32 0, i32 2		; <ptr>:954 [#uses=0]
	br label %955

; <label>:955		; preds = %953, %952
	shufflevector <4 x i32> zeroinitializer, <4 x i32> undef, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >		; <<4 x i32>>:956 [#uses=0]
	icmp eq i32 0, 0		; <i1>:957 [#uses=1]
	br i1 %957, label %xStoreDestAddressWithMask.exit461, label %958

; <label>:958		; preds = %955
	shufflevector <4 x i32> zeroinitializer, <4 x i32> undef, <4 x i32> zeroinitializer		; <<4 x i32>>:959 [#uses=1]
	call void @llvm.ppc.altivec.stvewx( <4 x i32> %959, ptr null )
	br label %xStoreDestAddressWithMask.exit461

xStoreDestAddressWithMask.exit461:		; preds = %958, %955
	load <4 x float>, ptr %0		; <<4 x float>>:960 [#uses=0]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:961 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 3, i32 0		; <ptr>:962 [#uses=0]
	br i1 false, label %968, label %xST.exit463

xST.exit463:		; preds = %xStoreDestAddressWithMask.exit461
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 3, i32 1		; <ptr>:963 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 3, i32 2		; <ptr>:964 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 3, i32 3		; <ptr>:965 [#uses=0]
	load <4 x float>, ptr %0		; <<4 x float>>:966 [#uses=3]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:967 [#uses=0]
	br i1 false, label %972, label %969

; <label>:968		; preds = %xStoreDestAddressWithMask.exit461
	unreachable

; <label>:969		; preds = %xST.exit463
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 2, i32 1		; <ptr>:970 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 2, i32 2		; <ptr>:971 [#uses=1]
	store <4 x float> %966, ptr %971
	store <4 x float> %966, ptr null
	br label %xST.exit465

; <label>:972		; preds = %xST.exit463
	call <4 x i32> @llvm.ppc.altivec.vsel( <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <<4 x i32>>:973 [#uses=0]
	store <4 x float> zeroinitializer, ptr null
	store <4 x float> zeroinitializer, ptr null
	load <4 x float>, ptr null		; <<4 x float>>:974 [#uses=0]
	bitcast <4 x float> %966 to <4 x i32>		; <<4 x i32>>:975 [#uses=1]
	call <4 x i32> @llvm.ppc.altivec.vsel( <4 x i32> zeroinitializer, <4 x i32> %975, <4 x i32> zeroinitializer )		; <<4 x i32>>:976 [#uses=1]
	bitcast <4 x i32> %976 to <4 x float>		; <<4 x float>>:977 [#uses=1]
	store <4 x float> %977, ptr null
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 2, i32 3		; <ptr>:978 [#uses=0]
	bitcast <4 x float> zeroinitializer to <4 x i32>		; <<4 x i32>>:979 [#uses=1]
	call <4 x i32> @llvm.ppc.altivec.vsel( <4 x i32> %979, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <<4 x i32>>:980 [#uses=1]
	bitcast <4 x i32> %980 to <4 x float>		; <<4 x float>>:981 [#uses=0]
	br label %xST.exit465

xST.exit465:		; preds = %972, %969
	load <4 x float>, ptr %0		; <<4 x float>>:982 [#uses=3]
	icmp eq i32 0, 0		; <i1>:983 [#uses=1]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 0		; <ptr>:984 [#uses=1]
	br i1 %983, label %989, label %985

; <label>:985		; preds = %xST.exit465
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 1		; <ptr>:986 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 2		; <ptr>:987 [#uses=1]
	store <4 x float> %982, ptr %987
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 3		; <ptr>:988 [#uses=0]
	br label %xST.exit467

; <label>:989		; preds = %xST.exit465
	bitcast <4 x float> %982 to <4 x i32>		; <<4 x i32>>:990 [#uses=0]
	shufflevector <4 x i32> zeroinitializer, <4 x i32> undef, <4 x i32> zeroinitializer		; <<4 x i32>>:991 [#uses=0]
	store <4 x float> zeroinitializer, ptr %984
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 1		; <ptr>:992 [#uses=0]
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:993 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 2		; <ptr>:994 [#uses=0]
	bitcast <4 x i32> zeroinitializer to <4 x float>		; <<4 x float>>:995 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 1, i32 3		; <ptr>:996 [#uses=0]
	bitcast <4 x float> zeroinitializer to <4 x i32>		; <<4 x i32>>:997 [#uses=1]
	bitcast <4 x float> %982 to <4 x i32>		; <<4 x i32>>:998 [#uses=1]
	shufflevector <4 x i32> zeroinitializer, <4 x i32> undef, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >		; <<4 x i32>>:999 [#uses=1]
	call <4 x i32> @llvm.ppc.altivec.vsel( <4 x i32> %997, <4 x i32> %998, <4 x i32> %999 )		; <<4 x i32>>:1000 [#uses=1]
	bitcast <4 x i32> %1000 to <4 x float>		; <<4 x float>>:1001 [#uses=0]
	br label %xST.exit467

xST.exit467:		; preds = %989, %985
	load <4 x float>, ptr %0		; <<4 x float>>:1002 [#uses=5]
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:1003 [#uses=2]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> %1003, <4 x i32> zeroinitializer )		; <i32>:1004 [#uses=0]
	br i1 false, label %1011, label %1005

; <label>:1005		; preds = %xST.exit467
	load <4 x float>, ptr null		; <<4 x float>>:1006 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 1		; <ptr>:1007 [#uses=1]
	load <4 x float>, ptr %1007		; <<4 x float>>:1008 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:1009 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 3		; <ptr>:1010 [#uses=0]
	br label %xST.exit469

; <label>:1011		; preds = %xST.exit467
	shufflevector <4 x i32> %1003, <4 x i32> undef, <4 x i32> zeroinitializer		; <<4 x i32>>:1012 [#uses=0]
	icmp eq i32 0, 0		; <i1>:1013 [#uses=1]
	br i1 %1013, label %1015, label %1014

; <label>:1014		; preds = %1011
	br label %1015

; <label>:1015		; preds = %1014, %1011
	%.07472 = phi <4 x float> [ %1002, %1014 ], [ %.27474, %1011 ]		; <<4 x float>> [#uses=0]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:1016 [#uses=1]
	icmp eq i32 %1016, 0		; <i1>:1017 [#uses=1]
	br i1 %1017, label %1021, label %1018

; <label>:1018		; preds = %1015
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 1		; <ptr>:1019 [#uses=0]
	shufflevector <4 x float> %1002, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 1, i32 2, i32 7 >		; <<4 x float>>:1020 [#uses=0]
	br label %1021

; <label>:1021		; preds = %1018, %1015
	%.07467 = phi <4 x float> [ %1002, %1018 ], [ %.27469, %1015 ]		; <<4 x float>> [#uses=2]
	icmp eq i32 0, 0		; <i1>:1022 [#uses=1]
	br i1 %1022, label %1025, label %1023

; <label>:1023		; preds = %1021
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 2		; <ptr>:1024 [#uses=1]
	store <4 x float> zeroinitializer, ptr %1024
	br label %1025

; <label>:1025		; preds = %1023, %1021
	%.07462 = phi <4 x float> [ %1002, %1023 ], [ %.27464, %1021 ]		; <<4 x float>> [#uses=2]
	icmp eq i32 0, 0		; <i1>:1026 [#uses=1]
	br i1 %1026, label %xST.exit469, label %1027

; <label>:1027		; preds = %1025
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 3		; <ptr>:1028 [#uses=0]
	br label %xST.exit469

xST.exit469:		; preds = %1027, %1025, %1005
	%.17463 = phi <4 x float> [ %.27464, %1005 ], [ %.07462, %1027 ], [ %.07462, %1025 ]		; <<4 x float>> [#uses=1]
	%.17468 = phi <4 x float> [ %.27469, %1005 ], [ %.07467, %1027 ], [ %.07467, %1025 ]		; <<4 x float>> [#uses=1]
	%.07489 = phi <4 x float> [ %1002, %1005 ], [ %.17490, %1027 ], [ %.17490, %1025 ]		; <<4 x float>> [#uses=1]
	load <4 x float>, ptr null		; <<4 x float>>:1029 [#uses=0]
	load <4 x float>, ptr null		; <<4 x float>>:1030 [#uses=0]
	fsub <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:1031 [#uses=1]
	br i1 false, label %1037, label %1032

; <label>:1032		; preds = %xST.exit469
	load <4 x float>, ptr null		; <<4 x float>>:1033 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 2		; <ptr>:1034 [#uses=1]
	load <4 x float>, ptr %1034		; <<4 x float>>:1035 [#uses=0]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 3		; <ptr>:1036 [#uses=0]
	br label %xST.exit472

; <label>:1037		; preds = %xST.exit469
	icmp eq i32 0, 0		; <i1>:1038 [#uses=1]
	br i1 %1038, label %1040, label %1039

; <label>:1039		; preds = %1037
	br label %1040

; <label>:1040		; preds = %1039, %1037
	%.07507 = phi <4 x float> [ zeroinitializer, %1039 ], [ zeroinitializer, %1037 ]		; <<4 x float>> [#uses=0]
	icmp eq i32 0, 0		; <i1>:1041 [#uses=1]
	br i1 %1041, label %1045, label %1042

; <label>:1042		; preds = %1040
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 4, i32 1		; <ptr>:1043 [#uses=1]
	load <4 x float>, ptr %1043		; <<4 x float>>:1044 [#uses=0]
	br label %1045

; <label>:1045		; preds = %1042, %1040
	br i1 false, label %1048, label %1046

; <label>:1046		; preds = %1045
	shufflevector <4 x float> %1031, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 1, i32 2, i32 7 >		; <<4 x float>>:1047 [#uses=0]
	br label %1048

; <label>:1048		; preds = %1046, %1045
	icmp eq i32 0, 0		; <i1>:1049 [#uses=1]
	br i1 %1049, label %xST.exit472, label %1050

; <label>:1050		; preds = %1048
	br label %xST.exit472

xST.exit472:		; preds = %1050, %1048, %1032
	br i1 false, label %1052, label %1051

; <label>:1051		; preds = %xST.exit472
	br label %xST.exit474

; <label>:1052		; preds = %xST.exit472
	br i1 false, label %1054, label %1053

; <label>:1053		; preds = %1052
	br label %1054

; <label>:1054		; preds = %1053, %1052
	br i1 false, label %1056, label %1055

; <label>:1055		; preds = %1054
	br label %1056

; <label>:1056		; preds = %1055, %1054
	br i1 false, label %1058, label %1057

; <label>:1057		; preds = %1056
	br label %1058

; <label>:1058		; preds = %1057, %1056
	br i1 false, label %xST.exit474, label %1059

; <label>:1059		; preds = %1058
	br label %xST.exit474

xST.exit474:		; preds = %1059, %1058, %1051
	load <4 x float>, ptr null		; <<4 x float>>:1060 [#uses=1]
	fmul <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:1061 [#uses=1]
	fmul <4 x float> %1060, zeroinitializer		; <<4 x float>>:1062 [#uses=2]
	br i1 false, label %1065, label %1063

; <label>:1063		; preds = %xST.exit474
	shufflevector <4 x float> %1062, <4 x float> zeroinitializer, <4 x i32> < i32 0, i32 1, i32 2, i32 7 >		; <<4 x float>>:1064 [#uses=1]
	store <4 x float> %1064, ptr null
	br label %xST.exit476

; <label>:1065		; preds = %xST.exit474
	br i1 false, label %1067, label %1066

; <label>:1066		; preds = %1065
	br label %1067

; <label>:1067		; preds = %1066, %1065
	shufflevector <4 x i32> zeroinitializer, <4 x i32> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >		; <<4 x i32>>:1068 [#uses=0]
	br i1 false, label %1070, label %1069

; <label>:1069		; preds = %1067
	br label %1070

; <label>:1070		; preds = %1069, %1067
	br i1 false, label %1072, label %1071

; <label>:1071		; preds = %1070
	br label %1072

; <label>:1072		; preds = %1071, %1070
	br i1 false, label %xST.exit476, label %1073

; <label>:1073		; preds = %1072
	br label %xST.exit476

xST.exit476:		; preds = %1073, %1072, %1063
	%.07551 = phi <4 x float> [ %1062, %1063 ], [ %.17552, %1073 ], [ %.17552, %1072 ]		; <<4 x float>> [#uses=1]
	%.07555 = phi <4 x float> [ %1061, %1063 ], [ %.17556, %1073 ], [ %.17556, %1072 ]		; <<4 x float>> [#uses=1]
	br i1 false, label %1075, label %1074

; <label>:1074		; preds = %xST.exit476
	br label %xST.exit479

; <label>:1075		; preds = %xST.exit476
	br i1 false, label %1077, label %1076

; <label>:1076		; preds = %1075
	br label %1077

; <label>:1077		; preds = %1076, %1075
	br i1 false, label %1079, label %1078

; <label>:1078		; preds = %1077
	br label %1079

; <label>:1079		; preds = %1078, %1077
	br i1 false, label %1081, label %1080

; <label>:1080		; preds = %1079
	br label %1081

; <label>:1081		; preds = %1080, %1079
	br i1 false, label %xST.exit479, label %1082

; <label>:1082		; preds = %1081
	br label %xST.exit479

xST.exit479:		; preds = %1082, %1081, %1074
	br i1 false, label %1084, label %1083

; <label>:1083		; preds = %xST.exit479
	br label %xST.exit482

; <label>:1084		; preds = %xST.exit479
	br i1 false, label %1086, label %1085

; <label>:1085		; preds = %1084
	br label %1086

; <label>:1086		; preds = %1085, %1084
	br i1 false, label %1088, label %1087

; <label>:1087		; preds = %1086
	br label %1088

; <label>:1088		; preds = %1087, %1086
	br i1 false, label %1090, label %1089

; <label>:1089		; preds = %1088
	br label %1090

; <label>:1090		; preds = %1089, %1088
	br i1 false, label %xST.exit482, label %1091

; <label>:1091		; preds = %1090
	br label %xST.exit482

xST.exit482:		; preds = %1091, %1090, %1083
	br i1 false, label %1093, label %1092

; <label>:1092		; preds = %xST.exit482
	br label %xST.exit486

; <label>:1093		; preds = %xST.exit482
	br i1 false, label %1095, label %1094

; <label>:1094		; preds = %1093
	br label %1095

; <label>:1095		; preds = %1094, %1093
	br i1 false, label %1097, label %1096

; <label>:1096		; preds = %1095
	br label %1097

; <label>:1097		; preds = %1096, %1095
	br i1 false, label %1099, label %1098

; <label>:1098		; preds = %1097
	br label %1099

; <label>:1099		; preds = %1098, %1097
	br i1 false, label %xST.exit486, label %1100

; <label>:1100		; preds = %1099
	br label %xST.exit486

xST.exit486:		; preds = %1100, %1099, %1092
	br i1 false, label %1102, label %1101

; <label>:1101		; preds = %xST.exit486
	br label %xST.exit489

; <label>:1102		; preds = %xST.exit486
	br i1 false, label %1104, label %1103

; <label>:1103		; preds = %1102
	br label %1104

; <label>:1104		; preds = %1103, %1102
	br i1 false, label %1106, label %1105

; <label>:1105		; preds = %1104
	br label %1106

; <label>:1106		; preds = %1105, %1104
	br i1 false, label %1108, label %1107

; <label>:1107		; preds = %1106
	br label %1108

; <label>:1108		; preds = %1107, %1106
	br i1 false, label %xST.exit489, label %1109

; <label>:1109		; preds = %1108
	br label %xST.exit489

xST.exit489:		; preds = %1109, %1108, %1101
	br i1 false, label %1111, label %1110

; <label>:1110		; preds = %xST.exit489
	br label %xST.exit492

; <label>:1111		; preds = %xST.exit489
	br i1 false, label %1113, label %1112

; <label>:1112		; preds = %1111
	br label %1113

; <label>:1113		; preds = %1112, %1111
	br i1 false, label %1115, label %1114

; <label>:1114		; preds = %1113
	br label %1115

; <label>:1115		; preds = %1114, %1113
	br i1 false, label %1117, label %1116

; <label>:1116		; preds = %1115
	br label %1117

; <label>:1117		; preds = %1116, %1115
	br i1 false, label %xST.exit492, label %1118

; <label>:1118		; preds = %1117
	br label %xST.exit492

xST.exit492:		; preds = %1118, %1117, %1110
	load <4 x float>, ptr null		; <<4 x float>>:1119 [#uses=1]
	fmul <4 x float> %1119, zeroinitializer		; <<4 x float>>:1120 [#uses=1]
	fmul <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:1121 [#uses=1]
	br i1 false, label %1123, label %1122

; <label>:1122		; preds = %xST.exit492
	br label %xST.exit495

; <label>:1123		; preds = %xST.exit492
	br i1 false, label %1125, label %1124

; <label>:1124		; preds = %1123
	br label %1125

; <label>:1125		; preds = %1124, %1123
	br i1 false, label %1127, label %1126

; <label>:1126		; preds = %1125
	br label %1127

; <label>:1127		; preds = %1126, %1125
	br i1 false, label %1129, label %1128

; <label>:1128		; preds = %1127
	br label %1129

; <label>:1129		; preds = %1128, %1127
	br i1 false, label %xST.exit495, label %1130

; <label>:1130		; preds = %1129
	br label %xST.exit495

xST.exit495:		; preds = %1130, %1129, %1122
	%.07582 = phi <4 x float> [ %1121, %1122 ], [ %.17583, %1130 ], [ %.17583, %1129 ]		; <<4 x float>> [#uses=1]
	%.07590 = phi <4 x float> [ %1120, %1122 ], [ %.17591, %1130 ], [ %.17591, %1129 ]		; <<4 x float>> [#uses=1]
	load <4 x float>, ptr null		; <<4 x float>>:1131 [#uses=1]
	fadd <4 x float> %1131, zeroinitializer		; <<4 x float>>:1132 [#uses=1]
	fadd <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:1133 [#uses=1]
	br i1 false, label %1135, label %1134

; <label>:1134		; preds = %xST.exit495
	br label %xST.exit498

; <label>:1135		; preds = %xST.exit495
	br i1 false, label %1137, label %1136

; <label>:1136		; preds = %1135
	br label %1137

; <label>:1137		; preds = %1136, %1135
	br i1 false, label %1139, label %1138

; <label>:1138		; preds = %1137
	br label %1139

; <label>:1139		; preds = %1138, %1137
	br i1 false, label %1141, label %1140

; <label>:1140		; preds = %1139
	br label %1141

; <label>:1141		; preds = %1140, %1139
	br i1 false, label %xST.exit498, label %1142

; <label>:1142		; preds = %1141
	br label %xST.exit498

xST.exit498:		; preds = %1142, %1141, %1134
	%.07617 = phi <4 x float> [ %1133, %1134 ], [ %.17618, %1142 ], [ %.17618, %1141 ]		; <<4 x float>> [#uses=1]
	%.07621 = phi <4 x float> [ %1132, %1134 ], [ %.17622, %1142 ], [ %.17622, %1141 ]		; <<4 x float>> [#uses=1]
	load <4 x float>, ptr null		; <<4 x float>>:1143 [#uses=1]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 2		; <ptr>:1144 [#uses=1]
	load <4 x float>, ptr %1144		; <<4 x float>>:1145 [#uses=1]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 3		; <ptr>:1146 [#uses=1]
	load <4 x float>, ptr %1146		; <<4 x float>>:1147 [#uses=1]
	shufflevector <4 x float> %1143, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:1148 [#uses=1]
	shufflevector <4 x float> %1145, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:1149 [#uses=1]
	shufflevector <4 x float> %1147, <4 x float> undef, <4 x i32> zeroinitializer		; <<4 x float>>:1150 [#uses=1]
	fmul <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:1151 [#uses=1]
	fmul <4 x float> zeroinitializer, %1148		; <<4 x float>>:1152 [#uses=1]
	fmul <4 x float> zeroinitializer, %1149		; <<4 x float>>:1153 [#uses=1]
	fmul <4 x float> zeroinitializer, %1150		; <<4 x float>>:1154 [#uses=1]
	br i1 false, label %1156, label %1155

; <label>:1155		; preds = %xST.exit498
	br label %xST.exit501

; <label>:1156		; preds = %xST.exit498
	br i1 false, label %1158, label %1157

; <label>:1157		; preds = %1156
	br label %1158

; <label>:1158		; preds = %1157, %1156
	br i1 false, label %1160, label %1159

; <label>:1159		; preds = %1158
	br label %1160

; <label>:1160		; preds = %1159, %1158
	br i1 false, label %1162, label %1161

; <label>:1161		; preds = %1160
	br label %1162

; <label>:1162		; preds = %1161, %1160
	br i1 false, label %xST.exit501, label %1163

; <label>:1163		; preds = %1162
	br label %xST.exit501

xST.exit501:		; preds = %1163, %1162, %1155
	%.07652 = phi <4 x float> [ %1154, %1155 ], [ %.17653, %1163 ], [ %.17653, %1162 ]		; <<4 x float>> [#uses=1]
	%.07656 = phi <4 x float> [ %1153, %1155 ], [ %.17657, %1163 ], [ %.17657, %1162 ]		; <<4 x float>> [#uses=1]
	%.07660 = phi <4 x float> [ %1152, %1155 ], [ %.17661, %1163 ], [ %.17661, %1162 ]		; <<4 x float>> [#uses=1]
	%.07664 = phi <4 x float> [ %1151, %1155 ], [ %.17665, %1163 ], [ %.17665, %1162 ]		; <<4 x float>> [#uses=1]
	load <4 x float>, ptr null		; <<4 x float>>:1164 [#uses=1]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 2		; <ptr>:1165 [#uses=1]
	load <4 x float>, ptr %1165		; <<4 x float>>:1166 [#uses=1]
	getelementptr [193 x [4 x <4 x float>]], ptr null, i32 0, i32 0, i32 3		; <ptr>:1167 [#uses=1]
	load <4 x float>, ptr %1167		; <<4 x float>>:1168 [#uses=1]
	fadd <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:1169 [#uses=1]
	fadd <4 x float> zeroinitializer, %1164		; <<4 x float>>:1170 [#uses=1]
	fadd <4 x float> zeroinitializer, %1166		; <<4 x float>>:1171 [#uses=1]
	fadd <4 x float> zeroinitializer, %1168		; <<4 x float>>:1172 [#uses=1]
	br i1 false, label %1174, label %1173

; <label>:1173		; preds = %xST.exit501
	br label %xST.exit504

; <label>:1174		; preds = %xST.exit501
	br i1 false, label %1176, label %1175

; <label>:1175		; preds = %1174
	br label %1176

; <label>:1176		; preds = %1175, %1174
	br i1 false, label %1178, label %1177

; <label>:1177		; preds = %1176
	br label %1178

; <label>:1178		; preds = %1177, %1176
	br i1 false, label %1180, label %1179

; <label>:1179		; preds = %1178
	br label %1180

; <label>:1180		; preds = %1179, %1178
	br i1 false, label %xST.exit504, label %1181

; <label>:1181		; preds = %1180
	br label %xST.exit504

xST.exit504:		; preds = %1181, %1180, %1173
	%.07722 = phi <4 x float> [ %1172, %1173 ], [ %.17723, %1181 ], [ %.17723, %1180 ]		; <<4 x float>> [#uses=1]
	%.07726 = phi <4 x float> [ %1171, %1173 ], [ %.17727, %1181 ], [ %.17727, %1180 ]		; <<4 x float>> [#uses=1]
	%.07730 = phi <4 x float> [ %1170, %1173 ], [ %.17731, %1181 ], [ %.17731, %1180 ]		; <<4 x float>> [#uses=1]
	%.07734 = phi <4 x float> [ %1169, %1173 ], [ %.17735, %1181 ], [ %.17735, %1180 ]		; <<4 x float>> [#uses=1]
	fadd <4 x float> zeroinitializer, zeroinitializer		; <<4 x float>>:1182 [#uses=1]
	br i1 false, label %1184, label %1183

; <label>:1183		; preds = %xST.exit504
	br label %xST.exit507

; <label>:1184		; preds = %xST.exit504
	br i1 false, label %1186, label %1185

; <label>:1185		; preds = %1184
	br label %1186

; <label>:1186		; preds = %1185, %1184
	br i1 false, label %1188, label %1187

; <label>:1187		; preds = %1186
	store <4 x float> zeroinitializer, ptr null
	br label %1188

; <label>:1188		; preds = %1187, %1186
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:1189 [#uses=1]
	shufflevector <4 x i32> %1189, <4 x i32> undef, <4 x i32> < i32 2, i32 2, i32 2, i32 2 >		; <<4 x i32>>:1190 [#uses=1]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> %1190, <4 x i32> zeroinitializer )		; <i32>:1191 [#uses=1]
	icmp eq i32 %1191, 0		; <i1>:1192 [#uses=1]
	br i1 %1192, label %1196, label %1193

; <label>:1193		; preds = %1188
	load <4 x float>, ptr null		; <<4 x float>>:1194 [#uses=1]
	shufflevector <4 x float> zeroinitializer, <4 x float> %1194, <4 x i32> < i32 0, i32 1, i32 2, i32 7 >		; <<4 x float>>:1195 [#uses=1]
	store <4 x float> %1195, ptr null
	br label %1196

; <label>:1196		; preds = %1193, %1188
	%.07742 = phi <4 x float> [ zeroinitializer, %1193 ], [ zeroinitializer, %1188 ]		; <<4 x float>> [#uses=0]
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:1197 [#uses=1]
	shufflevector <4 x i32> %1197, <4 x i32> undef, <4 x i32> < i32 3, i32 3, i32 3, i32 3 >		; <<4 x i32>>:1198 [#uses=1]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> %1198, <4 x i32> zeroinitializer )		; <i32>:1199 [#uses=1]
	icmp eq i32 %1199, 0		; <i1>:1200 [#uses=1]
	br i1 %1200, label %xST.exit507, label %1201

; <label>:1201		; preds = %1196
	store <4 x float> zeroinitializer, ptr null
	br label %xST.exit507

xST.exit507:		; preds = %1201, %1196, %1183
	%.07769 = phi <4 x float> [ %1182, %1183 ], [ %.17770, %1201 ], [ %.17770, %1196 ]		; <<4 x float>> [#uses=1]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer )		; <i32>:1202 [#uses=1]
	icmp eq i32 %1202, 0		; <i1>:1203 [#uses=1]
	br i1 %1203, label %1207, label %1204

; <label>:1204		; preds = %xST.exit507
	load <4 x float>, ptr null		; <<4 x float>>:1205 [#uses=1]
	shufflevector <4 x float> zeroinitializer, <4 x float> %1205, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:1206 [#uses=1]
	store <4 x float> %1206, ptr null
	br label %1207

; <label>:1207		; preds = %1204, %xST.exit507
	load <4 x i32>, ptr %.sub7896		; <<4 x i32>>:1208 [#uses=1]
	shufflevector <4 x i32> %1208, <4 x i32> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >		; <<4 x i32>>:1209 [#uses=1]
	call i32 @llvm.ppc.altivec.vcmpequw.p( i32 0, <4 x i32> %1209, <4 x i32> zeroinitializer )		; <i32>:1210 [#uses=1]
	icmp eq i32 %1210, 0		; <i1>:1211 [#uses=1]
	br i1 %1211, label %1215, label %1212

; <label>:1212		; preds = %1207
	load <4 x float>, ptr null		; <<4 x float>>:1213 [#uses=1]
	shufflevector <4 x float> zeroinitializer, <4 x float> %1213, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >		; <<4 x float>>:1214 [#uses=1]
	store <4 x float> %1214, ptr null
	br label %1215

; <label>:1215		; preds = %1212, %1207
	store <4 x float> zeroinitializer, ptr null
	br label %xLS.exit449
}

declare <4 x i32> @llvm.ppc.altivec.vsel(<4 x i32>, <4 x i32>, <4 x i32>)

declare void @llvm.ppc.altivec.stvewx(<4 x i32>, ptr)

declare <4 x float> @llvm.ppc.altivec.vrsqrtefp(<4 x float>)

declare <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32>, i32)

declare i32 @llvm.ppc.altivec.vcmpequw.p(i32, <4 x i32>, <4 x i32>)

declare <4 x i32> @llvm.ppc.altivec.vcmpgtfp(<4 x float>, <4 x float>)

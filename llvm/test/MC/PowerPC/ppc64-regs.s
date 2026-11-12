# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck %s

#CHECK: .cfi_startproc
#CHECK: .cfi_offset r0, 0
#CHECK: .cfi_offset r1, 8
#CHECK: .cfi_offset r2, 16
#CHECK: .cfi_offset r3, 24
#CHECK: .cfi_offset r4, 32
#CHECK: .cfi_offset r5, 40
#CHECK: .cfi_offset r6, 48
#CHECK: .cfi_offset r7, 56
#CHECK: .cfi_offset r8, 64
#CHECK: .cfi_offset r9, 72
#CHECK: .cfi_offset r10, 80
#CHECK: .cfi_offset r11, 88
#CHECK: .cfi_offset r12, 96
#CHECK: .cfi_offset r13, 104
#CHECK: .cfi_offset r14, 112
#CHECK: .cfi_offset r15, 120
#CHECK: .cfi_offset r16, 128
#CHECK: .cfi_offset r17, 136
#CHECK: .cfi_offset r18, 144
#CHECK: .cfi_offset r19, 152
#CHECK: .cfi_offset r20, 160
#CHECK: .cfi_offset r21, 168
#CHECK: .cfi_offset r22, 176
#CHECK: .cfi_offset r22, 184
#CHECK: .cfi_offset r23, 192
#CHECK: .cfi_offset r24, 200
#CHECK: .cfi_offset r25, 208
#CHECK: .cfi_offset r26, 216
#CHECK: .cfi_offset r27, 224
#CHECK: .cfi_offset r28, 232
#CHECK: .cfi_offset r29, 240
#CHECK: .cfi_offset r30, 248
#CHECK: .cfi_offset r31, 256

#CHECK: .cfi_offset f0, 300
#CHECK: .cfi_offset f1, 308
#CHECK: .cfi_offset f2, 316
#CHECK: .cfi_offset f3, 324
#CHECK: .cfi_offset f4, 332
#CHECK: .cfi_offset f5, 340
#CHECK: .cfi_offset f6, 348
#CHECK: .cfi_offset f7, 356
#CHECK: .cfi_offset f8, 364
#CHECK: .cfi_offset f9, 372
#CHECK: .cfi_offset f10, 380
#CHECK: .cfi_offset f11, 388
#CHECK: .cfi_offset f12, 396
#CHECK: .cfi_offset f13, 404
#CHECK: .cfi_offset f14, 412
#CHECK: .cfi_offset f15, 420
#CHECK: .cfi_offset f16, 428
#CHECK: .cfi_offset f17, 436
#CHECK: .cfi_offset f18, 444
#CHECK: .cfi_offset f19, 452
#CHECK: .cfi_offset f20, 460
#CHECK: .cfi_offset f21, 468
#CHECK: .cfi_offset f22, 476
#CHECK: .cfi_offset f22, 484
#CHECK: .cfi_offset f23, 492
#CHECK: .cfi_offset f24, 500
#CHECK: .cfi_offset f25, 508
#CHECK: .cfi_offset f26, 516
#CHECK: .cfi_offset f27, 524
#CHECK: .cfi_offset f28, 532
#CHECK: .cfi_offset f29, 540
#CHECK: .cfi_offset f30, 548
#CHECK: .cfi_offset f31, 556

#CHECK: .cfi_offset lr, 600
#CHECK: .cfi_offset ctr, 608
#CHECK: .cfi_offset vrsave, 616

#CHECK: .cfi_offset cr0, 620
#CHECK: .cfi_offset cr1, 621
#CHECK: .cfi_offset cr2, 622
#CHECK: .cfi_offset cr3, 623
#CHECK: .cfi_offset cr4, 624
#CHECK: .cfi_offset cr5, 625
#CHECK: .cfi_offset cr6, 626
#CHECK: .cfi_offset cr7, 627

#CHECK: .cfi_offset v0, 700
#CHECK: .cfi_offset v1, 716
#CHECK: .cfi_offset v2, 732
#CHECK: .cfi_offset v3, 748
#CHECK: .cfi_offset v4, 764
#CHECK: .cfi_offset v5, 780
#CHECK: .cfi_offset v6, 796
#CHECK: .cfi_offset v7, 812
#CHECK: .cfi_offset v8, 828
#CHECK: .cfi_offset v9, 844
#CHECK: .cfi_offset v10, 860
#CHECK: .cfi_offset v11, 876
#CHECK: .cfi_offset v12, 892
#CHECK: .cfi_offset v13, 908
#CHECK: .cfi_offset v14, 924
#CHECK: .cfi_offset v15, 940
#CHECK: .cfi_offset v16, 956
#CHECK: .cfi_offset v17, 972
#CHECK: .cfi_offset v18, 988
#CHECK: .cfi_offset v19, 1004
#CHECK: .cfi_offset v20, 1020
#CHECK: .cfi_offset v21, 1036
#CHECK: .cfi_offset v22, 1052
#CHECK: .cfi_offset v22, 1068
#CHECK: .cfi_offset v23, 1084
#CHECK: .cfi_offset v24, 1100
#CHECK: .cfi_offset v25, 1116
#CHECK: .cfi_offset v26, 1132
#CHECK: .cfi_offset v27, 1148
#CHECK: .cfi_offset v28, 1164
#CHECK: .cfi_offset v29, 1180
#CHECK: .cfi_offset v30, 1196
#CHECK: .cfi_offset v31, 1212
#CHECK: .cfi_endproc

	.cfi_startproc
	.cfi_offset r0,0
	.cfi_offset r1,8
	.cfi_offset r2,16
	.cfi_offset r3,24
	.cfi_offset r4,32
	.cfi_offset r5,40
	.cfi_offset r6,48
	.cfi_offset r7,56
	.cfi_offset r8,64
	.cfi_offset r9,72
	.cfi_offset r10,80
	.cfi_offset r11,88
	.cfi_offset r12,96
	.cfi_offset r13,104
	.cfi_offset r14,112
	.cfi_offset r15,120
	.cfi_offset r16,128
	.cfi_offset r17,136
	.cfi_offset r18,144
	.cfi_offset r19,152
	.cfi_offset r20,160
	.cfi_offset r21,168
	.cfi_offset r22,176
	.cfi_offset r22,184
	.cfi_offset r23,192
	.cfi_offset r24,200
	.cfi_offset r25,208
	.cfi_offset r26,216
	.cfi_offset r27,224
	.cfi_offset r28,232
	.cfi_offset r29,240
	.cfi_offset r30,248
	.cfi_offset r31,256

	.cfi_offset f0,300
	.cfi_offset f1,308
	.cfi_offset f2,316
	.cfi_offset f3,324
	.cfi_offset f4,332
	.cfi_offset f5,340
	.cfi_offset f6,348
	.cfi_offset f7,356
	.cfi_offset f8,364
	.cfi_offset f9,372
	.cfi_offset f10,380
	.cfi_offset f11,388
	.cfi_offset f12,396
	.cfi_offset f13,404
	.cfi_offset f14,412
	.cfi_offset f15,420
	.cfi_offset f16,428
	.cfi_offset f17,436
	.cfi_offset f18,444
	.cfi_offset f19,452
	.cfi_offset f20,460
	.cfi_offset f21,468
	.cfi_offset f22,476
	.cfi_offset f22,484
	.cfi_offset f23,492
	.cfi_offset f24,500
	.cfi_offset f25,508
	.cfi_offset f26,516
	.cfi_offset f27,524
	.cfi_offset f28,532
	.cfi_offset f29,540
	.cfi_offset f30,548
	.cfi_offset f31,556

	.cfi_offset lr,600
	.cfi_offset ctr,608
	.cfi_offset vrsave,616
	.cfi_offset cr0,620
	.cfi_offset cr1,621
	.cfi_offset cr2,622
	.cfi_offset cr3,623
	.cfi_offset cr4,624
	.cfi_offset cr5,625
	.cfi_offset cr6,626
	.cfi_offset cr7,627

	.cfi_offset v0,700
	.cfi_offset v1,716
	.cfi_offset v2,732
	.cfi_offset v3,748
	.cfi_offset v4,764
	.cfi_offset v5,780
	.cfi_offset v6,796
	.cfi_offset v7,812
	.cfi_offset v8,828
	.cfi_offset v9,844
	.cfi_offset v10,860
	.cfi_offset v11,876
	.cfi_offset v12,892
	.cfi_offset v13,908
	.cfi_offset v14,924
	.cfi_offset v15,940
	.cfi_offset v16,956
	.cfi_offset v17,972
	.cfi_offset v18,988
	.cfi_offset v19,1004
	.cfi_offset v20,1020
	.cfi_offset v21,1036
	.cfi_offset v22,1052
	.cfi_offset v22,1068
	.cfi_offset v23,1084
	.cfi_offset v24,1100
	.cfi_offset v25,1116
	.cfi_offset v26,1132
	.cfi_offset v27,1148
	.cfi_offset v28,1164
	.cfi_offset v29,1180
	.cfi_offset v30,1196
	.cfi_offset v31,1212

	.cfi_endproc

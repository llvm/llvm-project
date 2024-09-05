; RUN: opt < %s -passes=simplifycfg -S | FileCheck --check-prefixes=CHECK-1000 %s
; RUN: opt < %s -max-phi-entries-increase-after-removing-empty-block=989 -passes=simplifycfg -S | FileCheck --check-prefixes=CHECK-989 %s
; RUN: opt < %s -max-phi-entries-increase-after-removing-empty-block=489 -passes=simplifycfg -S | FileCheck --check-prefixes=CHECK-489 %s

; This test has the following CFG:
;   1. entry has a switch to 100 blocks: BB1 - BB100
;   2. For BB1 to BB50, it branches to BB101 and BB103
;   3. For BB51 to BB100, it branches to BB102 and BB103
;   4. BB101, BB102, BB103 branch to Merge unconditionally
;   5. Merge has 10 phis(x1 - x10).
; 
; If we remove BB103, it will increase the number of phi entries by (100 - 1) * 10 = 990.
; If we remove BB101 / BB102, it will increase the number of phi entries by (50 - 1) * 10 = 490.
;
; By default, in SimplifyCFG, we will not remove a block if it will increase more than 1000 phi entries.
; In the first test, BB103 will be removed, and every phi will have 102(3 + 100 - 1) phi entries.
; In the second test, we set max-phi-entries-increase-after-removing-empty-block to be 989, then BB103 should not be removed, 
; but BB101 and BB102 can be removed.
; In the third test, we set  max-phi-entries-increase-after-removing-empty-block to be 489, then no BB can be removed.

; CHECK-1000: %x1 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){101})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-1000: %x2 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){101})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-1000: %x3 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){101})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-1000: %x4 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){101})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-1000: %x5 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){101})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-1000: %x6 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){101})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-1000: %x7 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){101})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-1000: %x8 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){101})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-1000: %x9 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){101})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-1000: %x10 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){101})\[ [0-9], %BB[0-9]+ \]}}

; CHECK-989: %x1 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){100})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-989: %x2 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){100})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-989: %x3 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){100})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-989: %x4 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){100})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-989: %x5 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){100})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-989: %x6 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){100})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-989: %x7 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){100})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-989: %x8 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){100})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-989: %x9 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){100})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-989: %x10 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){100})\[ [0-9], %BB[0-9]+ \]}}

; CHECK-489: %x1 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){2})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-489: %x2 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){2})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-489: %x3 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){2})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-489: %x4 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){2})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-489: %x5 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){2})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-489: %x6 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){2})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-489: %x7 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){2})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-489: %x8 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){2})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-489: %x9 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){2})\[ [0-9], %BB[0-9]+ \]}}
; CHECK-489: %x10 = phi i16 {{((\[ [0-9], %BB[0-9]+ \], ){2})\[ [0-9], %BB[0-9]+ \]}}


;
define void @example(i32 %a, ptr %array) {
entry:
  switch i32 %a, label %BB1 [
    i32 1, label %BB1
    i32 2, label %BB2
    i32 3, label %BB3
    i32 4, label %BB4
    i32 5, label %BB5
    i32 6, label %BB6
    i32 7, label %BB7
    i32 8, label %BB8
    i32 9, label %BB9
    i32 10, label %BB10
    i32 11, label %BB11
    i32 12, label %BB12
    i32 13, label %BB13
    i32 14, label %BB14
    i32 15, label %BB15
    i32 16, label %BB16
    i32 17, label %BB17
    i32 18, label %BB18
    i32 19, label %BB19
    i32 20, label %BB20
    i32 21, label %BB21
    i32 22, label %BB22
    i32 23, label %BB23
    i32 24, label %BB24
    i32 25, label %BB25
    i32 26, label %BB26
    i32 27, label %BB27
    i32 28, label %BB28
    i32 29, label %BB29
    i32 30, label %BB30
    i32 31, label %BB31
    i32 32, label %BB32
    i32 33, label %BB33
    i32 34, label %BB34
    i32 35, label %BB35
    i32 36, label %BB36
    i32 37, label %BB37
    i32 38, label %BB38
    i32 39, label %BB39
    i32 40, label %BB40
    i32 41, label %BB41
    i32 42, label %BB42
    i32 43, label %BB43
    i32 44, label %BB44
    i32 45, label %BB45
    i32 46, label %BB46
    i32 47, label %BB47
    i32 48, label %BB48
    i32 49, label %BB49
    i32 50, label %BB50
    i32 51, label %BB51
    i32 52, label %BB52
    i32 53, label %BB53
    i32 54, label %BB54
    i32 55, label %BB55
    i32 56, label %BB56
    i32 57, label %BB57
    i32 58, label %BB58
    i32 59, label %BB59
    i32 60, label %BB60
    i32 61, label %BB61
    i32 62, label %BB62
    i32 63, label %BB63
    i32 64, label %BB64
    i32 65, label %BB65
    i32 66, label %BB66
    i32 67, label %BB67
    i32 68, label %BB68
    i32 69, label %BB69
    i32 70, label %BB70
    i32 71, label %BB71
    i32 72, label %BB72
    i32 73, label %BB73
    i32 74, label %BB74
    i32 75, label %BB75
    i32 76, label %BB76
    i32 77, label %BB77
    i32 78, label %BB78
    i32 79, label %BB79
    i32 80, label %BB80
    i32 81, label %BB81
    i32 82, label %BB82
    i32 83, label %BB83
    i32 84, label %BB84
    i32 85, label %BB85
    i32 86, label %BB86
    i32 87, label %BB87
    i32 88, label %BB88
    i32 89, label %BB89
    i32 90, label %BB90
    i32 91, label %BB91
    i32 92, label %BB92
    i32 93, label %BB93
    i32 94, label %BB94
    i32 95, label %BB95
    i32 96, label %BB96
    i32 97, label %BB97
    i32 98, label %BB98
    i32 99, label %BB99
    i32 100, label %BB100
  ]

BB1:                                              ; preds = %default, %entry
  %elem1 = getelementptr i32, ptr %array, i32 1
  %val1 = load i32, ptr %elem1, align 4
  %cmp1 = icmp eq i32 %val1, 1
  br i1 %cmp1, label %BB101, label %BB103

BB2:                                              ; preds = %entry
  %elem2 = getelementptr i32, ptr %array, i32 2
  %val2 = load i32, ptr %elem2, align 4
  %cmp2 = icmp eq i32 %val2, 2
  br i1 %cmp2, label %BB101, label %BB103

BB3:                                              ; preds = %entry
  %elem3 = getelementptr i32, ptr %array, i32 3
  %val3 = load i32, ptr %elem3, align 4
  %cmp3 = icmp eq i32 %val3, 3
  br i1 %cmp3, label %BB101, label %BB103

BB4:                                              ; preds = %entry
  %elem4 = getelementptr i32, ptr %array, i32 4
  %val4 = load i32, ptr %elem4, align 4
  %cmp4 = icmp eq i32 %val4, 4
  br i1 %cmp4, label %BB101, label %BB103

BB5:                                              ; preds = %entry
  %elem5 = getelementptr i32, ptr %array, i32 5
  %val5 = load i32, ptr %elem5, align 4
  %cmp5 = icmp eq i32 %val5, 5
  br i1 %cmp5, label %BB101, label %BB103

BB6:                                              ; preds = %entry
  %elem6 = getelementptr i32, ptr %array, i32 6
  %val6 = load i32, ptr %elem6, align 4
  %cmp6 = icmp eq i32 %val6, 6
  br i1 %cmp6, label %BB101, label %BB103

BB7:                                              ; preds = %entry
  %elem7 = getelementptr i32, ptr %array, i32 7
  %val7 = load i32, ptr %elem7, align 4
  %cmp7 = icmp eq i32 %val7, 7
  br i1 %cmp7, label %BB101, label %BB103

BB8:                                              ; preds = %entry
  %elem8 = getelementptr i32, ptr %array, i32 8
  %val8 = load i32, ptr %elem8, align 4
  %cmp8 = icmp eq i32 %val8, 8
  br i1 %cmp8, label %BB101, label %BB103

BB9:                                              ; preds = %entry
  %elem9 = getelementptr i32, ptr %array, i32 9
  %val9 = load i32, ptr %elem9, align 4
  %cmp9 = icmp eq i32 %val9, 9
  br i1 %cmp9, label %BB101, label %BB103

BB10:                                             ; preds = %entry
  %elem10 = getelementptr i32, ptr %array, i32 10
  %val10 = load i32, ptr %elem10, align 4
  %cmp10 = icmp eq i32 %val10, 10
  br i1 %cmp10, label %BB101, label %BB103

BB11:                                             ; preds = %entry
  %elem11 = getelementptr i32, ptr %array, i32 11
  %val11 = load i32, ptr %elem11, align 4
  %cmp11 = icmp eq i32 %val11, 11
  br i1 %cmp11, label %BB101, label %BB103

BB12:                                             ; preds = %entry
  %elem12 = getelementptr i32, ptr %array, i32 12
  %val12 = load i32, ptr %elem12, align 4
  %cmp12 = icmp eq i32 %val12, 12
  br i1 %cmp12, label %BB101, label %BB103

BB13:                                             ; preds = %entry
  %elem13 = getelementptr i32, ptr %array, i32 13
  %val13 = load i32, ptr %elem13, align 4
  %cmp13 = icmp eq i32 %val13, 13
  br i1 %cmp13, label %BB101, label %BB103

BB14:                                             ; preds = %entry
  %elem14 = getelementptr i32, ptr %array, i32 14
  %val14 = load i32, ptr %elem14, align 4
  %cmp14 = icmp eq i32 %val14, 14
  br i1 %cmp14, label %BB101, label %BB103

BB15:                                             ; preds = %entry
  %elem15 = getelementptr i32, ptr %array, i32 15
  %val15 = load i32, ptr %elem15, align 4
  %cmp15 = icmp eq i32 %val15, 15
  br i1 %cmp15, label %BB101, label %BB103

BB16:                                             ; preds = %entry
  %elem16 = getelementptr i32, ptr %array, i32 16
  %val16 = load i32, ptr %elem16, align 4
  %cmp16 = icmp eq i32 %val16, 16
  br i1 %cmp16, label %BB101, label %BB103

BB17:                                             ; preds = %entry
  %elem17 = getelementptr i32, ptr %array, i32 17
  %val17 = load i32, ptr %elem17, align 4
  %cmp17 = icmp eq i32 %val17, 17
  br i1 %cmp17, label %BB101, label %BB103

BB18:                                             ; preds = %entry
  %elem18 = getelementptr i32, ptr %array, i32 18
  %val18 = load i32, ptr %elem18, align 4
  %cmp18 = icmp eq i32 %val18, 18
  br i1 %cmp18, label %BB101, label %BB103

BB19:                                             ; preds = %entry
  %elem19 = getelementptr i32, ptr %array, i32 19
  %val19 = load i32, ptr %elem19, align 4
  %cmp19 = icmp eq i32 %val19, 19
  br i1 %cmp19, label %BB101, label %BB103

BB20:                                             ; preds = %entry
  %elem20 = getelementptr i32, ptr %array, i32 20
  %val20 = load i32, ptr %elem20, align 4
  %cmp20 = icmp eq i32 %val20, 20
  br i1 %cmp20, label %BB101, label %BB103

BB21:                                             ; preds = %entry
  %elem21 = getelementptr i32, ptr %array, i32 21
  %val21 = load i32, ptr %elem21, align 4
  %cmp21 = icmp eq i32 %val21, 21
  br i1 %cmp21, label %BB101, label %BB103

BB22:                                             ; preds = %entry
  %elem22 = getelementptr i32, ptr %array, i32 22
  %val22 = load i32, ptr %elem22, align 4
  %cmp22 = icmp eq i32 %val22, 22
  br i1 %cmp22, label %BB101, label %BB103

BB23:                                             ; preds = %entry
  %elem23 = getelementptr i32, ptr %array, i32 23
  %val23 = load i32, ptr %elem23, align 4
  %cmp23 = icmp eq i32 %val23, 23
  br i1 %cmp23, label %BB101, label %BB103

BB24:                                             ; preds = %entry
  %elem24 = getelementptr i32, ptr %array, i32 24
  %val24 = load i32, ptr %elem24, align 4
  %cmp24 = icmp eq i32 %val24, 24
  br i1 %cmp24, label %BB101, label %BB103

BB25:                                             ; preds = %entry
  %elem25 = getelementptr i32, ptr %array, i32 25
  %val25 = load i32, ptr %elem25, align 4
  %cmp25 = icmp eq i32 %val25, 25
  br i1 %cmp25, label %BB101, label %BB103

BB26:                                             ; preds = %entry
  %elem26 = getelementptr i32, ptr %array, i32 26
  %val26 = load i32, ptr %elem26, align 4
  %cmp26 = icmp eq i32 %val26, 26
  br i1 %cmp26, label %BB101, label %BB103

BB27:                                             ; preds = %entry
  %elem27 = getelementptr i32, ptr %array, i32 27
  %val27 = load i32, ptr %elem27, align 4
  %cmp27 = icmp eq i32 %val27, 27
  br i1 %cmp27, label %BB101, label %BB103

BB28:                                             ; preds = %entry
  %elem28 = getelementptr i32, ptr %array, i32 28
  %val28 = load i32, ptr %elem28, align 4
  %cmp28 = icmp eq i32 %val28, 28
  br i1 %cmp28, label %BB101, label %BB103

BB29:                                             ; preds = %entry
  %elem29 = getelementptr i32, ptr %array, i32 29
  %val29 = load i32, ptr %elem29, align 4
  %cmp29 = icmp eq i32 %val29, 29
  br i1 %cmp29, label %BB101, label %BB103

BB30:                                             ; preds = %entry
  %elem30 = getelementptr i32, ptr %array, i32 30
  %val30 = load i32, ptr %elem30, align 4
  %cmp30 = icmp eq i32 %val30, 30
  br i1 %cmp30, label %BB101, label %BB103

BB31:                                             ; preds = %entry
  %elem31 = getelementptr i32, ptr %array, i32 31
  %val31 = load i32, ptr %elem31, align 4
  %cmp31 = icmp eq i32 %val31, 31
  br i1 %cmp31, label %BB101, label %BB103

BB32:                                             ; preds = %entry
  %elem32 = getelementptr i32, ptr %array, i32 32
  %val32 = load i32, ptr %elem32, align 4
  %cmp32 = icmp eq i32 %val32, 32
  br i1 %cmp32, label %BB101, label %BB103

BB33:                                             ; preds = %entry
  %elem33 = getelementptr i32, ptr %array, i32 33
  %val33 = load i32, ptr %elem33, align 4
  %cmp33 = icmp eq i32 %val33, 33
  br i1 %cmp33, label %BB101, label %BB103

BB34:                                             ; preds = %entry
  %elem34 = getelementptr i32, ptr %array, i32 34
  %val34 = load i32, ptr %elem34, align 4
  %cmp34 = icmp eq i32 %val34, 34
  br i1 %cmp34, label %BB101, label %BB103

BB35:                                             ; preds = %entry
  %elem35 = getelementptr i32, ptr %array, i32 35
  %val35 = load i32, ptr %elem35, align 4
  %cmp35 = icmp eq i32 %val35, 35
  br i1 %cmp35, label %BB101, label %BB103

BB36:                                             ; preds = %entry
  %elem36 = getelementptr i32, ptr %array, i32 36
  %val36 = load i32, ptr %elem36, align 4
  %cmp36 = icmp eq i32 %val36, 36
  br i1 %cmp36, label %BB101, label %BB103

BB37:                                             ; preds = %entry
  %elem37 = getelementptr i32, ptr %array, i32 37
  %val37 = load i32, ptr %elem37, align 4
  %cmp37 = icmp eq i32 %val37, 37
  br i1 %cmp37, label %BB101, label %BB103

BB38:                                             ; preds = %entry
  %elem38 = getelementptr i32, ptr %array, i32 38
  %val38 = load i32, ptr %elem38, align 4
  %cmp38 = icmp eq i32 %val38, 38
  br i1 %cmp38, label %BB101, label %BB103

BB39:                                             ; preds = %entry
  %elem39 = getelementptr i32, ptr %array, i32 39
  %val39 = load i32, ptr %elem39, align 4
  %cmp39 = icmp eq i32 %val39, 39
  br i1 %cmp39, label %BB101, label %BB103

BB40:                                             ; preds = %entry
  %elem40 = getelementptr i32, ptr %array, i32 40
  %val40 = load i32, ptr %elem40, align 4
  %cmp40 = icmp eq i32 %val40, 40
  br i1 %cmp40, label %BB101, label %BB103

BB41:                                             ; preds = %entry
  %elem41 = getelementptr i32, ptr %array, i32 41
  %val41 = load i32, ptr %elem41, align 4
  %cmp41 = icmp eq i32 %val41, 41
  br i1 %cmp41, label %BB101, label %BB103

BB42:                                             ; preds = %entry
  %elem42 = getelementptr i32, ptr %array, i32 42
  %val42 = load i32, ptr %elem42, align 4
  %cmp42 = icmp eq i32 %val42, 42
  br i1 %cmp42, label %BB101, label %BB103

BB43:                                             ; preds = %entry
  %elem43 = getelementptr i32, ptr %array, i32 43
  %val43 = load i32, ptr %elem43, align 4
  %cmp43 = icmp eq i32 %val43, 43
  br i1 %cmp43, label %BB101, label %BB103

BB44:                                             ; preds = %entry
  %elem44 = getelementptr i32, ptr %array, i32 44
  %val44 = load i32, ptr %elem44, align 4
  %cmp44 = icmp eq i32 %val44, 44
  br i1 %cmp44, label %BB101, label %BB103

BB45:                                             ; preds = %entry
  %elem45 = getelementptr i32, ptr %array, i32 45
  %val45 = load i32, ptr %elem45, align 4
  %cmp45 = icmp eq i32 %val45, 45
  br i1 %cmp45, label %BB101, label %BB103

BB46:                                             ; preds = %entry
  %elem46 = getelementptr i32, ptr %array, i32 46
  %val46 = load i32, ptr %elem46, align 4
  %cmp46 = icmp eq i32 %val46, 46
  br i1 %cmp46, label %BB101, label %BB103

BB47:                                             ; preds = %entry
  %elem47 = getelementptr i32, ptr %array, i32 47
  %val47 = load i32, ptr %elem47, align 4
  %cmp47 = icmp eq i32 %val47, 47
  br i1 %cmp47, label %BB101, label %BB103

BB48:                                             ; preds = %entry
  %elem48 = getelementptr i32, ptr %array, i32 48
  %val48 = load i32, ptr %elem48, align 4
  %cmp48 = icmp eq i32 %val48, 48
  br i1 %cmp48, label %BB101, label %BB103

BB49:                                             ; preds = %entry
  %elem49 = getelementptr i32, ptr %array, i32 49
  %val49 = load i32, ptr %elem49, align 4
  %cmp49 = icmp eq i32 %val49, 49
  br i1 %cmp49, label %BB101, label %BB103

BB50:                                             ; preds = %entry
  %elem50 = getelementptr i32, ptr %array, i32 50
  %val50 = load i32, ptr %elem50, align 4
  %cmp50 = icmp eq i32 %val50, 50
  br i1 %cmp50, label %BB101, label %BB103

BB51:                                             ; preds = %entry
  %elem51 = getelementptr i32, ptr %array, i32 51
  %val51 = load i32, ptr %elem51, align 4
  %cmp51 = icmp eq i32 %val51, 51
  br i1 %cmp51, label %BB102, label %BB103

BB52:                                             ; preds = %entry
  %elem52 = getelementptr i32, ptr %array, i32 52
  %val52 = load i32, ptr %elem52, align 4
  %cmp52 = icmp eq i32 %val52, 52
  br i1 %cmp52, label %BB102, label %BB103

BB53:                                             ; preds = %entry
  %elem53 = getelementptr i32, ptr %array, i32 53
  %val53 = load i32, ptr %elem53, align 4
  %cmp53 = icmp eq i32 %val53, 53
  br i1 %cmp53, label %BB102, label %BB103

BB54:                                             ; preds = %entry
  %elem54 = getelementptr i32, ptr %array, i32 54
  %val54 = load i32, ptr %elem54, align 4
  %cmp54 = icmp eq i32 %val54, 54
  br i1 %cmp54, label %BB102, label %BB103

BB55:                                             ; preds = %entry
  %elem55 = getelementptr i32, ptr %array, i32 55
  %val55 = load i32, ptr %elem55, align 4
  %cmp55 = icmp eq i32 %val55, 55
  br i1 %cmp55, label %BB102, label %BB103

BB56:                                             ; preds = %entry
  %elem56 = getelementptr i32, ptr %array, i32 56
  %val56 = load i32, ptr %elem56, align 4
  %cmp56 = icmp eq i32 %val56, 56
  br i1 %cmp56, label %BB102, label %BB103

BB57:                                             ; preds = %entry
  %elem57 = getelementptr i32, ptr %array, i32 57
  %val57 = load i32, ptr %elem57, align 4
  %cmp57 = icmp eq i32 %val57, 57
  br i1 %cmp57, label %BB102, label %BB103

BB58:                                             ; preds = %entry
  %elem58 = getelementptr i32, ptr %array, i32 58
  %val58 = load i32, ptr %elem58, align 4
  %cmp58 = icmp eq i32 %val58, 58
  br i1 %cmp58, label %BB102, label %BB103

BB59:                                             ; preds = %entry
  %elem59 = getelementptr i32, ptr %array, i32 59
  %val59 = load i32, ptr %elem59, align 4
  %cmp59 = icmp eq i32 %val59, 59
  br i1 %cmp59, label %BB102, label %BB103

BB60:                                             ; preds = %entry
  %elem60 = getelementptr i32, ptr %array, i32 60
  %val60 = load i32, ptr %elem60, align 4
  %cmp60 = icmp eq i32 %val60, 60
  br i1 %cmp60, label %BB102, label %BB103

BB61:                                             ; preds = %entry
  %elem61 = getelementptr i32, ptr %array, i32 61
  %val61 = load i32, ptr %elem61, align 4
  %cmp61 = icmp eq i32 %val61, 61
  br i1 %cmp61, label %BB102, label %BB103

BB62:                                             ; preds = %entry
  %elem62 = getelementptr i32, ptr %array, i32 62
  %val62 = load i32, ptr %elem62, align 4
  %cmp62 = icmp eq i32 %val62, 62
  br i1 %cmp62, label %BB102, label %BB103

BB63:                                             ; preds = %entry
  %elem63 = getelementptr i32, ptr %array, i32 63
  %val63 = load i32, ptr %elem63, align 4
  %cmp63 = icmp eq i32 %val63, 63
  br i1 %cmp63, label %BB102, label %BB103

BB64:                                             ; preds = %entry
  %elem64 = getelementptr i32, ptr %array, i32 64
  %val64 = load i32, ptr %elem64, align 4
  %cmp64 = icmp eq i32 %val64, 64
  br i1 %cmp64, label %BB102, label %BB103

BB65:                                             ; preds = %entry
  %elem65 = getelementptr i32, ptr %array, i32 65
  %val65 = load i32, ptr %elem65, align 4
  %cmp65 = icmp eq i32 %val65, 65
  br i1 %cmp65, label %BB102, label %BB103

BB66:                                             ; preds = %entry
  %elem66 = getelementptr i32, ptr %array, i32 66
  %val66 = load i32, ptr %elem66, align 4
  %cmp66 = icmp eq i32 %val66, 66
  br i1 %cmp66, label %BB102, label %BB103

BB67:                                             ; preds = %entry
  %elem67 = getelementptr i32, ptr %array, i32 67
  %val67 = load i32, ptr %elem67, align 4
  %cmp67 = icmp eq i32 %val67, 67
  br i1 %cmp67, label %BB102, label %BB103

BB68:                                             ; preds = %entry
  %elem68 = getelementptr i32, ptr %array, i32 68
  %val68 = load i32, ptr %elem68, align 4
  %cmp68 = icmp eq i32 %val68, 68
  br i1 %cmp68, label %BB102, label %BB103

BB69:                                             ; preds = %entry
  %elem69 = getelementptr i32, ptr %array, i32 69
  %val69 = load i32, ptr %elem69, align 4
  %cmp69 = icmp eq i32 %val69, 69
  br i1 %cmp69, label %BB102, label %BB103

BB70:                                             ; preds = %entry
  %elem70 = getelementptr i32, ptr %array, i32 70
  %val70 = load i32, ptr %elem70, align 4
  %cmp70 = icmp eq i32 %val70, 70
  br i1 %cmp70, label %BB102, label %BB103

BB71:                                             ; preds = %entry
  %elem71 = getelementptr i32, ptr %array, i32 71
  %val71 = load i32, ptr %elem71, align 4
  %cmp71 = icmp eq i32 %val71, 71
  br i1 %cmp71, label %BB102, label %BB103

BB72:                                             ; preds = %entry
  %elem72 = getelementptr i32, ptr %array, i32 72
  %val72 = load i32, ptr %elem72, align 4
  %cmp72 = icmp eq i32 %val72, 72
  br i1 %cmp72, label %BB102, label %BB103

BB73:                                             ; preds = %entry
  %elem73 = getelementptr i32, ptr %array, i32 73
  %val73 = load i32, ptr %elem73, align 4
  %cmp73 = icmp eq i32 %val73, 73
  br i1 %cmp73, label %BB102, label %BB103

BB74:                                             ; preds = %entry
  %elem74 = getelementptr i32, ptr %array, i32 74
  %val74 = load i32, ptr %elem74, align 4
  %cmp74 = icmp eq i32 %val74, 74
  br i1 %cmp74, label %BB102, label %BB103

BB75:                                             ; preds = %entry
  %elem75 = getelementptr i32, ptr %array, i32 75
  %val75 = load i32, ptr %elem75, align 4
  %cmp75 = icmp eq i32 %val75, 75
  br i1 %cmp75, label %BB102, label %BB103

BB76:                                             ; preds = %entry
  %elem76 = getelementptr i32, ptr %array, i32 76
  %val76 = load i32, ptr %elem76, align 4
  %cmp76 = icmp eq i32 %val76, 76
  br i1 %cmp76, label %BB102, label %BB103

BB77:                                             ; preds = %entry
  %elem77 = getelementptr i32, ptr %array, i32 77
  %val77 = load i32, ptr %elem77, align 4
  %cmp77 = icmp eq i32 %val77, 77
  br i1 %cmp77, label %BB102, label %BB103

BB78:                                             ; preds = %entry
  %elem78 = getelementptr i32, ptr %array, i32 78
  %val78 = load i32, ptr %elem78, align 4
  %cmp78 = icmp eq i32 %val78, 78
  br i1 %cmp78, label %BB102, label %BB103

BB79:                                             ; preds = %entry
  %elem79 = getelementptr i32, ptr %array, i32 79
  %val79 = load i32, ptr %elem79, align 4
  %cmp79 = icmp eq i32 %val79, 79
  br i1 %cmp79, label %BB102, label %BB103

BB80:                                             ; preds = %entry
  %elem80 = getelementptr i32, ptr %array, i32 80
  %val80 = load i32, ptr %elem80, align 4
  %cmp80 = icmp eq i32 %val80, 80
  br i1 %cmp80, label %BB102, label %BB103

BB81:                                             ; preds = %entry
  %elem81 = getelementptr i32, ptr %array, i32 81
  %val81 = load i32, ptr %elem81, align 4
  %cmp81 = icmp eq i32 %val81, 81
  br i1 %cmp81, label %BB102, label %BB103

BB82:                                             ; preds = %entry
  %elem82 = getelementptr i32, ptr %array, i32 82
  %val82 = load i32, ptr %elem82, align 4
  %cmp82 = icmp eq i32 %val82, 82
  br i1 %cmp82, label %BB102, label %BB103

BB83:                                             ; preds = %entry
  %elem83 = getelementptr i32, ptr %array, i32 83
  %val83 = load i32, ptr %elem83, align 4
  %cmp83 = icmp eq i32 %val83, 83
  br i1 %cmp83, label %BB102, label %BB103

BB84:                                             ; preds = %entry
  %elem84 = getelementptr i32, ptr %array, i32 84
  %val84 = load i32, ptr %elem84, align 4
  %cmp84 = icmp eq i32 %val84, 84
  br i1 %cmp84, label %BB102, label %BB103

BB85:                                             ; preds = %entry
  %elem85 = getelementptr i32, ptr %array, i32 85
  %val85 = load i32, ptr %elem85, align 4
  %cmp85 = icmp eq i32 %val85, 85
  br i1 %cmp85, label %BB102, label %BB103

BB86:                                             ; preds = %entry
  %elem86 = getelementptr i32, ptr %array, i32 86
  %val86 = load i32, ptr %elem86, align 4
  %cmp86 = icmp eq i32 %val86, 86
  br i1 %cmp86, label %BB102, label %BB103

BB87:                                             ; preds = %entry
  %elem87 = getelementptr i32, ptr %array, i32 87
  %val87 = load i32, ptr %elem87, align 4
  %cmp87 = icmp eq i32 %val87, 87
  br i1 %cmp87, label %BB102, label %BB103

BB88:                                             ; preds = %entry
  %elem88 = getelementptr i32, ptr %array, i32 88
  %val88 = load i32, ptr %elem88, align 4
  %cmp88 = icmp eq i32 %val88, 88
  br i1 %cmp88, label %BB102, label %BB103

BB89:                                             ; preds = %entry
  %elem89 = getelementptr i32, ptr %array, i32 89
  %val89 = load i32, ptr %elem89, align 4
  %cmp89 = icmp eq i32 %val89, 89
  br i1 %cmp89, label %BB102, label %BB103

BB90:                                             ; preds = %entry
  %elem90 = getelementptr i32, ptr %array, i32 90
  %val90 = load i32, ptr %elem90, align 4
  %cmp90 = icmp eq i32 %val90, 90
  br i1 %cmp90, label %BB102, label %BB103

BB91:                                             ; preds = %entry
  %elem91 = getelementptr i32, ptr %array, i32 91
  %val91 = load i32, ptr %elem91, align 4
  %cmp91 = icmp eq i32 %val91, 91
  br i1 %cmp91, label %BB102, label %BB103

BB92:                                             ; preds = %entry
  %elem92 = getelementptr i32, ptr %array, i32 92
  %val92 = load i32, ptr %elem92, align 4
  %cmp92 = icmp eq i32 %val92, 92
  br i1 %cmp92, label %BB102, label %BB103

BB93:                                             ; preds = %entry
  %elem93 = getelementptr i32, ptr %array, i32 93
  %val93 = load i32, ptr %elem93, align 4
  %cmp93 = icmp eq i32 %val93, 93
  br i1 %cmp93, label %BB102, label %BB103

BB94:                                             ; preds = %entry
  %elem94 = getelementptr i32, ptr %array, i32 94
  %val94 = load i32, ptr %elem94, align 4
  %cmp94 = icmp eq i32 %val94, 94
  br i1 %cmp94, label %BB102, label %BB103

BB95:                                             ; preds = %entry
  %elem95 = getelementptr i32, ptr %array, i32 95
  %val95 = load i32, ptr %elem95, align 4
  %cmp95 = icmp eq i32 %val95, 95
  br i1 %cmp95, label %BB102, label %BB103

BB96:                                             ; preds = %entry
  %elem96 = getelementptr i32, ptr %array, i32 96
  %val96 = load i32, ptr %elem96, align 4
  %cmp96 = icmp eq i32 %val96, 96
  br i1 %cmp96, label %BB102, label %BB103

BB97:                                             ; preds = %entry
  %elem97 = getelementptr i32, ptr %array, i32 97
  %val97 = load i32, ptr %elem97, align 4
  %cmp97 = icmp eq i32 %val97, 97
  br i1 %cmp97, label %BB102, label %BB103

BB98:                                             ; preds = %entry
  %elem98 = getelementptr i32, ptr %array, i32 98
  %val98 = load i32, ptr %elem98, align 4
  %cmp98 = icmp eq i32 %val98, 98
  br i1 %cmp98, label %BB102, label %BB103

BB99:                                             ; preds = %entry
  %elem99 = getelementptr i32, ptr %array, i32 99
  %val99 = load i32, ptr %elem99, align 4
  %cmp99 = icmp eq i32 %val99, 99
  br i1 %cmp99, label %BB102, label %BB103

BB100:                                            ; preds = %entry
  %elem100 = getelementptr i32, ptr %array, i32 100
  %val100 = load i32, ptr %elem100, align 4
  %cmp100 = icmp eq i32 %val100, 100
  br i1 %cmp100, label %BB102, label %BB103

BB103:                                            ; preds = %BB100, %BB99, %BB98, %BB97, %BB96, %BB95, %BB94, %BB93, %BB92, %BB91, %BB90, %BB89, %BB88, %BB87, %BB86, %BB85, %BB84, %BB83, %BB82, %BB81, %BB80, %BB79, %BB78, %BB77, %BB76, %BB75, %BB74, %BB73, %BB72, %BB71, %BB70, %BB69, %BB68, %BB67, %BB66, %BB65, %BB64, %BB63, %BB62, %BB61, %BB60, %BB59, %BB58, %BB57, %BB56, %BB55, %BB54, %BB53, %BB52, %BB51, %BB50, %BB49, %BB48, %BB47, %BB46, %BB45, %BB44, %BB43, %BB42, %BB41, %BB40, %BB39, %BB38, %BB37, %BB36, %BB35, %BB34, %BB33, %BB32, %BB31, %BB30, %BB29, %BB28, %BB27, %BB26, %BB25, %BB24, %BB23, %BB22, %BB21, %BB20, %BB19, %BB18, %BB17, %BB16, %BB15, %BB14, %BB13, %BB12, %BB11, %BB10, %BB9, %BB8, %BB7, %BB6, %BB5, %BB4, %BB3, %BB2, %BB1
  br label %Merge

BB101:                                            ; preds = %BB50, %BB49, %BB48, %BB47, %BB46, %BB45, %BB44, %BB43, %BB42, %BB41, %BB40, %BB39, %BB38, %BB37, %BB36, %BB35, %BB34, %BB33, %BB32, %BB31, %BB30, %BB29, %BB28, %BB27, %BB26, %BB25, %BB24, %BB23, %BB22, %BB21, %BB20, %BB19, %BB18, %BB17, %BB16, %BB15, %BB14, %BB13, %BB12, %BB11, %BB10, %BB9, %BB8, %BB7, %BB6, %BB5, %BB4, %BB3, %BB2, %BB1
  br label %Merge

BB102:                                            ; preds = %BB100, %BB99, %BB98, %BB97, %BB96, %BB95, %BB94, %BB93, %BB92, %BB91, %BB90, %BB89, %BB88, %BB87, %BB86, %BB85, %BB84, %BB83, %BB82, %BB81, %BB80, %BB79, %BB78, %BB77, %BB76, %BB75, %BB74, %BB73, %BB72, %BB71, %BB70, %BB69, %BB68, %BB67, %BB66, %BB65, %BB64, %BB63, %BB62, %BB61, %BB60, %BB59, %BB58, %BB57, %BB56, %BB55, %BB54, %BB53, %BB52, %BB51
  br label %Merge

Merge:                                            ; preds = %BB103, %BB102, %BB101
  %x1 = phi i16 [ 1, %BB103 ], [ 0, %BB101 ], [ 2, %BB102 ]
  %x2 = phi i16 [ 2 , %BB103 ], [ 0, %BB101 ], [ 2, %BB102 ]
  %x3 = phi i16 [ 3 , %BB103 ], [ 0, %BB101 ], [ 2, %BB102 ]
  %x4 = phi i16 [ 4 , %BB103 ], [ 0, %BB101 ], [ 2, %BB102 ]
  %x5 = phi i16 [ 5 , %BB103 ], [ 0, %BB101 ], [ 2, %BB102 ]
  %x6 = phi i16 [ 6 , %BB103 ], [ 0, %BB101 ], [ 2, %BB102 ]
  %x7 = phi i16 [ 7 , %BB103 ], [ 0, %BB101 ], [ 2, %BB102 ]
  %x8 = phi i16 [ 8 , %BB103 ], [ 0, %BB101 ], [ 2, %BB102 ]
  %x9 = phi i16 [ 9 , %BB103 ], [ 0, %BB101 ], [ 2, %BB102 ]
  %x10 = phi i16 [ 0, %BB103 ], [ 0, %BB101 ], [ 2, %BB102 ]
  ret void
}

; RUN: llc < %s -march=mips -mcpu=mips32r6 -o /dev/null

; Test that this file is compiled in a reasonable time period. Without the
; optimization level check in findBetterNeighbors, this test demonstrates
; a severe compile time regression (30~ minutes) vs. <10 seconds at 'optnone'.

declare i8 @k(ptr)

define void @d(i32 signext %e4) #1 {
entry:
  %e4.addr = alloca i32, align 4
  %old_val = alloca i8, align 1
  %new_val = alloca i8, align 1
  %simd = alloca i8, align 1
  %code = alloca [269 x i8], align 1
  store i32 %e4, ptr %e4.addr, align 4
  %call = call zeroext i8 @k(ptr %simd)
  store i8 %call, ptr %simd, align 1

  store i8 32, ptr %code, align 1
  %arrayinit.element = getelementptr inbounds i8, ptr %code, i32 1
  %a2 = load i8, ptr %old_val, align 1
  store i8 %a2, ptr %arrayinit.element, align 1
  %arrayinit.element1 = getelementptr inbounds i8, ptr %arrayinit.element, i32 1
  store i8 -3, ptr %arrayinit.element1, align 1
  %arrayinit.element2 = getelementptr inbounds i8, ptr %arrayinit.element1, i32 1
  store i8 0, ptr %arrayinit.element2, align 1
  %arrayinit.element3 = getelementptr inbounds i8, ptr %arrayinit.element2, i32 1
  store i8 33, ptr %arrayinit.element3, align 1
  %arrayinit.element4 = getelementptr inbounds i8, ptr %arrayinit.element3, i32 1
  %a3 = load i8, ptr %simd, align 1
  store i8 %a3, ptr %arrayinit.element4, align 1
  %arrayinit.element5 = getelementptr inbounds i8, ptr %arrayinit.element4, i32 1
  store i8 32, ptr %arrayinit.element5, align 1
  %arrayinit.element6 = getelementptr inbounds i8, ptr %arrayinit.element5, i32 1
  %a4 = load i8, ptr %simd, align 1
  store i8 %a4, ptr %arrayinit.element6, align 1
  %arrayinit.element7 = getelementptr inbounds i8, ptr %arrayinit.element6, i32 1
  store i8 32, ptr %arrayinit.element7, align 1
  %arrayinit.element8 = getelementptr inbounds i8, ptr %arrayinit.element7, i32 1
  %a5 = load i8, ptr %new_val, align 1
  store i8 %a5, ptr %arrayinit.element8, align 1
  %arrayinit.element9 = getelementptr inbounds i8, ptr %arrayinit.element8, i32 1
  store i8 -3, ptr %arrayinit.element9, align 1
  %arrayinit.element10 = getelementptr inbounds i8, ptr %arrayinit.element9, i32 1
  store i8 2, ptr %arrayinit.element10, align 1
  %arrayinit.element11 = getelementptr inbounds i8, ptr %arrayinit.element10, i32 1
  store i8 0, ptr %arrayinit.element11, align 1
  %arrayinit.element12 = getelementptr inbounds i8, ptr %arrayinit.element11, i32 1
  store i8 33, ptr %arrayinit.element12, align 1
  %arrayinit.element13 = getelementptr inbounds i8, ptr %arrayinit.element12, i32 1
  %a6 = load i8, ptr %simd, align 1
  store i8 %a6, ptr %arrayinit.element13, align 1
  %arrayinit.element14 = getelementptr inbounds i8, ptr %arrayinit.element13, i32 1
  store i8 32, ptr %arrayinit.element14, align 1
  %arrayinit.element15 = getelementptr inbounds i8, ptr %arrayinit.element14, i32 1
  %a7 = load i8, ptr %new_val, align 1
  store i8 %a7, ptr %arrayinit.element15, align 1
  %arrayinit.element16 = getelementptr inbounds i8, ptr %arrayinit.element15, i32 1
  store i8 32, ptr %arrayinit.element16, align 1
  %arrayinit.element17 = getelementptr inbounds i8, ptr %arrayinit.element16, i32 1
  %a8 = load i8, ptr %simd, align 1
  store i8 %a8, ptr %arrayinit.element17, align 1
  %arrayinit.element18 = getelementptr inbounds i8, ptr %arrayinit.element17, i32 1
  store i8 -3, ptr %arrayinit.element18, align 1
  %arrayinit.element19 = getelementptr inbounds i8, ptr %arrayinit.element18, i32 1
  store i8 1, ptr %arrayinit.element19, align 1
  %arrayinit.element20 = getelementptr inbounds i8, ptr %arrayinit.element19, i32 1
  store i8 0, ptr %arrayinit.element20, align 1
  %arrayinit.element21 = getelementptr inbounds i8, ptr %arrayinit.element20, i32 1
  store i8 92, ptr %arrayinit.element21, align 1
  %arrayinit.element22 = getelementptr inbounds i8, ptr %arrayinit.element21, i32 1
  store i8 4, ptr %arrayinit.element22, align 1
  %arrayinit.element23 = getelementptr inbounds i8, ptr %arrayinit.element22, i32 1
  store i8 64, ptr %arrayinit.element23, align 1
  %arrayinit.element24 = getelementptr inbounds i8, ptr %arrayinit.element23, i32 1
  store i8 65, ptr %arrayinit.element24, align 1
  %arrayinit.element25 = getelementptr inbounds i8, ptr %arrayinit.element24, i32 1
  store i8 0, ptr %arrayinit.element25, align 1
  %arrayinit.element26 = getelementptr inbounds i8, ptr %arrayinit.element25, i32 1
  store i8 15, ptr %arrayinit.element26, align 1
  %arrayinit.element27 = getelementptr inbounds i8, ptr %arrayinit.element26, i32 1
  store i8 11, ptr %arrayinit.element27, align 1
  %arrayinit.element28 = getelementptr inbounds i8, ptr %arrayinit.element27, i32 1
  store i8 32, ptr %arrayinit.element28, align 1
  %arrayinit.element29 = getelementptr inbounds i8, ptr %arrayinit.element28, i32 1
  %a9 = load i8, ptr %old_val, align 1
  store i8 %a9, ptr %arrayinit.element29, align 1
  %arrayinit.element30 = getelementptr inbounds i8, ptr %arrayinit.element29, i32 1
  store i8 32, ptr %arrayinit.element30, align 1
  %arrayinit.element31 = getelementptr inbounds i8, ptr %arrayinit.element30, i32 1
  %a10 = load i8, ptr %simd, align 1
  store i8 %a10, ptr %arrayinit.element31, align 1
  %arrayinit.element32 = getelementptr inbounds i8, ptr %arrayinit.element31, i32 1
  store i8 -3, ptr %arrayinit.element32, align 1
  %arrayinit.element33 = getelementptr inbounds i8, ptr %arrayinit.element32, i32 1
  store i8 1, ptr %arrayinit.element33, align 1
  %arrayinit.element34 = getelementptr inbounds i8, ptr %arrayinit.element33, i32 1
  store i8 1, ptr %arrayinit.element34, align 1
  %arrayinit.element35 = getelementptr inbounds i8, ptr %arrayinit.element34, i32 1
  store i8 92, ptr %arrayinit.element35, align 1
  %arrayinit.element36 = getelementptr inbounds i8, ptr %arrayinit.element35, i32 1
  store i8 4, ptr %arrayinit.element36, align 1
  %arrayinit.element37 = getelementptr inbounds i8, ptr %arrayinit.element36, i32 1
  store i8 64, ptr %arrayinit.element37, align 1
  %arrayinit.element38 = getelementptr inbounds i8, ptr %arrayinit.element37, i32 1
  store i8 65, ptr %arrayinit.element38, align 1
  %arrayinit.element39 = getelementptr inbounds i8, ptr %arrayinit.element38, i32 1
  store i8 0, ptr %arrayinit.element39, align 1
  %arrayinit.element40 = getelementptr inbounds i8, ptr %arrayinit.element39, i32 1
  store i8 15, ptr %arrayinit.element40, align 1
  %arrayinit.element41 = getelementptr inbounds i8, ptr %arrayinit.element40, i32 1
  store i8 11, ptr %arrayinit.element41, align 1
  %arrayinit.element42 = getelementptr inbounds i8, ptr %arrayinit.element41, i32 1
  store i8 32, ptr %arrayinit.element42, align 1
  %arrayinit.element43 = getelementptr inbounds i8, ptr %arrayinit.element42, i32 1
  %a11 = load i8, ptr %old_val, align 1
  store i8 %a11, ptr %arrayinit.element43, align 1
  %arrayinit.element44 = getelementptr inbounds i8, ptr %arrayinit.element43, i32 1
  store i8 32, ptr %arrayinit.element44, align 1
  %arrayinit.element45 = getelementptr inbounds i8, ptr %arrayinit.element44, i32 1
  %a12 = load i8, ptr %simd, align 1
  store i8 %a12, ptr %arrayinit.element45, align 1
  %arrayinit.element46 = getelementptr inbounds i8, ptr %arrayinit.element45, i32 1
  store i8 -3, ptr %arrayinit.element46, align 1
  %arrayinit.element47 = getelementptr inbounds i8, ptr %arrayinit.element46, i32 1
  store i8 1, ptr %arrayinit.element47, align 1
  %arrayinit.element48 = getelementptr inbounds i8, ptr %arrayinit.element47, i32 1
  store i8 2, ptr %arrayinit.element48, align 1
  %arrayinit.element49 = getelementptr inbounds i8, ptr %arrayinit.element48, i32 1
  store i8 92, ptr %arrayinit.element49, align 1
  %arrayinit.element50 = getelementptr inbounds i8, ptr %arrayinit.element49, i32 1
  store i8 4, ptr %arrayinit.element50, align 1
  %arrayinit.element51 = getelementptr inbounds i8, ptr %arrayinit.element50, i32 1
  store i8 64, ptr %arrayinit.element51, align 1
  %arrayinit.element52 = getelementptr inbounds i8, ptr %arrayinit.element51, i32 1
  store i8 65, ptr %arrayinit.element52, align 1
  %arrayinit.element53 = getelementptr inbounds i8, ptr %arrayinit.element52, i32 1
  store i8 0, ptr %arrayinit.element53, align 1
  %arrayinit.element54 = getelementptr inbounds i8, ptr %arrayinit.element53, i32 1
  store i8 15, ptr %arrayinit.element54, align 1
  %arrayinit.element55 = getelementptr inbounds i8, ptr %arrayinit.element54, i32 1
  store i8 11, ptr %arrayinit.element55, align 1
  %arrayinit.element56 = getelementptr inbounds i8, ptr %arrayinit.element55, i32 1
  store i8 32, ptr %arrayinit.element56, align 1
  %arrayinit.element57 = getelementptr inbounds i8, ptr %arrayinit.element56, i32 1
  %a13 = load i8, ptr %old_val, align 1
  store i8 %a13, ptr %arrayinit.element57, align 1
  %arrayinit.element58 = getelementptr inbounds i8, ptr %arrayinit.element57, i32 1
  store i8 32, ptr %arrayinit.element58, align 1
  %arrayinit.element59 = getelementptr inbounds i8, ptr %arrayinit.element58, i32 1
  %a14 = load i8, ptr %simd, align 1
  store i8 %a14, ptr %arrayinit.element59, align 1
  %arrayinit.element60 = getelementptr inbounds i8, ptr %arrayinit.element59, i32 1
  store i8 -3, ptr %arrayinit.element60, align 1
  %arrayinit.element61 = getelementptr inbounds i8, ptr %arrayinit.element60, i32 1
  store i8 1, ptr %arrayinit.element61, align 1
  %arrayinit.element62 = getelementptr inbounds i8, ptr %arrayinit.element61, i32 1
  store i8 3, ptr %arrayinit.element62, align 1
  %arrayinit.element63 = getelementptr inbounds i8, ptr %arrayinit.element62, i32 1
  store i8 92, ptr %arrayinit.element63, align 1
  %arrayinit.element64 = getelementptr inbounds i8, ptr %arrayinit.element63, i32 1
  store i8 4, ptr %arrayinit.element64, align 1
  %arrayinit.element65 = getelementptr inbounds i8, ptr %arrayinit.element64, i32 1
  store i8 64, ptr %arrayinit.element65, align 1
  %arrayinit.element66 = getelementptr inbounds i8, ptr %arrayinit.element65, i32 1
  store i8 65, ptr %arrayinit.element66, align 1
  %arrayinit.element67 = getelementptr inbounds i8, ptr %arrayinit.element66, i32 1
  store i8 0, ptr %arrayinit.element67, align 1
  %arrayinit.element68 = getelementptr inbounds i8, ptr %arrayinit.element67, i32 1
  store i8 15, ptr %arrayinit.element68, align 1
  %arrayinit.element69 = getelementptr inbounds i8, ptr %arrayinit.element68, i32 1
  store i8 11, ptr %arrayinit.element69, align 1
  %arrayinit.element70 = getelementptr inbounds i8, ptr %arrayinit.element69, i32 1
  store i8 32, ptr %arrayinit.element70, align 1
  %arrayinit.element71 = getelementptr inbounds i8, ptr %arrayinit.element70, i32 1
  %a15 = load i8, ptr %simd, align 1
  store i8 %a15, ptr %arrayinit.element71, align 1
  %arrayinit.element72 = getelementptr inbounds i8, ptr %arrayinit.element71, i32 1
  store i8 32, ptr %arrayinit.element72, align 1
  %arrayinit.element73 = getelementptr inbounds i8, ptr %arrayinit.element72, i32 1
  %a16 = load i8, ptr %new_val, align 1
  store i8 %a16, ptr %arrayinit.element73, align 1
  %arrayinit.element74 = getelementptr inbounds i8, ptr %arrayinit.element73, i32 1
  store i8 -3, ptr %arrayinit.element74, align 1
  %arrayinit.element75 = getelementptr inbounds i8, ptr %arrayinit.element74, i32 1
  store i8 2, ptr %arrayinit.element75, align 1
  %arrayinit.element76 = getelementptr inbounds i8, ptr %arrayinit.element75, i32 1
  store i8 1, ptr %arrayinit.element76, align 1
  %arrayinit.element77 = getelementptr inbounds i8, ptr %arrayinit.element76, i32 1
  store i8 33, ptr %arrayinit.element77, align 1
  %arrayinit.element78 = getelementptr inbounds i8, ptr %arrayinit.element77, i32 1
  %a17 = load i8, ptr %simd, align 1
  store i8 %a17, ptr %arrayinit.element78, align 1
  %arrayinit.element79 = getelementptr inbounds i8, ptr %arrayinit.element78, i32 1
  store i8 32, ptr %arrayinit.element79, align 1
  %arrayinit.element80 = getelementptr inbounds i8, ptr %arrayinit.element79, i32 1
  %a18 = load i8, ptr %new_val, align 1
  store i8 %a18, ptr %arrayinit.element80, align 1
  %arrayinit.element81 = getelementptr inbounds i8, ptr %arrayinit.element80, i32 1
  store i8 32, ptr %arrayinit.element81, align 1
  %arrayinit.element82 = getelementptr inbounds i8, ptr %arrayinit.element81, i32 1
  %a19 = load i8, ptr %simd, align 1
  store i8 %a19, ptr %arrayinit.element82, align 1
  %arrayinit.element83 = getelementptr inbounds i8, ptr %arrayinit.element82, i32 1
  store i8 -3, ptr %arrayinit.element83, align 1
  %arrayinit.element84 = getelementptr inbounds i8, ptr %arrayinit.element83, i32 1
  store i8 1, ptr %arrayinit.element84, align 1
  %arrayinit.element85 = getelementptr inbounds i8, ptr %arrayinit.element84, i32 1
  store i8 0, ptr %arrayinit.element85, align 1
  %arrayinit.element86 = getelementptr inbounds i8, ptr %arrayinit.element85, i32 1
  store i8 92, ptr %arrayinit.element86, align 1
  %arrayinit.element87 = getelementptr inbounds i8, ptr %arrayinit.element86, i32 1
  store i8 4, ptr %arrayinit.element87, align 1
  %arrayinit.element88 = getelementptr inbounds i8, ptr %arrayinit.element87, i32 1
  store i8 64, ptr %arrayinit.element88, align 1
  %arrayinit.element89 = getelementptr inbounds i8, ptr %arrayinit.element88, i32 1
  store i8 65, ptr %arrayinit.element89, align 1
  %arrayinit.element90 = getelementptr inbounds i8, ptr %arrayinit.element89, i32 1
  store i8 0, ptr %arrayinit.element90, align 1
  %arrayinit.element91 = getelementptr inbounds i8, ptr %arrayinit.element90, i32 1
  store i8 15, ptr %arrayinit.element91, align 1
  %arrayinit.element92 = getelementptr inbounds i8, ptr %arrayinit.element91, i32 1
  store i8 11, ptr %arrayinit.element92, align 1
  %arrayinit.element93 = getelementptr inbounds i8, ptr %arrayinit.element92, i32 1
  store i8 32, ptr %arrayinit.element93, align 1
  %arrayinit.element94 = getelementptr inbounds i8, ptr %arrayinit.element93, i32 1
  %a20 = load i8, ptr %new_val, align 1
  store i8 %a20, ptr %arrayinit.element94, align 1
  %arrayinit.element95 = getelementptr inbounds i8, ptr %arrayinit.element94, i32 1
  store i8 32, ptr %arrayinit.element95, align 1
  %arrayinit.element96 = getelementptr inbounds i8, ptr %arrayinit.element95, i32 1
  %a21 = load i8, ptr %simd, align 1
  store i8 %a21, ptr %arrayinit.element96, align 1
  %arrayinit.element97 = getelementptr inbounds i8, ptr %arrayinit.element96, i32 1
  store i8 -3, ptr %arrayinit.element97, align 1
  %arrayinit.element98 = getelementptr inbounds i8, ptr %arrayinit.element97, i32 1
  store i8 1, ptr %arrayinit.element98, align 1
  %arrayinit.element99 = getelementptr inbounds i8, ptr %arrayinit.element98, i32 1
  store i8 1, ptr %arrayinit.element99, align 1
  %arrayinit.element100 = getelementptr inbounds i8, ptr %arrayinit.element99, i32 1
  store i8 92, ptr %arrayinit.element100, align 1
  %arrayinit.element101 = getelementptr inbounds i8, ptr %arrayinit.element100, i32 1
  store i8 4, ptr %arrayinit.element101, align 1
  %arrayinit.element102 = getelementptr inbounds i8, ptr %arrayinit.element101, i32 1
  store i8 64, ptr %arrayinit.element102, align 1
  %arrayinit.element103 = getelementptr inbounds i8, ptr %arrayinit.element102, i32 1
  store i8 65, ptr %arrayinit.element103, align 1
  %arrayinit.element104 = getelementptr inbounds i8, ptr %arrayinit.element103, i32 1
  store i8 0, ptr %arrayinit.element104, align 1
  %arrayinit.element105 = getelementptr inbounds i8, ptr %arrayinit.element104, i32 1
  store i8 15, ptr %arrayinit.element105, align 1
  %arrayinit.element106 = getelementptr inbounds i8, ptr %arrayinit.element105, i32 1
  store i8 11, ptr %arrayinit.element106, align 1
  %arrayinit.element107 = getelementptr inbounds i8, ptr %arrayinit.element106, i32 1
  store i8 32, ptr %arrayinit.element107, align 1
  %arrayinit.element108 = getelementptr inbounds i8, ptr %arrayinit.element107, i32 1
  %a22 = load i8, ptr %old_val, align 1
  store i8 %a22, ptr %arrayinit.element108, align 1
  %arrayinit.element109 = getelementptr inbounds i8, ptr %arrayinit.element108, i32 1
  store i8 32, ptr %arrayinit.element109, align 1
  %arrayinit.element110 = getelementptr inbounds i8, ptr %arrayinit.element109, i32 1
  %a23 = load i8, ptr %simd, align 1
  store i8 %a23, ptr %arrayinit.element110, align 1
  %arrayinit.element111 = getelementptr inbounds i8, ptr %arrayinit.element110, i32 1
  store i8 -3, ptr %arrayinit.element111, align 1
  %arrayinit.element112 = getelementptr inbounds i8, ptr %arrayinit.element111, i32 1
  store i8 1, ptr %arrayinit.element112, align 1
  %arrayinit.element113 = getelementptr inbounds i8, ptr %arrayinit.element112, i32 1
  store i8 2, ptr %arrayinit.element113, align 1
  %arrayinit.element114 = getelementptr inbounds i8, ptr %arrayinit.element113, i32 1
  store i8 92, ptr %arrayinit.element114, align 1
  %arrayinit.element115 = getelementptr inbounds i8, ptr %arrayinit.element114, i32 1
  store i8 4, ptr %arrayinit.element115, align 1
  %arrayinit.element116 = getelementptr inbounds i8, ptr %arrayinit.element115, i32 1
  store i8 64, ptr %arrayinit.element116, align 1
  %arrayinit.element117 = getelementptr inbounds i8, ptr %arrayinit.element116, i32 1
  store i8 65, ptr %arrayinit.element117, align 1
  %arrayinit.element118 = getelementptr inbounds i8, ptr %arrayinit.element117, i32 1
  store i8 0, ptr %arrayinit.element118, align 1
  %arrayinit.element119 = getelementptr inbounds i8, ptr %arrayinit.element118, i32 1
  store i8 15, ptr %arrayinit.element119, align 1
  %arrayinit.element120 = getelementptr inbounds i8, ptr %arrayinit.element119, i32 1
  store i8 11, ptr %arrayinit.element120, align 1
  %arrayinit.element121 = getelementptr inbounds i8, ptr %arrayinit.element120, i32 1
  store i8 32, ptr %arrayinit.element121, align 1
  %arrayinit.element122 = getelementptr inbounds i8, ptr %arrayinit.element121, i32 1
  %a24 = load i8, ptr %old_val, align 1
  store i8 %a24, ptr %arrayinit.element122, align 1
  %arrayinit.element123 = getelementptr inbounds i8, ptr %arrayinit.element122, i32 1
  store i8 32, ptr %arrayinit.element123, align 1
  %arrayinit.element124 = getelementptr inbounds i8, ptr %arrayinit.element123, i32 1
  %a25 = load i8, ptr %simd, align 1
  store i8 %a25, ptr %arrayinit.element124, align 1
  %arrayinit.element125 = getelementptr inbounds i8, ptr %arrayinit.element124, i32 1
  store i8 -3, ptr %arrayinit.element125, align 1
  %arrayinit.element126 = getelementptr inbounds i8, ptr %arrayinit.element125, i32 1
  store i8 1, ptr %arrayinit.element126, align 1
  %arrayinit.element127 = getelementptr inbounds i8, ptr %arrayinit.element126, i32 1
  store i8 3, ptr %arrayinit.element127, align 1
  %arrayinit.element128 = getelementptr inbounds i8, ptr %arrayinit.element127, i32 1
  store i8 92, ptr %arrayinit.element128, align 1
  %arrayinit.element129 = getelementptr inbounds i8, ptr %arrayinit.element128, i32 1
  store i8 4, ptr %arrayinit.element129, align 1
  %arrayinit.element130 = getelementptr inbounds i8, ptr %arrayinit.element129, i32 1
  store i8 64, ptr %arrayinit.element130, align 1
  %arrayinit.element131 = getelementptr inbounds i8, ptr %arrayinit.element130, i32 1
  store i8 65, ptr %arrayinit.element131, align 1
  %arrayinit.element132 = getelementptr inbounds i8, ptr %arrayinit.element131, i32 1
  store i8 0, ptr %arrayinit.element132, align 1
  %arrayinit.element133 = getelementptr inbounds i8, ptr %arrayinit.element132, i32 1
  store i8 15, ptr %arrayinit.element133, align 1
  %arrayinit.element134 = getelementptr inbounds i8, ptr %arrayinit.element133, i32 1
  store i8 11, ptr %arrayinit.element134, align 1
  %arrayinit.element135 = getelementptr inbounds i8, ptr %arrayinit.element134, i32 1
  store i8 32, ptr %arrayinit.element135, align 1
  %arrayinit.element136 = getelementptr inbounds i8, ptr %arrayinit.element135, i32 1
  %a26 = load i8, ptr %simd, align 1
  store i8 %a26, ptr %arrayinit.element136, align 1
  %arrayinit.element137 = getelementptr inbounds i8, ptr %arrayinit.element136, i32 1
  store i8 32, ptr %arrayinit.element137, align 1
  %arrayinit.element138 = getelementptr inbounds i8, ptr %arrayinit.element137, i32 1
  %a27 = load i8, ptr %new_val, align 1
  store i8 %a27, ptr %arrayinit.element138, align 1
  %arrayinit.element139 = getelementptr inbounds i8, ptr %arrayinit.element138, i32 1
  store i8 -3, ptr %arrayinit.element139, align 1
  %arrayinit.element140 = getelementptr inbounds i8, ptr %arrayinit.element139, i32 1
  store i8 2, ptr %arrayinit.element140, align 1
  %arrayinit.element141 = getelementptr inbounds i8, ptr %arrayinit.element140, i32 1
  store i8 2, ptr %arrayinit.element141, align 1
  %arrayinit.element142 = getelementptr inbounds i8, ptr %arrayinit.element141, i32 1
  store i8 33, ptr %arrayinit.element142, align 1
  %arrayinit.element143 = getelementptr inbounds i8, ptr %arrayinit.element142, i32 1
  %a28 = load i8, ptr %simd, align 1
  store i8 %a28, ptr %arrayinit.element143, align 1
  %arrayinit.element144 = getelementptr inbounds i8, ptr %arrayinit.element143, i32 1
  store i8 32, ptr %arrayinit.element144, align 1
  %arrayinit.element145 = getelementptr inbounds i8, ptr %arrayinit.element144, i32 1
  %a29 = load i8, ptr %new_val, align 1
  store i8 %a29, ptr %arrayinit.element145, align 1
  %arrayinit.element146 = getelementptr inbounds i8, ptr %arrayinit.element145, i32 1
  store i8 32, ptr %arrayinit.element146, align 1
  %arrayinit.element147 = getelementptr inbounds i8, ptr %arrayinit.element146, i32 1
  %a30 = load i8, ptr %simd, align 1
  store i8 %a30, ptr %arrayinit.element147, align 1
  %arrayinit.element148 = getelementptr inbounds i8, ptr %arrayinit.element147, i32 1
  store i8 -3, ptr %arrayinit.element148, align 1
  %arrayinit.element149 = getelementptr inbounds i8, ptr %arrayinit.element148, i32 1
  store i8 1, ptr %arrayinit.element149, align 1
  %arrayinit.element150 = getelementptr inbounds i8, ptr %arrayinit.element149, i32 1
  store i8 0, ptr %arrayinit.element150, align 1
  %arrayinit.element151 = getelementptr inbounds i8, ptr %arrayinit.element150, i32 1
  store i8 92, ptr %arrayinit.element151, align 1
  %arrayinit.element152 = getelementptr inbounds i8, ptr %arrayinit.element151, i32 1
  store i8 4, ptr %arrayinit.element152, align 1
  %arrayinit.element153 = getelementptr inbounds i8, ptr %arrayinit.element152, i32 1
  store i8 64, ptr %arrayinit.element153, align 1
  %arrayinit.element154 = getelementptr inbounds i8, ptr %arrayinit.element153, i32 1
  store i8 65, ptr %arrayinit.element154, align 1
  %arrayinit.element155 = getelementptr inbounds i8, ptr %arrayinit.element154, i32 1
  store i8 0, ptr %arrayinit.element155, align 1
  %arrayinit.element156 = getelementptr inbounds i8, ptr %arrayinit.element155, i32 1
  store i8 15, ptr %arrayinit.element156, align 1
  %arrayinit.element157 = getelementptr inbounds i8, ptr %arrayinit.element156, i32 1
  store i8 11, ptr %arrayinit.element157, align 1
  %arrayinit.element158 = getelementptr inbounds i8, ptr %arrayinit.element157, i32 1
  store i8 32, ptr %arrayinit.element158, align 1
  %arrayinit.element159 = getelementptr inbounds i8, ptr %arrayinit.element158, i32 1
  %a31 = load i8, ptr %new_val, align 1
  store i8 %a31, ptr %arrayinit.element159, align 1
  %arrayinit.element160 = getelementptr inbounds i8, ptr %arrayinit.element159, i32 1
  store i8 32, ptr %arrayinit.element160, align 1
  %arrayinit.element161 = getelementptr inbounds i8, ptr %arrayinit.element160, i32 1
  %a32 = load i8, ptr %simd, align 1
  store i8 %a32, ptr %arrayinit.element161, align 1
  %arrayinit.element162 = getelementptr inbounds i8, ptr %arrayinit.element161, i32 1
  store i8 -3, ptr %arrayinit.element162, align 1
  %arrayinit.element163 = getelementptr inbounds i8, ptr %arrayinit.element162, i32 1
  store i8 1, ptr %arrayinit.element163, align 1
  %arrayinit.element164 = getelementptr inbounds i8, ptr %arrayinit.element163, i32 1
  store i8 1, ptr %arrayinit.element164, align 1
  %arrayinit.element165 = getelementptr inbounds i8, ptr %arrayinit.element164, i32 1
  store i8 92, ptr %arrayinit.element165, align 1
  %arrayinit.element166 = getelementptr inbounds i8, ptr %arrayinit.element165, i32 1
  store i8 4, ptr %arrayinit.element166, align 1
  %arrayinit.element167 = getelementptr inbounds i8, ptr %arrayinit.element166, i32 1
  store i8 64, ptr %arrayinit.element167, align 1
  %arrayinit.element168 = getelementptr inbounds i8, ptr %arrayinit.element167, i32 1
  store i8 65, ptr %arrayinit.element168, align 1
  %arrayinit.element169 = getelementptr inbounds i8, ptr %arrayinit.element168, i32 1
  store i8 0, ptr %arrayinit.element169, align 1
  %arrayinit.element170 = getelementptr inbounds i8, ptr %arrayinit.element169, i32 1
  store i8 15, ptr %arrayinit.element170, align 1
  %arrayinit.element171 = getelementptr inbounds i8, ptr %arrayinit.element170, i32 1
  store i8 11, ptr %arrayinit.element171, align 1
  %arrayinit.element172 = getelementptr inbounds i8, ptr %arrayinit.element171, i32 1
  store i8 32, ptr %arrayinit.element172, align 1
  %arrayinit.element173 = getelementptr inbounds i8, ptr %arrayinit.element172, i32 1
  %a33 = load i8, ptr %new_val, align 1
  store i8 %a33, ptr %arrayinit.element173, align 1
  %arrayinit.element174 = getelementptr inbounds i8, ptr %arrayinit.element173, i32 1
  store i8 32, ptr %arrayinit.element174, align 1
  %arrayinit.element175 = getelementptr inbounds i8, ptr %arrayinit.element174, i32 1
  %a34 = load i8, ptr %simd, align 1
  store i8 %a34, ptr %arrayinit.element175, align 1
  %arrayinit.element176 = getelementptr inbounds i8, ptr %arrayinit.element175, i32 1
  store i8 -3, ptr %arrayinit.element176, align 1
  %arrayinit.element177 = getelementptr inbounds i8, ptr %arrayinit.element176, i32 1
  store i8 1, ptr %arrayinit.element177, align 1
  %arrayinit.element178 = getelementptr inbounds i8, ptr %arrayinit.element177, i32 1
  store i8 2, ptr %arrayinit.element178, align 1
  %arrayinit.element179 = getelementptr inbounds i8, ptr %arrayinit.element178, i32 1
  store i8 92, ptr %arrayinit.element179, align 1
  %arrayinit.element180 = getelementptr inbounds i8, ptr %arrayinit.element179, i32 1
  store i8 4, ptr %arrayinit.element180, align 1
  %arrayinit.element181 = getelementptr inbounds i8, ptr %arrayinit.element180, i32 1
  store i8 64, ptr %arrayinit.element181, align 1
  %arrayinit.element182 = getelementptr inbounds i8, ptr %arrayinit.element181, i32 1
  store i8 65, ptr %arrayinit.element182, align 1
  %arrayinit.element183 = getelementptr inbounds i8, ptr %arrayinit.element182, i32 1
  store i8 0, ptr %arrayinit.element183, align 1
  %arrayinit.element184 = getelementptr inbounds i8, ptr %arrayinit.element183, i32 1
  store i8 15, ptr %arrayinit.element184, align 1
  %arrayinit.element185 = getelementptr inbounds i8, ptr %arrayinit.element184, i32 1
  store i8 11, ptr %arrayinit.element185, align 1
  %arrayinit.element186 = getelementptr inbounds i8, ptr %arrayinit.element185, i32 1
  store i8 32, ptr %arrayinit.element186, align 1
  %arrayinit.element187 = getelementptr inbounds i8, ptr %arrayinit.element186, i32 1
  %a35 = load i8, ptr %old_val, align 1
  store i8 %a35, ptr %arrayinit.element187, align 1
  %arrayinit.element188 = getelementptr inbounds i8, ptr %arrayinit.element187, i32 1
  store i8 32, ptr %arrayinit.element188, align 1
  %arrayinit.element189 = getelementptr inbounds i8, ptr %arrayinit.element188, i32 1
  %a36 = load i8, ptr %simd, align 1
  store i8 %a36, ptr %arrayinit.element189, align 1
  %arrayinit.element190 = getelementptr inbounds i8, ptr %arrayinit.element189, i32 1
  store i8 -3, ptr %arrayinit.element190, align 1
  %arrayinit.element191 = getelementptr inbounds i8, ptr %arrayinit.element190, i32 1
  store i8 1, ptr %arrayinit.element191, align 1
  %arrayinit.element192 = getelementptr inbounds i8, ptr %arrayinit.element191, i32 1
  store i8 3, ptr %arrayinit.element192, align 1
  %arrayinit.element193 = getelementptr inbounds i8, ptr %arrayinit.element192, i32 1
  store i8 92, ptr %arrayinit.element193, align 1
  %arrayinit.element194 = getelementptr inbounds i8, ptr %arrayinit.element193, i32 1
  store i8 4, ptr %arrayinit.element194, align 1
  %arrayinit.element195 = getelementptr inbounds i8, ptr %arrayinit.element194, i32 1
  store i8 64, ptr %arrayinit.element195, align 1
  %arrayinit.element196 = getelementptr inbounds i8, ptr %arrayinit.element195, i32 1
  store i8 65, ptr %arrayinit.element196, align 1
  %arrayinit.element197 = getelementptr inbounds i8, ptr %arrayinit.element196, i32 1
  store i8 0, ptr %arrayinit.element197, align 1
  %arrayinit.element198 = getelementptr inbounds i8, ptr %arrayinit.element197, i32 1
  store i8 15, ptr %arrayinit.element198, align 1
  %arrayinit.element199 = getelementptr inbounds i8, ptr %arrayinit.element198, i32 1
  store i8 11, ptr %arrayinit.element199, align 1
  %arrayinit.element200 = getelementptr inbounds i8, ptr %arrayinit.element199, i32 1
  store i8 32, ptr %arrayinit.element200, align 1
  %arrayinit.element201 = getelementptr inbounds i8, ptr %arrayinit.element200, i32 1
  %a37 = load i8, ptr %simd, align 1
  store i8 %a37, ptr %arrayinit.element201, align 1
  %arrayinit.element202 = getelementptr inbounds i8, ptr %arrayinit.element201, i32 1
  store i8 32, ptr %arrayinit.element202, align 1
  %arrayinit.element203 = getelementptr inbounds i8, ptr %arrayinit.element202, i32 1
  %a38 = load i8, ptr %new_val, align 1
  store i8 %a38, ptr %arrayinit.element203, align 1
  %arrayinit.element204 = getelementptr inbounds i8, ptr %arrayinit.element203, i32 1
  store i8 -3, ptr %arrayinit.element204, align 1
  %arrayinit.element205 = getelementptr inbounds i8, ptr %arrayinit.element204, i32 1
  store i8 2, ptr %arrayinit.element205, align 1
  %arrayinit.element206 = getelementptr inbounds i8, ptr %arrayinit.element205, i32 1
  store i8 3, ptr %arrayinit.element206, align 1
  %arrayinit.element207 = getelementptr inbounds i8, ptr %arrayinit.element206, i32 1
  store i8 33, ptr %arrayinit.element207, align 1
  %arrayinit.element208 = getelementptr inbounds i8, ptr %arrayinit.element207, i32 1
  %a39 = load i8, ptr %simd, align 1
  store i8 %a39, ptr %arrayinit.element208, align 1
  %arrayinit.element209 = getelementptr inbounds i8, ptr %arrayinit.element208, i32 1
  store i8 32, ptr %arrayinit.element209, align 1
  %arrayinit.element210 = getelementptr inbounds i8, ptr %arrayinit.element209, i32 1
  %a40 = load i8, ptr %new_val, align 1
  store i8 %a40, ptr %arrayinit.element210, align 1
  %arrayinit.element211 = getelementptr inbounds i8, ptr %arrayinit.element210, i32 1
  store i8 32, ptr %arrayinit.element211, align 1
  %arrayinit.element212 = getelementptr inbounds i8, ptr %arrayinit.element211, i32 1
  %a41 = load i8, ptr %simd, align 1
  store i8 %a41, ptr %arrayinit.element212, align 1
  %arrayinit.element213 = getelementptr inbounds i8, ptr %arrayinit.element212, i32 1
  store i8 -3, ptr %arrayinit.element213, align 1
  %arrayinit.element214 = getelementptr inbounds i8, ptr %arrayinit.element213, i32 1
  store i8 1, ptr %arrayinit.element214, align 1
  %arrayinit.element215 = getelementptr inbounds i8, ptr %arrayinit.element214, i32 1
  store i8 0, ptr %arrayinit.element215, align 1
  %arrayinit.element216 = getelementptr inbounds i8, ptr %arrayinit.element215, i32 1
  store i8 92, ptr %arrayinit.element216, align 1
  %arrayinit.element217 = getelementptr inbounds i8, ptr %arrayinit.element216, i32 1
  store i8 4, ptr %arrayinit.element217, align 1
  %arrayinit.element218 = getelementptr inbounds i8, ptr %arrayinit.element217, i32 1
  store i8 64, ptr %arrayinit.element218, align 1
  %arrayinit.element219 = getelementptr inbounds i8, ptr %arrayinit.element218, i32 1
  store i8 65, ptr %arrayinit.element219, align 1
  %arrayinit.element220 = getelementptr inbounds i8, ptr %arrayinit.element219, i32 1
  store i8 0, ptr %arrayinit.element220, align 1
  %arrayinit.element221 = getelementptr inbounds i8, ptr %arrayinit.element220, i32 1
  store i8 15, ptr %arrayinit.element221, align 1
  %arrayinit.element222 = getelementptr inbounds i8, ptr %arrayinit.element221, i32 1
  store i8 11, ptr %arrayinit.element222, align 1
  %arrayinit.element223 = getelementptr inbounds i8, ptr %arrayinit.element222, i32 1
  store i8 32, ptr %arrayinit.element223, align 1
  %arrayinit.element224 = getelementptr inbounds i8, ptr %arrayinit.element223, i32 1
  %a42 = load i8, ptr %new_val, align 1
  store i8 %a42, ptr %arrayinit.element224, align 1
  %arrayinit.element225 = getelementptr inbounds i8, ptr %arrayinit.element224, i32 1
  store i8 32, ptr %arrayinit.element225, align 1
  %arrayinit.element226 = getelementptr inbounds i8, ptr %arrayinit.element225, i32 1
  %a43 = load i8, ptr %simd, align 1
  store i8 %a43, ptr %arrayinit.element226, align 1
  %arrayinit.element227 = getelementptr inbounds i8, ptr %arrayinit.element226, i32 1
  store i8 -3, ptr %arrayinit.element227, align 1
  %arrayinit.element228 = getelementptr inbounds i8, ptr %arrayinit.element227, i32 1
  store i8 1, ptr %arrayinit.element228, align 1
  %arrayinit.element229 = getelementptr inbounds i8, ptr %arrayinit.element228, i32 1
  store i8 1, ptr %arrayinit.element229, align 1
  %arrayinit.element230 = getelementptr inbounds i8, ptr %arrayinit.element229, i32 1
  store i8 92, ptr %arrayinit.element230, align 1
  %arrayinit.element231 = getelementptr inbounds i8, ptr %arrayinit.element230, i32 1
  store i8 4, ptr %arrayinit.element231, align 1
  %arrayinit.element232 = getelementptr inbounds i8, ptr %arrayinit.element231, i32 1
  store i8 64, ptr %arrayinit.element232, align 1
  %arrayinit.element233 = getelementptr inbounds i8, ptr %arrayinit.element232, i32 1
  store i8 65, ptr %arrayinit.element233, align 1
  %arrayinit.element234 = getelementptr inbounds i8, ptr %arrayinit.element233, i32 1
  store i8 0, ptr %arrayinit.element234, align 1
  %arrayinit.element235 = getelementptr inbounds i8, ptr %arrayinit.element234, i32 1
  store i8 15, ptr %arrayinit.element235, align 1
  %arrayinit.element236 = getelementptr inbounds i8, ptr %arrayinit.element235, i32 1
  store i8 11, ptr %arrayinit.element236, align 1
  %arrayinit.element237 = getelementptr inbounds i8, ptr %arrayinit.element236, i32 1
  store i8 32, ptr %arrayinit.element237, align 1
  %arrayinit.element238 = getelementptr inbounds i8, ptr %arrayinit.element237, i32 1
  %a44 = load i8, ptr %new_val, align 1
  store i8 %a44, ptr %arrayinit.element238, align 1
  %arrayinit.element239 = getelementptr inbounds i8, ptr %arrayinit.element238, i32 1
  store i8 32, ptr %arrayinit.element239, align 1
  %arrayinit.element240 = getelementptr inbounds i8, ptr %arrayinit.element239, i32 1
  %a45 = load i8, ptr %simd, align 1
  store i8 %a45, ptr %arrayinit.element240, align 1
  %arrayinit.element241 = getelementptr inbounds i8, ptr %arrayinit.element240, i32 1
  store i8 -3, ptr %arrayinit.element241, align 1
  %arrayinit.element242 = getelementptr inbounds i8, ptr %arrayinit.element241, i32 1
  store i8 1, ptr %arrayinit.element242, align 1
  %arrayinit.element243 = getelementptr inbounds i8, ptr %arrayinit.element242, i32 1
  store i8 2, ptr %arrayinit.element243, align 1
  %arrayinit.element244 = getelementptr inbounds i8, ptr %arrayinit.element243, i32 1
  store i8 92, ptr %arrayinit.element244, align 1
  %arrayinit.element245 = getelementptr inbounds i8, ptr %arrayinit.element244, i32 1
  store i8 4, ptr %arrayinit.element245, align 1
  %arrayinit.element246 = getelementptr inbounds i8, ptr %arrayinit.element245, i32 1
  store i8 64, ptr %arrayinit.element246, align 1
  %arrayinit.element247 = getelementptr inbounds i8, ptr %arrayinit.element246, i32 1
  store i8 65, ptr %arrayinit.element247, align 1
  %arrayinit.element248 = getelementptr inbounds i8, ptr %arrayinit.element247, i32 1
  store i8 0, ptr %arrayinit.element248, align 1
  %arrayinit.element249 = getelementptr inbounds i8, ptr %arrayinit.element248, i32 1
  store i8 15, ptr %arrayinit.element249, align 1
  %arrayinit.element250 = getelementptr inbounds i8, ptr %arrayinit.element249, i32 1
  store i8 11, ptr %arrayinit.element250, align 1
  %arrayinit.element251 = getelementptr inbounds i8, ptr %arrayinit.element250, i32 1
  store i8 32, ptr %arrayinit.element251, align 1
  %arrayinit.element252 = getelementptr inbounds i8, ptr %arrayinit.element251, i32 1
  %a46 = load i8, ptr %new_val, align 1
  store i8 %a46, ptr %arrayinit.element252, align 1
  %arrayinit.element253 = getelementptr inbounds i8, ptr %arrayinit.element252, i32 1
  store i8 32, ptr %arrayinit.element253, align 1
  %arrayinit.element254 = getelementptr inbounds i8, ptr %arrayinit.element253, i32 1
  %a47 = load i8, ptr %simd, align 1
  store i8 %a47, ptr %arrayinit.element254, align 1
  %arrayinit.element255 = getelementptr inbounds i8, ptr %arrayinit.element254, i32 1
  store i8 -3, ptr %arrayinit.element255, align 1
  %arrayinit.element256 = getelementptr inbounds i8, ptr %arrayinit.element255, i32 1
  store i8 1, ptr %arrayinit.element256, align 1
  %arrayinit.element257 = getelementptr inbounds i8, ptr %arrayinit.element256, i32 1
  store i8 3, ptr %arrayinit.element257, align 1
  %arrayinit.element258 = getelementptr inbounds i8, ptr %arrayinit.element257, i32 1
  store i8 92, ptr %arrayinit.element258, align 1
  %arrayinit.element259 = getelementptr inbounds i8, ptr %arrayinit.element258, i32 1
  store i8 4, ptr %arrayinit.element259, align 1
  %arrayinit.element260 = getelementptr inbounds i8, ptr %arrayinit.element259, i32 1
  store i8 64, ptr %arrayinit.element260, align 1
  %arrayinit.element261 = getelementptr inbounds i8, ptr %arrayinit.element260, i32 1
  store i8 65, ptr %arrayinit.element261, align 1
  %arrayinit.element262 = getelementptr inbounds i8, ptr %arrayinit.element261, i32 1
  store i8 0, ptr %arrayinit.element262, align 1
  %arrayinit.element263 = getelementptr inbounds i8, ptr %arrayinit.element262, i32 1
  store i8 15, ptr %arrayinit.element263, align 1
  %arrayinit.element264 = getelementptr inbounds i8, ptr %arrayinit.element263, i32 1
  store i8 11, ptr %arrayinit.element264, align 1
  %arrayinit.element265 = getelementptr inbounds i8, ptr %arrayinit.element264, i32 1
  store i8 65, ptr %arrayinit.element265, align 1
  %arrayinit.element266 = getelementptr inbounds i8, ptr %arrayinit.element265, i32 1
  store i8 1, ptr %arrayinit.element266, align 1
  %arrayinit.element267 = getelementptr inbounds i8, ptr %arrayinit.element266, i32 1
  store i8 15, ptr %arrayinit.element267, align 1
  %add.ptr = getelementptr inbounds i8, ptr %code, i32 269
  call void @g(ptr %code, ptr %add.ptr)
  ret void
}

declare void @g(ptr, ptr)

attributes #1 = { noinline nounwind optnone ssp uwtable }

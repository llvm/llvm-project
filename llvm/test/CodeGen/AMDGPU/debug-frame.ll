; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=asm -o - -emit-heterogeneous-dwarf-as-user-ops %s | FileCheck --check-prefixes=CHECK,WAVE64,GFX900 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -amdgpu-spill-vgpr-to-agpr=0 -filetype=asm -o - -emit-heterogeneous-dwarf-as-user-ops %s | FileCheck --check-prefixes=CHECK,WAVE64,GFX90A-V2A-DIS %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -amdgpu-spill-vgpr-to-agpr=1 -filetype=asm -o - -emit-heterogeneous-dwarf-as-user-ops %s | FileCheck --check-prefixes=CHECK,WAVE64,GFX90A-V2A-EN %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -filetype=asm -o - -emit-heterogeneous-dwarf-as-user-ops %s | FileCheck --check-prefixes=CHECK,WAVE32 %s

; CHECK-LABEL: kern1:
; CHECK: .cfi_startproc

; CHECK-NOT: .cfi_{{.*}}

; CHECK: %bb.0:
; DW_CFA_def_cfa_expression [0x0f]
;   BLOCK_LENGTH ULEB128(3)=[0x04]
;     DW_OP_lit0 [0x30]
;     DW_OP_lit6 [0x36]
;     DW_OP_LLVM_user [0xe9]
;     DW_OP_LLVM_form_aspace_address [0x02]
; CHECK-NEXT: .cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02
; PC_64 = 16
; CHECK-NEXT: .cfi_undefined 16

; CHECK-NOT: .cfi_{{.*}}

; CHECK: .cfi_endproc
define protected amdgpu_kernel void @kern1() #0 {
entry:
  ret void
}

; CHECK-LABEL: func_no_clobber:
; CHECK: .cfi_startproc

; CHECK-NOT: .cfi_{{.*}}

; CHECK: %bb.0:
; SGPR32 = 64
; CHECK-NEXT: .cfi_llvm_def_aspace_cfa 64, 0, 6
; CHECK-NEXT: .cfi_llvm_register_pair 16, 62, 32, 63, 32

; CHECK-NOT: .cfi_{{.*}}

; CHECK: .cfi_endproc
define hidden void @func_no_clobber() #0 {
entry:
  ret void
}

; CHECK-LABEL: {{^}}callee_need_to_spill_fp_to_memory:
; CHECK: .cfi_startproc

; SGPR33 = 65
; CHECK: s_mov_b32 [[FP_SCRATCH_COPY:s[0-9]+]], s33
; CHECK: s_mov_b32 s33, s32
; CHECK: v_mov_b32_e32 [[TMP_VGPR:v[0-9]+]], [[FP_SCRATCH_COPY]]
; GFX900: buffer_store_dword [[TMP_VGPR]], off, s[0:3], s33 offset:448 ; 4-byte Folded Spill
; GFX90A-V2A-DIS: buffer_store_dword [[TMP_VGPR]], off, s[0:3], s33 offset:448 ; 4-byte Folded Spill
; GFX90A-V2A-EN: buffer_store_dword [[TMP_VGPR]], off, s[0:3], s33 offset:320 ; 4-byte Folded Spill

; GFX900: .cfi_offset 65, 28672
; GFX90A-V2A-DIS: .cfi_offset 65, 28672
; GFX90A-V2A-EN: .cfi_offset 65, 20480
; WAVE32: .cfi_offset 65, 14336

; CHECK: .cfi_endproc
define void @callee_need_to_spill_fp_to_memory() #1 {
  call void asm sideeffect "; clobber nonpreserved SGPRs",
    "~{s4},~{s5},~{s6},~{s7},~{s8},~{s9}
    ,~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19}
    ,~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29}
    ,~{vcc}"()

  call void asm sideeffect "; clobber all VGPRs",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9}
    ,~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19}
    ,~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29}
    ,~{v30},~{v31},~{v32},~{v33},~{v34},~{v35},~{v36},~{v37},~{v38},~{v39}
    ,~{v40},~{v41},~{v42},~{v43},~{v44},~{v45},~{v46},~{v47},~{v48},~{v49}
    ,~{v50},~{v51},~{v52},~{v53},~{v54},~{v55},~{v56},~{v57},~{v58},~{v59}
    ,~{v60},~{v61},~{v62},~{v63},~{v64},~{v65},~{v66},~{v67},~{v68},~{v69}
    ,~{v70},~{v71},~{v72},~{v73},~{v74},~{v75},~{v76},~{v77},~{v78},~{v79}
    ,~{v80},~{v81},~{v82},~{v83},~{v84},~{v85},~{v86},~{v87},~{v88},~{v89}
    ,~{v90},~{v91},~{v92},~{v93},~{v94},~{v95},~{v96},~{v97},~{v98},~{v99}
    ,~{v100},~{v101},~{v102},~{v103},~{v104},~{v105},~{v106},~{v107},~{v108},~{v109}
    ,~{v110},~{v111},~{v112},~{v113},~{v114},~{v115},~{v116},~{v117},~{v118},~{v119}
    ,~{v120},~{v121},~{v122},~{v123},~{v124},~{v125},~{v126},~{v127},~{v128},~{v129}
    ,~{v130},~{v131},~{v132},~{v133},~{v134},~{v135},~{v136},~{v137},~{v138},~{v139}
    ,~{v140},~{v141},~{v142},~{v143},~{v144},~{v145},~{v146},~{v147},~{v148},~{v149}
    ,~{v150},~{v151},~{v152},~{v153},~{v154},~{v155},~{v156},~{v157},~{v158},~{v159}
    ,~{v160},~{v161},~{v162},~{v163},~{v164},~{v165},~{v166},~{v167},~{v168},~{v169}
    ,~{v170},~{v171},~{v172},~{v173},~{v174},~{v175},~{v176},~{v177},~{v178},~{v179}
    ,~{v180},~{v181},~{v182},~{v183},~{v184},~{v185},~{v186},~{v187},~{v188},~{v189}
    ,~{v190},~{v191},~{v192},~{v193},~{v194},~{v195},~{v196},~{v197},~{v198},~{v199}
    ,~{v200},~{v201},~{v202},~{v203},~{v204},~{v205},~{v206},~{v207},~{v208},~{v209}
    ,~{v210},~{v211},~{v212},~{v213},~{v214},~{v215},~{v216},~{v217},~{v218},~{v219}
    ,~{v220},~{v221},~{v222},~{v223},~{v224},~{v225},~{v226},~{v227},~{v228},~{v229}
    ,~{v230},~{v231},~{v232},~{v233},~{v234},~{v235},~{v236},~{v237},~{v238},~{v239}
    ,~{v240},~{v241},~{v242},~{v243},~{v244},~{v245},~{v246},~{v247},~{v248},~{v249}
    ,~{v250},~{v251},~{v252},~{v253},~{v254},~{v255}"()
  ret void
}

declare hidden void @ex() #0

; CHECK-LABEL: func_call_clobber:
; CHECK: .cfi_startproc

; CHECK-NOT: .cfi_{{.*}}

; CHECK: %bb.0:
; CHECK-NEXT: .cfi_llvm_def_aspace_cfa 64, 0, 6
; CHECK-NEXT: .cfi_llvm_register_pair 16, 62, 32, 63, 32

; VGPR0_wave64 = 2560
; WAVE64-NEXT: .cfi_undefined 2560
; WAVE64-NEXT: .cfi_undefined 2561
; WAVE64-NEXT: .cfi_undefined 2562
; WAVE64-NEXT: .cfi_undefined 2563
; WAVE64-NEXT: .cfi_undefined 2564
; WAVE64-NEXT: .cfi_undefined 2565
; WAVE64-NEXT: .cfi_undefined 2566
; WAVE64-NEXT: .cfi_undefined 2567
; WAVE64-NEXT: .cfi_undefined 2568
; WAVE64-NEXT: .cfi_undefined 2569
; WAVE64-NEXT: .cfi_undefined 2570
; WAVE64-NEXT: .cfi_undefined 2571
; WAVE64-NEXT: .cfi_undefined 2572
; WAVE64-NEXT: .cfi_undefined 2573
; WAVE64-NEXT: .cfi_undefined 2574
; WAVE64-NEXT: .cfi_undefined 2575
; WAVE64-NEXT: .cfi_undefined 2576
; WAVE64-NEXT: .cfi_undefined 2577
; WAVE64-NEXT: .cfi_undefined 2578
; WAVE64-NEXT: .cfi_undefined 2579
; WAVE64-NEXT: .cfi_undefined 2580
; WAVE64-NEXT: .cfi_undefined 2581
; WAVE64-NEXT: .cfi_undefined 2582
; WAVE64-NEXT: .cfi_undefined 2583
; WAVE64-NEXT: .cfi_undefined 2584
; WAVE64-NEXT: .cfi_undefined 2585
; WAVE64-NEXT: .cfi_undefined 2586
; WAVE64-NEXT: .cfi_undefined 2587
; WAVE64-NEXT: .cfi_undefined 2588
; WAVE64-NEXT: .cfi_undefined 2589
; WAVE64-NEXT: .cfi_undefined 2590
; WAVE64-NEXT: .cfi_undefined 2591
; WAVE64-NEXT: .cfi_undefined 2592
; WAVE64-NEXT: .cfi_undefined 2593
; WAVE64-NEXT: .cfi_undefined 2594
; WAVE64-NEXT: .cfi_undefined 2595
; WAVE64-NEXT: .cfi_undefined 2596
; WAVE64-NEXT: .cfi_undefined 2597
; WAVE64-NEXT: .cfi_undefined 2598
; WAVE64-NEXT: .cfi_undefined 2599

; VPGR48_wave64 = 2608
; WAVE64-NEXT: .cfi_undefined 2608
; WAVE64-NEXT: .cfi_undefined 2609
; WAVE64-NEXT: .cfi_undefined 2610
; WAVE64-NEXT: .cfi_undefined 2611
; WAVE64-NEXT: .cfi_undefined 2612
; WAVE64-NEXT: .cfi_undefined 2613
; WAVE64-NEXT: .cfi_undefined 2614
; WAVE64-NEXT: .cfi_undefined 2615

; WAVE64-NEXT: .cfi_undefined 2624
; WAVE64-NEXT: .cfi_undefined 2625
; WAVE64-NEXT: .cfi_undefined 2626
; WAVE64-NEXT: .cfi_undefined 2627
; WAVE64-NEXT: .cfi_undefined 2628
; WAVE64-NEXT: .cfi_undefined 2629
; WAVE64-NEXT: .cfi_undefined 2630
; WAVE64-NEXT: .cfi_undefined 2631

; WAVE64-NEXT: .cfi_undefined 2640
; WAVE64-NEXT: .cfi_undefined 2641
; WAVE64-NEXT: .cfi_undefined 2642
; WAVE64-NEXT: .cfi_undefined 2643
; WAVE64-NEXT: .cfi_undefined 2644
; WAVE64-NEXT: .cfi_undefined 2645
; WAVE64-NEXT: .cfi_undefined 2646
; WAVE64-NEXT: .cfi_undefined 2647

; WAVE64-NEXT: .cfi_undefined 2656
; WAVE64-NEXT: .cfi_undefined 2657
; WAVE64-NEXT: .cfi_undefined 2658
; WAVE64-NEXT: .cfi_undefined 2659
; WAVE64-NEXT: .cfi_undefined 2660
; WAVE64-NEXT: .cfi_undefined 2661
; WAVE64-NEXT: .cfi_undefined 2662
; WAVE64-NEXT: .cfi_undefined 2663

; WAVE64-NEXT: .cfi_undefined 2672
; WAVE64-NEXT: .cfi_undefined 2673
; WAVE64-NEXT: .cfi_undefined 2674
; WAVE64-NEXT: .cfi_undefined 2675
; WAVE64-NEXT: .cfi_undefined 2676
; WAVE64-NEXT: .cfi_undefined 2677
; WAVE64-NEXT: .cfi_undefined 2678
; WAVE64-NEXT: .cfi_undefined 2679

; WAVE64-NEXT: .cfi_undefined 2688
; WAVE64-NEXT: .cfi_undefined 2689
; WAVE64-NEXT: .cfi_undefined 2690
; WAVE64-NEXT: .cfi_undefined 2691
; WAVE64-NEXT: .cfi_undefined 2692
; WAVE64-NEXT: .cfi_undefined 2693
; WAVE64-NEXT: .cfi_undefined 2694
; WAVE64-NEXT: .cfi_undefined 2695

; WAVE64-NEXT: .cfi_undefined 2704
; WAVE64-NEXT: .cfi_undefined 2705
; WAVE64-NEXT: .cfi_undefined 2706
; WAVE64-NEXT: .cfi_undefined 2707
; WAVE64-NEXT: .cfi_undefined 2708
; WAVE64-NEXT: .cfi_undefined 2709
; WAVE64-NEXT: .cfi_undefined 2710
; WAVE64-NEXT: .cfi_undefined 2711

; WAVE64-NEXT: .cfi_undefined 2720
; WAVE64-NEXT: .cfi_undefined 2721
; WAVE64-NEXT: .cfi_undefined 2722
; WAVE64-NEXT: .cfi_undefined 2723
; WAVE64-NEXT: .cfi_undefined 2724
; WAVE64-NEXT: .cfi_undefined 2725
; WAVE64-NEXT: .cfi_undefined 2726
; WAVE64-NEXT: .cfi_undefined 2727

; WAVE64-NEXT: .cfi_undefined 2736
; WAVE64-NEXT: .cfi_undefined 2737
; WAVE64-NEXT: .cfi_undefined 2738
; WAVE64-NEXT: .cfi_undefined 2739
; WAVE64-NEXT: .cfi_undefined 2740
; WAVE64-NEXT: .cfi_undefined 2741
; WAVE64-NEXT: .cfi_undefined 2742
; WAVE64-NEXT: .cfi_undefined 2743

; WAVE64-NEXT: .cfi_undefined 2752
; WAVE64-NEXT: .cfi_undefined 2753
; WAVE64-NEXT: .cfi_undefined 2754
; WAVE64-NEXT: .cfi_undefined 2755
; WAVE64-NEXT: .cfi_undefined 2756
; WAVE64-NEXT: .cfi_undefined 2757
; WAVE64-NEXT: .cfi_undefined 2758
; WAVE64-NEXT: .cfi_undefined 2759

; WAVE64-NEXT: .cfi_undefined 2768
; WAVE64-NEXT: .cfi_undefined 2769
; WAVE64-NEXT: .cfi_undefined 2770
; WAVE64-NEXT: .cfi_undefined 2771
; WAVE64-NEXT: .cfi_undefined 2772
; WAVE64-NEXT: .cfi_undefined 2773
; WAVE64-NEXT: .cfi_undefined 2774
; WAVE64-NEXT: .cfi_undefined 2775

; WAVE64-NEXT: .cfi_undefined 2784
; WAVE64-NEXT: .cfi_undefined 2785
; WAVE64-NEXT: .cfi_undefined 2786
; WAVE64-NEXT: .cfi_undefined 2787
; WAVE64-NEXT: .cfi_undefined 2788
; WAVE64-NEXT: .cfi_undefined 2789
; WAVE64-NEXT: .cfi_undefined 2790
; WAVE64-NEXT: .cfi_undefined 2791

; WAVE64-NEXT: .cfi_undefined 2800
; WAVE64-NEXT: .cfi_undefined 2801
; WAVE64-NEXT: .cfi_undefined 2802
; WAVE64-NEXT: .cfi_undefined 2803
; WAVE64-NEXT: .cfi_undefined 2804
; WAVE64-NEXT: .cfi_undefined 2805
; WAVE64-NEXT: .cfi_undefined 2806
; WAVE64-NEXT: .cfi_undefined 2807

; AGPR0_wave64 = 3072
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3072
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3073
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3074
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3075
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3076
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3077
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3078
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3079
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3080
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3081
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3082
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3083
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3084
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3085
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3086
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3087
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3088
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3089
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3090
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3091
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3092
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3093
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3094
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3095
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3096
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3097
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3098
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3099
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3100
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3101
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3102
; GFX90A-V2A-DIS-NEXT: .cfi_undefined 3103

; GFX90A-V2A-EN-NEXT: .cfi_undefined 3072
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3073
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3074
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3075
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3076
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3077
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3078
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3079
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3080
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3081
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3082
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3083
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3084
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3085
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3086
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3087
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3088
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3089
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3090
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3091
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3092
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3093
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3094
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3095
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3096
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3097
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3098
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3099
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3100
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3101
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3102
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3103

; VGPR0_wave32 = 1536
; WAVE32-NEXT: .cfi_undefined 1536
; WAVE32-NEXT: .cfi_undefined 1537
; WAVE32-NEXT: .cfi_undefined 1538
; WAVE32-NEXT: .cfi_undefined 1539
; WAVE32-NEXT: .cfi_undefined 1540
; WAVE32-NEXT: .cfi_undefined 1541
; WAVE32-NEXT: .cfi_undefined 1542
; WAVE32-NEXT: .cfi_undefined 1543
; WAVE32-NEXT: .cfi_undefined 1544
; WAVE32-NEXT: .cfi_undefined 1545
; WAVE32-NEXT: .cfi_undefined 1546
; WAVE32-NEXT: .cfi_undefined 1547
; WAVE32-NEXT: .cfi_undefined 1548
; WAVE32-NEXT: .cfi_undefined 1549
; WAVE32-NEXT: .cfi_undefined 1550
; WAVE32-NEXT: .cfi_undefined 1551
; WAVE32-NEXT: .cfi_undefined 1552
; WAVE32-NEXT: .cfi_undefined 1553
; WAVE32-NEXT: .cfi_undefined 1554
; WAVE32-NEXT: .cfi_undefined 1555
; WAVE32-NEXT: .cfi_undefined 1556
; WAVE32-NEXT: .cfi_undefined 1557
; WAVE32-NEXT: .cfi_undefined 1558
; WAVE32-NEXT: .cfi_undefined 1559
; WAVE32-NEXT: .cfi_undefined 1560
; WAVE32-NEXT: .cfi_undefined 1561
; WAVE32-NEXT: .cfi_undefined 1562
; WAVE32-NEXT: .cfi_undefined 1563
; WAVE32-NEXT: .cfi_undefined 1564
; WAVE32-NEXT: .cfi_undefined 1565
; WAVE32-NEXT: .cfi_undefined 1566
; WAVE32-NEXT: .cfi_undefined 1567
; WAVE32-NEXT: .cfi_undefined 1568
; WAVE32-NEXT: .cfi_undefined 1569
; WAVE32-NEXT: .cfi_undefined 1570
; WAVE32-NEXT: .cfi_undefined 1571
; WAVE32-NEXT: .cfi_undefined 1572
; WAVE32-NEXT: .cfi_undefined 1573
; WAVE32-NEXT: .cfi_undefined 1574
; WAVE32-NEXT: .cfi_undefined 1575

; VPGR48_wave64 = 1584
; WAVE32-NEXT: .cfi_undefined 1584
; WAVE32-NEXT: .cfi_undefined 1585
; WAVE32-NEXT: .cfi_undefined 1586
; WAVE32-NEXT: .cfi_undefined 1587
; WAVE32-NEXT: .cfi_undefined 1588
; WAVE32-NEXT: .cfi_undefined 1589
; WAVE32-NEXT: .cfi_undefined 1590
; WAVE32-NEXT: .cfi_undefined 1591

; WAVE32-NEXT: .cfi_undefined 1600
; WAVE32-NEXT: .cfi_undefined 1601
; WAVE32-NEXT: .cfi_undefined 1602
; WAVE32-NEXT: .cfi_undefined 1603
; WAVE32-NEXT: .cfi_undefined 1604
; WAVE32-NEXT: .cfi_undefined 1605
; WAVE32-NEXT: .cfi_undefined 1606
; WAVE32-NEXT: .cfi_undefined 1607

; WAVE32-NEXT: .cfi_undefined 1616
; WAVE32-NEXT: .cfi_undefined 1617
; WAVE32-NEXT: .cfi_undefined 1618
; WAVE32-NEXT: .cfi_undefined 1619
; WAVE32-NEXT: .cfi_undefined 1620
; WAVE32-NEXT: .cfi_undefined 1621
; WAVE32-NEXT: .cfi_undefined 1622
; WAVE32-NEXT: .cfi_undefined 1623

; WAVE32-NEXT: .cfi_undefined 1632
; WAVE32-NEXT: .cfi_undefined 1633
; WAVE32-NEXT: .cfi_undefined 1634
; WAVE32-NEXT: .cfi_undefined 1635
; WAVE32-NEXT: .cfi_undefined 1636
; WAVE32-NEXT: .cfi_undefined 1637
; WAVE32-NEXT: .cfi_undefined 1638
; WAVE32-NEXT: .cfi_undefined 1639

; WAVE32-NEXT: .cfi_undefined 1648
; WAVE32-NEXT: .cfi_undefined 1649
; WAVE32-NEXT: .cfi_undefined 1650
; WAVE32-NEXT: .cfi_undefined 1651
; WAVE32-NEXT: .cfi_undefined 1652
; WAVE32-NEXT: .cfi_undefined 1653
; WAVE32-NEXT: .cfi_undefined 1654
; WAVE32-NEXT: .cfi_undefined 1655

; WAVE32-NEXT: .cfi_undefined 1664
; WAVE32-NEXT: .cfi_undefined 1665
; WAVE32-NEXT: .cfi_undefined 1666
; WAVE32-NEXT: .cfi_undefined 1667
; WAVE32-NEXT: .cfi_undefined 1668
; WAVE32-NEXT: .cfi_undefined 1669
; WAVE32-NEXT: .cfi_undefined 1670
; WAVE32-NEXT: .cfi_undefined 1671

; WAVE32-NEXT: .cfi_undefined 1680
; WAVE32-NEXT: .cfi_undefined 1681
; WAVE32-NEXT: .cfi_undefined 1682
; WAVE32-NEXT: .cfi_undefined 1683
; WAVE32-NEXT: .cfi_undefined 1684
; WAVE32-NEXT: .cfi_undefined 1685
; WAVE32-NEXT: .cfi_undefined 1686
; WAVE32-NEXT: .cfi_undefined 1687

; WAVE32-NEXT: .cfi_undefined 1696
; WAVE32-NEXT: .cfi_undefined 1697
; WAVE32-NEXT: .cfi_undefined 1698
; WAVE32-NEXT: .cfi_undefined 1699
; WAVE32-NEXT: .cfi_undefined 1700
; WAVE32-NEXT: .cfi_undefined 1701
; WAVE32-NEXT: .cfi_undefined 1702
; WAVE32-NEXT: .cfi_undefined 1703

; WAVE32-NEXT: .cfi_undefined 1712
; WAVE32-NEXT: .cfi_undefined 1713
; WAVE32-NEXT: .cfi_undefined 1714
; WAVE32-NEXT: .cfi_undefined 1715
; WAVE32-NEXT: .cfi_undefined 1716
; WAVE32-NEXT: .cfi_undefined 1717
; WAVE32-NEXT: .cfi_undefined 1718
; WAVE32-NEXT: .cfi_undefined 1719

; WAVE32-NEXT: .cfi_undefined 1728
; WAVE32-NEXT: .cfi_undefined 1729
; WAVE32-NEXT: .cfi_undefined 1730
; WAVE32-NEXT: .cfi_undefined 1731
; WAVE32-NEXT: .cfi_undefined 1732
; WAVE32-NEXT: .cfi_undefined 1733
; WAVE32-NEXT: .cfi_undefined 1734
; WAVE32-NEXT: .cfi_undefined 1735

; WAVE32-NEXT: .cfi_undefined 1744
; WAVE32-NEXT: .cfi_undefined 1745
; WAVE32-NEXT: .cfi_undefined 1746
; WAVE32-NEXT: .cfi_undefined 1747
; WAVE32-NEXT: .cfi_undefined 1748
; WAVE32-NEXT: .cfi_undefined 1749
; WAVE32-NEXT: .cfi_undefined 1750
; WAVE32-NEXT: .cfi_undefined 1751

; WAVE32-NEXT: .cfi_undefined 1760
; WAVE32-NEXT: .cfi_undefined 1761
; WAVE32-NEXT: .cfi_undefined 1762
; WAVE32-NEXT: .cfi_undefined 1763
; WAVE32-NEXT: .cfi_undefined 1764
; WAVE32-NEXT: .cfi_undefined 1765
; WAVE32-NEXT: .cfi_undefined 1766
; WAVE32-NEXT: .cfi_undefined 1767

; WAVE32-NEXT: .cfi_undefined 1776
; WAVE32-NEXT: .cfi_undefined 1777
; WAVE32-NEXT: .cfi_undefined 1778
; WAVE32-NEXT: .cfi_undefined 1779
; WAVE32-NEXT: .cfi_undefined 1780
; WAVE32-NEXT: .cfi_undefined 1781
; WAVE32-NEXT: .cfi_undefined 1782
; WAVE32-NEXT: .cfi_undefined 1783


; SGPR0 = 32
; CHECK-NEXT: .cfi_undefined 32
; CHECK-NEXT: .cfi_undefined 33
; CHECK-NEXT: .cfi_undefined 34
; CHECK-NEXT: .cfi_undefined 35
; CHECK-NEXT: .cfi_undefined 36
; CHECK-NEXT: .cfi_undefined 37
; CHECK-NEXT: .cfi_undefined 38
; CHECK-NEXT: .cfi_undefined 39
; CHECK-NEXT: .cfi_undefined 40
; CHECK-NEXT: .cfi_undefined 41
; CHECK-NEXT: .cfi_undefined 42
; CHECK-NEXT: .cfi_undefined 43
; CHECK-NEXT: .cfi_undefined 44
; CHECK-NEXT: .cfi_undefined 45
; CHECK-NEXT: .cfi_undefined 46
; CHECK-NEXT: .cfi_undefined 47
; CHECK-NEXT: .cfi_undefined 48
; CHECK-NEXT: .cfi_undefined 49
; CHECK-NEXT: .cfi_undefined 50
; CHECK-NEXT: .cfi_undefined 51
; CHECK-NEXT: .cfi_undefined 52
; CHECK-NEXT: .cfi_undefined 53
; CHECK-NEXT: .cfi_undefined 54
; CHECK-NEXT: .cfi_undefined 55
; CHECK-NEXT: .cfi_undefined 56
; CHECK-NEXT: .cfi_undefined 57
; CHECK-NEXT: .cfi_undefined 58
; CHECK-NEXT: .cfi_undefined 59
; CHECK-NEXT: .cfi_undefined 60
; CHECK-NEXT: .cfi_undefined 61

; CHECK-NOT: .cfi_{{.*}}

; CHECK: s_mov_b32 [[FP_SCRATCH_COPY:s[0-9]+]], s33
; CHECK: s_mov_b32 s33, s32
; WAVE64: s_or_saveexec_b64 [[EXEC_MASK:s\[[0-9]+:[0-9]+\]]], -1
; WAVE32: s_or_saveexec_b32 [[EXEC_MASK:s[0-9]+]], -1
; CHECK-NEXT: buffer_store_dword v40, off, s[0:3], s33 ; 4-byte Folded Spill
; VGPR40_wave64 = 2600
; WAVE64-NEXT: .cfi_offset 2600, 0
; VGPR40_wave32 = 1576
; WAVE32-NEXT: .cfi_offset 1576, 0
; WAVE64: s_mov_b64 exec, [[EXEC_MASK]]
; WAVE32: s_mov_b32 exec_lo, [[EXEC_MASK]]

; CHECK-NOT: .cfi_{{.*}}

; CHECK: v_writelane_b32 v40, [[FP_SCRATCH_COPY]], 2
; WAVE64-NEXT: .cfi_llvm_vector_registers 65, 2600, 2, 32
; WAVE32-NEXT: .cfi_llvm_vector_registers 65, 1576, 2, 32

; CHECK-NOT: .cfi_{{.*}}

; SGPR33 = 65
; CHECK-NEXT: .cfi_def_cfa_register 65

; CHECK-NOT: .cfi_{{.*}}

; CHECK: s_addk_i32 s32,
; CHECK: v_readlane_b32 [[FP_SCRATCH_COPY:s[0-9]+]], v40, 2
; SGPR32 = 64
; CHECK: .cfi_def_cfa_register 64
; CHECK-NEXT: s_mov_b32 s33, [[FP_SCRATCH_COPY]]

; CHECK-NOT: .cfi_{{.*}}

; CHECK: .cfi_endproc
define hidden void @func_call_clobber() #0 {
entry:
  call void @ex() #0
  ret void
}

; CHECK-LABEL: func_spill_vgpr_to_vmem:
; CHECK: .cfi_startproc

; CHECK-NOT: .cfi_{{.*}}

; CHECK: %bb.0:
; SGPR32 = 64
; CHECK-NEXT: .cfi_llvm_def_aspace_cfa 64, 0, 6
; CHECK-NEXT: .cfi_llvm_register_pair 16, 62, 32, 63, 32
; GFX90A-V2A-EN-NEXT: .cfi_undefined 2560
; GFX90A-V2A-EN-NEXT: .cfi_undefined 2561
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3072
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3073

; CHECK-NOT: .cfi_{{.*}}

; WAVE32: buffer_store_dword v40, off, s[0:3], s32 offset:4 ; 4-byte Folded Spill
; GFX900: buffer_store_dword v40, off, s[0:3], s32 offset:4 ; 4-byte Folded Spill
; GFX90A-V2A-DIS: buffer_store_dword v40, off, s[0:3], s32 offset:12 ; 4-byte Folded Spill
; GFX90A-V2A-EN: v_accvgpr_write_b32 a[[#TMP_AGPR1:]], v[[#VGPR1:]]

; GFX900-NEXT: .cfi_llvm_vector_offset 2600, 32, 17, 64, 256
; GFX90A-V2A-DIS-NEXT: .cfi_llvm_vector_offset 2600, 32, 17, 64, 768
; GFX90A-V2A-EN-NEXT: .cfi_register [[#VGPR1+2560]], [[#TMP_AGPR1+3072]]

; WAVE32-NEXT: .cfi_llvm_vector_offset 1576, 32, 1, 32, 128

; CHECK-NOT: .cfi_{{.*}}

; WAVE32: buffer_store_dword v41, off, s[0:3], s32 ; 4-byte Folded Spill
; GFX900: buffer_store_dword v41, off, s[0:3], s32 ; 4-byte Folded Spill
; GFX90A-V2A-DIS: buffer_store_dword v41, off, s[0:3], s32 offset:8 ; 4-byte Folded Spill
; GFX90A-V2A-EN: v_accvgpr_write_b32 a[[#TMP_AGPR2:]], v[[#VGPR2:]]

; GFX900-NEXT: .cfi_llvm_vector_offset 2601, 32, 17, 64, 0
; GFX90A-V2A-DIS-NEXT: .cfi_llvm_vector_offset 2601, 32, 17, 64, 512
; GFX90A-V2A-EN-NEXT: .cfi_register [[#VGPR2+2560]], [[#TMP_AGPR2+3072]]

; WAVE32: .cfi_llvm_vector_offset 1577, 32, 1, 32, 0

; GFX90A-V2A-DIS: buffer_store_dword a32, off, s[0:3], s32 offset:4 ; 4-byte Folded Spill
; GFX90A-V2A-EN: v_accvgpr_read_b32 v[[#TMP_VGPR1:]], a[[#AGPR1:]]

; GFX90A-V2A-DIS-NEXT: .cfi_llvm_vector_offset 3104, 32, 17, 64, 256
; GFX90A-V2A-EN-NEXT: .cfi_register [[#AGPR1+3072]], [[#TMP_VGPR1+2560]]

; CHECK-NOT: .cfi_{{.*}}

; GFX90A-V2A-DIS: buffer_store_dword a33, off, s[0:3], s32 ; 4-byte Folded Spill
; GFX90A-V2A-EN: v_accvgpr_read_b32 v[[#TMP_VGPR2:]], a[[#AGPR2:]]

; GFX90A-V2A-DIS-NEXT: .cfi_llvm_vector_offset 3105, 32, 17, 64, 0
; GFX90A-V2A-EN-NEXT: .cfi_register [[#AGPR2+3072]], [[#TMP_VGPR2+2560]]

; CHECK-NOT: .cfi_{{.*}}

; CHECK: .cfi_endproc
define hidden void @func_spill_vgpr_to_vmem() #0 {
entry:
  call void asm sideeffect "; clobber", "~{v40}"() #0
  call void asm sideeffect "; clobber", "~{v41}"() #0
  call void asm sideeffect "; clobber", "~{a32}"() #0
  call void asm sideeffect "; clobber", "~{a33}"() #0
  ret void
}

; CHECK-LABEL: func_spill_vgpr_to_agpr:
; CHECK: .cfi_startproc

; CHECK-NOT: .cfi_{{.*}}

; CHECK: %bb.0:
; CHECK-NEXT: .cfi_llvm_def_aspace_cfa 64, 0, 6
; CHECK-NEXT: .cfi_llvm_register_pair 16, 62, 32, 63, 32
; GFX90A-V2A-EN-NEXT: .cfi_undefined 2560
; GFX90A-V2A-EN-NEXT: .cfi_undefined 2561
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3072
; GFX90A-V2A-EN-NEXT: .cfi_undefined 3073

; CHECK-NOT: .cfi_{{.*}}

; CHECK-NEXT: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX90A-V2A-EN-NEXT: v_accvgpr_write_b32 a[[#TMP_AGPR1:]], v[[#VGPR1:]]
; GFX90A-V2A-EN-NEXT: .cfi_register [[#VGPR1+2560]], [[#TMP_AGPR1+3072]]
; GFX90A-V2A-EN-NEXT: v_accvgpr_write_b32 a[[#TMP_AGPR2]], v[[#VGPR2]]
; GFX90A-V2A-EN-NEXT: .cfi_register [[#VGPR2+2560]], [[#TMP_AGPR2+3072]]
; GFX90A-V2A-EN-NEXT: v_accvgpr_read_b32 v[[#TMP_VGPR1:]], a[[#AGPR1:]]
; GFX90A-V2A-EN-NEXT: .cfi_register [[#AGPR1+3072]], [[#TMP_VGPR1+2560]]
; GFX90A-V2A-EN-NEXT: v_accvgpr_read_b32 v[[#TMP_VGPR2:]], a[[#AGPR2:]]
; GFX90A-V2A-EN-NEXT: .cfi_register [[#AGPR2+3072]], [[#TMP_VGPR2+2560]]
; GFX90A-V2A-EN: v_accvgpr_write_b32 a33, v1
; GFX90A-V2A-EN-NEXT: v_accvgpr_write_b32 a32, v0
; GFX90A-V2A-EN-NEXT: v_accvgpr_read_b32 v41, a1
; GFX90A-V2A-EN-NEXT: v_accvgpr_read_b32 v40, a0

; CHECK:	s_setpc_b64 s[30:31]

; CHECK-NOT:	.cfi_{{.*}}
; CHECK:	.cfi_endproc

define hidden void @func_spill_vgpr_to_agpr() #2 {
  call void asm sideeffect "; clobber", "~{v40}"()
  call void asm sideeffect "; clobber", "~{v41}"()
  call void asm sideeffect "; clobber", "~{a32}"()
  call void asm sideeffect "; clobber", "~{a33}"()
  ret void
}


; NOTE: Number of VGPRs available to kernel, and in turn number of corresponding CFIs generated,
; is dependent on waves/WG size. Since the intent here is to check whether we generate the correct
; CFIs, doing it for any one set of details is sufficient which also makes the test insensitive to
; changes in those details.
attributes #0 = { nounwind "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="128,128" }
attributes #1 = { nounwind "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="128,128" "frame-pointer"="all" }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "filename", directory: "directory")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}

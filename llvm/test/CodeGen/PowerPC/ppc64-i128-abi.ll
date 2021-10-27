; RUN: llc -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr8 < %s | FileCheck %s -check-prefix=CHECK-LE \
; RUN:   --implicit-check-not xxswapd

; RUN: llc -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -mcpu=pwr8 < %s | FileCheck %s -check-prefix=CHECK-BE

; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff \
; RUN:   -mcpu=pwr8 < %s | FileCheck %s -check-prefix=CHECK-BE

; RUN: llc -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -mcpu=pwr8 -mattr=-vsx < %s | FileCheck %s -check-prefix=CHECK-NOVSX

; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff \
; RUN:   -mcpu=pwr8 -mattr=-vsx < %s | FileCheck %s -check-prefix=CHECK-NOVSX

; RUN: llc -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr8 -mattr=-vsx < %s | FileCheck %s -check-prefix=CHECK-NOVSX \
; RUN:   --implicit-check-not xxswapd

; RUN: llc -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -mcpu=pwr8 -mattr=-vsx < %s | FileCheck %s -check-prefix=CHECK-BE-NOVSX

; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff \
; RUN:   -mcpu=pwr8 -mattr=-vsx < %s | FileCheck %s -check-prefix=CHECK-BE-NOVSX

; RUN: llc -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr8 -mattr=-vsx < %s | \
; RUN:   FileCheck %s -check-prefix=CHECK-LE-NOVSX --implicit-check-not xxswapd

; RUN: llc -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr9 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | \
; RUN:   FileCheck %s -check-prefix=CHECK-P9 --implicit-check-not xxswapd

; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff \
; RUN:   -mcpu=pwr9 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | FileCheck %s -check-prefix=CHECK-P9

; RUN: llc -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr9 -mattr=-vsx < %s | FileCheck %s -check-prefix=CHECK-NOVSX \
; RUN:   --implicit-check-not xxswapd

; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff \
; RUN:   -mcpu=pwr9 -mattr=-vsx < %s | FileCheck %s -check-prefix=CHECK-NOVSX

; RUN: llc -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr9 -mattr=-power9-vector -mattr=-direct-move < %s | \
; RUN:   FileCheck %s -check-prefix=CHECK-LE --implicit-check-not xxswapd

@x = common global <1 x i128> zeroinitializer, align 16
@y = common global <1 x i128> zeroinitializer, align 16
@a = common global i128 zeroinitializer, align 16
@b = common global i128 zeroinitializer, align 16

; VSX:
;   %a is passed in register 34
;   The value of 1 is stored in the TOC.
;   On LE, ensure the value of 1 is swapped before being used (using xxswapd).
; VMX (no VSX): 
;   %a is passed in register 2
;   The value of 1 is stored in the TOC.
;   No swaps are necessary when using P8 Vector instructions on LE
define <1 x i128> @v1i128_increment_by_one(<1 x i128> %a) nounwind {
       %tmp = add <1 x i128> %a, <i128 1>
       ret <1 x i128> %tmp  

; FIXME: Seems a 128-bit literal is materialized by loading from the TOC. There
;        should be a better way of doing this.

; CHECK-LE-LABEL: @v1i128_increment_by_one
; CHECK-LE: lxvd2x [[VAL:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK-LE: xxswapd 35, [[VAL]]
; CHECK-LE: vadduqm 2, 2, 3
; CHECK-LE: blr

; CHECK-P9-LABEL: @v1i128_increment_by_one
; The below FIXME is due to the lowering for BUILD_VECTOR that will be fixed
; in a subsequent patch.
; FIXME: li [[R1:r[0-9]+]], 1
; FIXME: li [[R2:r[0-9]+]], 0
; FIXME: mtvsrdd [[V1:v[0-9]+]], [[R2]], [[R1]]
; CHECK-P9: lxv [[V1:v[0-9]+]]
; CHECK-P9: vadduqm v2, v2, [[V1]]
; CHECK-P9: blr

; CHECK-BE-LABEL: @v1i128_increment_by_one
; CHECK-BE: lxvd2x 35, {{[0-9]+}}, {{[0-9]+}}
; CHECK-BE-NOT: xxswapd 
; CHECK-BE: vadduqm 2, 2, 3 
; CHECK-BE-NOT: xxswapd 34, {{[0-9]+}}
; CHECK-BE: blr

; CHECK-NOVSX-LABEL: @v1i128_increment_by_one
; CHECK-NOVSX-NOT: xxswapd {{[0-9]+}}, {{[0-9]+}}
; CHECK-NOVSX-NOT: stxvd2x {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK-NOVSX: lvx [[VAL:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK-NOVSX-NOT: lxvd2x {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK-NOVSX-NOT: xxswapd {{[0-9]+}}, {{[0-9]+}}
; CHECK-NOVSX: vadduqm 2, 2, [[VAL]]
; CHECK-NOVSX: blr
}

; VSX:
;   %a is passed in register 34
;   %b is passed in register 35
;   No swaps are necessary when using P8 Vector instructions on LE
; VMX (no VSX):
;   %a is passewd in register 2
;   %b is passed in register 3
;   On LE, do not need to swap contents of 2 and 3 because the lvx/stvx 
;   instructions no not swap elements
define <1 x i128> @v1i128_increment_by_val(<1 x i128> %a, <1 x i128> %b) nounwind {
       %tmp = add <1 x i128> %a, %b
       ret <1 x i128> %tmp

; CHECK-LE-LABEL: @v1i128_increment_by_val
; CHECK-LE-NOT: xxswapd
; CHECK-LE: adduqm 2, 2, 3
; CHECK-LE: blr

; CHECK-BE-LABEL: @v1i128_increment_by_val
; CHECK-BE-NOT: xxswapd {{[0-9]+}}, 34
; CHECK-BE-NOT: xxswapd {{[0-9]+}}, 35
; CHECK-BE-NOT: xxswapd 34, {{[0-9]+}}
; CHECK-BE: adduqm 2, 2, 3
; CHECK-BE: blr

; CHECK-NOVSX-LABEL: @v1i128_increment_by_val
; CHECK-NOVSX-NOT: xxswapd 34, {{[0-9]+}}
; CHECK-NOVSX: adduqm 2, 2, 3
; CHECK-NOVSX: blr
}

; Little Endian (VSX and VMX):
;   Lower 64-bits of %a are passed in register 3
;   Upper 64-bits of %a are passed in register 4
;   Increment lower 64-bits using addic (immediate value of 1)
;   Increment upper 64-bits using add zero extended
;   Results are placed in registers 3 and 4
; Big Endian (VSX and VMX)
;   Lower 64-bits of %a are passed in register 4
;   Upper 64-bits of %a are passed in register 3
;   Increment lower 64-bits using addic (immediate value of 1)
;   Increment upper 64-bits using add zero extended
;   Results are placed in registers 3 and 4
define i128 @i128_increment_by_one(i128 %a) nounwind {
       %tmp =  add i128 %a,  1
       ret i128 %tmp
; CHECK-LE-LABEL: @i128_increment_by_one
; CHECK-LE: addic 3, 3, 1
; CHECK-LE-NEXT: addze 4, 4
; CHECK-LE: blr

; CHECK-BE-LABEL: @i128_increment_by_one
; CHECK-BE: addic 4, 4, 1
; CHECK-BE-NEXT: addze 3, 3
; CHECK-BE: blr

; CHECK-LE-NOVSX-LABEL: @i128_increment_by_one
; CHECK-LE-NOVSX: addic 3, 3, 1
; CHECK-LE-NOVSX-NEXT: addze 4, 4
; CHECK-LE-NOVSX: blr

; CHECK-BE-NOVSX-LABEL: @i128_increment_by_one
; CHECK-BE-NOVSX: addic 4, 4, 1
; CHECK-BE-NOVSX-NEXT: addze 3, 3
; CHECK-BE-NOVSX: blr
}

; Little Endian (VSX and VMX):
;   Lower 64-bits of %a are passed in register 3
;   Upper 64-bits of %a are passed in register 4
;   Lower 64-bits of %b are passed in register 5
;   Upper 64-bits of %b are passed in register 6
;   Add the lower 64-bits using addc on registers 3 and 5
;   Add the upper 64-bits using adde on registers 4 and 6
;   Registers 3 and 4 should hold the result
; Big Endian (VSX and VMX):
;   Upper 64-bits of %a are passed in register 3
;   Lower 64-bits of %a are passed in register 4
;   Upper 64-bits of %b are passed in register 5
;   Lower 64-bits of %b are passed in register 6
;   Add the lower 64-bits using addc on registers 4 and 6
;   Add the upper 64-bits using adde on registers 3 and 5
;   Registers 3 and 4 should hold the result
define i128 @i128_increment_by_val(i128 %a, i128 %b) nounwind {
       %tmp =  add i128 %a, %b
       ret i128 %tmp
; CHECK-LE-LABEL: @i128_increment_by_val
; CHECK-LE: addc 3, 3, 5
; CHECK-LE-NEXT: adde 4, 4, 6
; CHECK-LE: blr

; CHECK-BE-LABEL: @i128_increment_by_val
; CHECK-BE: addc 4, 4, 6
; CHECK-BE-NEXT: adde 3, 3, 5
; CHECK-BE: blr

; CHECK-LE-NOVSX-LABEL: @i128_increment_by_val
; CHECK-LE-NOVSX: addc 3, 3, 5
; CHECK-LE-NOVSX-NEXT: adde 4, 4, 6
; CHECK-LE-NOVSX: blr

; CHECK-BE-NOVSX-LABEL: @i128_increment_by_val
; CHECK-BE-NOVSX: addc 4, 4, 6
; CHECK-BE-NOVSX-NEXT: adde 3, 3, 5
; CHECK-BE-NOVSX: blr
}


; Callsites for the routines defined above. 
; Ensure the parameters are loaded in the same order that is expected by the 
; callee. See comments for individual functions above for details on registers
; used for parameters.
define <1 x i128> @call_v1i128_increment_by_one() nounwind {
       %tmp = load <1 x i128>, <1 x i128>* @x, align 16
       %ret = call <1 x i128> @v1i128_increment_by_one(<1 x i128> %tmp)
       ret <1 x i128> %ret

; CHECK-LE-LABEL: @call_v1i128_increment_by_one
; CHECK-LE: lvx 2, {{[0-9]+}}, {{[0-9]+}}
; CHECK-LE: bl v1i128_increment_by_one
; CHECK-LE: blr

; CHECK-P9-LABEL: @call_v1i128_increment_by_one
; CHECK-P9: lxv
; CHECK-P9: bl {{.?}}v1i128_increment_by_one
; CHECK-P9: blr

; CHECK-BE-LABEL: @call_v1i128_increment_by_one
; CHECK-BE: lxvw4x 34, {{[0-9]+}}, {{[0-9]+}}
; CHECK-BE-NOT: xxswapd 34, {{[0-9]+}}
; CHECK-BE: bl {{.?}}v1i128_increment_by_one
; CHECK-BE: blr

; CHECK-NOVSX-LABEL: @call_v1i128_increment_by_one
; CHECK-NOVSX: lvx 2, {{[0-9]+}}, {{[0-9]+}}
; CHECK-NOVSX-NOT: xxswapd {{[0-9]+}}, {{[0-9]+}}
; CHECK-NOVSX: bl {{.?}}v1i128_increment_by_one
; CHECK-NOVSX: blr
}

define <1 x i128> @call_v1i128_increment_by_val() nounwind {
       %tmp = load <1 x i128>, <1 x i128>* @x, align 16
       %tmp2 = load <1 x i128>, <1 x i128>* @y, align 16
       %ret = call <1 x i128> @v1i128_increment_by_val(<1 x i128> %tmp, <1 x i128> %tmp2)
       ret <1 x i128> %ret

; CHECK-LE-LABEL: @call_v1i128_increment_by_val
; CHECK-LE: lvx 2, {{[0-9]+}}, {{[0-9]+}}
; CHECK-LE: lvx 3, {{[0-9]+}}, {{[0-9]+}}
; CHECK-LE: bl v1i128_increment_by_val
; CHECK-LE: blr

; CHECK-P9-LABEL: @call_v1i128_increment_by_val
; CHECK-P9-DAG: lxv v2
; CHECK-P9-DAG: lxv v3
; CHECK-P9: bl {{.?}}v1i128_increment_by_val
; CHECK-P9: blr

; CHECK-BE-LABEL: @call_v1i128_increment_by_val


; CHECK-BE-DAG: lxvw4x 35, {{[0-9]+}}, {{[0-9]+}}
; CHECK-BE-NOT: xxswapd 34, {{[0-9]+}}
; CHECK-BE-NOT: xxswapd 35, {{[0-9]+}}
; CHECK-BE: bl {{.?}}v1i128_increment_by_val
; CHECK-BE: blr

; CHECK-NOVSX-LABEL: @call_v1i128_increment_by_val
; CHECK-NOVSX-DAG: lvx 2, {{[0-9]+}}, {{[0-9]+}}
; CHECK-NOVSX-DAG: lvx 3, {{[0-9]+}}, {{[0-9]+}}
; CHECK-NOVSX-NOT: xxswapd 34, {{[0-9]+}}
; CHECK-NOVSX-NOT: xxswapd 35, {{[0-9]+}}
; CHECK-NOVSX: bl {{.?}}v1i128_increment_by_val
; CHECK-NOVSX: blr

}

define i128 @call_i128_increment_by_one() nounwind {
       %tmp = load i128, i128* @a, align 16
       %ret = call i128 @i128_increment_by_one(i128 %tmp)
       ret i128 %ret
;       %ret4 = call i128 @i128_increment_by_val(i128 %tmp2, i128 %tmp2)
; CHECK-LE-LABEL: @call_i128_increment_by_one
; CHECK-LE-DAG: ld 3, 0([[BASEREG:[0-9]+]])
; CHECK-LE-DAG: ld 4, 8([[BASEREG]])
; CHECK-LE: bl i128_increment_by_one
; CHECK-LE: blr

; CHECK-BE-LABEL: @call_i128_increment_by_one
; CHECK-BE-DAG: ld 3, 0([[BASEREG:[0-9]+]])
; CHECK-BE-DAG: ld 4, 8([[BASEREG]])
; CHECK-BE: bl {{.?}}i128_increment_by_one
; CHECK-BE: blr

; CHECK-NOVSX-LABEL: @call_i128_increment_by_one
; CHECK-NOVSX-DAG: ld 3, 0([[BASEREG:[0-9]+]])
; CHECK-NOVSX-DAG: ld 4, 8([[BASEREG]])
; CHECK-NOVSX: bl {{.?}}i128_increment_by_one
; CHECK-NOVSX: blr
}

define i128 @call_i128_increment_by_val() nounwind {
       %tmp = load i128, i128* @a, align 16
       %tmp2 = load i128, i128* @b, align 16
       %ret = call i128 @i128_increment_by_val(i128 %tmp, i128 %tmp2)
       ret i128 %ret
; CHECK-LE-LABEL: @call_i128_increment_by_val
; CHECK-LE-DAG: ld 3, 0([[P1BASEREG:[0-9]+]])
; CHECK-LE-DAG: ld 4, 8([[P1BASEREG]])
; CHECK-LE-DAG: ld 5, 0([[P2BASEREG:[0-9]+]])
; CHECK-LE-DAG: ld 6, 8([[P2BASEREG]])
; CHECK-LE: bl i128_increment_by_val
; CHECK-LE: blr

; CHECK-BE-LABEL: @call_i128_increment_by_val
; CHECK-BE-DAG: ld 3, 0([[P1BASEREG:[0-9]+]])
; CHECK-BE-DAG: ld 4, 8([[P1BASEREG]])
; CHECK-BE-DAG: ld 5, 0([[P2BASEREG:[0-9]+]])
; CHECK-BE-DAG: ld 6, 8([[P2BASEREG]])
; CHECK-BE: bl {{.?}}i128_increment_by_val
; CHECK-BE: blr

; CHECK-NOVSX-LABEL: @call_i128_increment_by_val
; CHECK-NOVSX-DAG: ld 3, 0([[P1BASEREG:[0-9]+]])
; CHECK-NOVSX-DAG: ld 4, 8([[P1BASEREG]])
; CHECK-NOVSX-DAG: ld 5, 0([[P2BASEREG:[0-9]+]])
; CHECK-NOVSX-DAG: ld 6, 8([[P2BASEREG]])
; CHECK-NOVSX: bl {{.?}}i128_increment_by_val
; CHECK-NOVSX: blr
}

define i128 @callee_i128_split(i32 %i, i128 %i1280, i32 %i4, i32 %i5,
                               i32 %i6, i32 %i7, i128 %i1281, i32 %i8, i128 %i1282){
entry:
  %tmp =  add i128 %i1280, %i1281
  %tmp1 =  add i128 %tmp, %i1282

  ret i128 %tmp1
}
; CHECK-LE-LABEL: @callee_i128_split
; CHECK-LE-DAG: ld [[TMPREG:[0-9]+]], [[OFFSET:[0-9]+]](1)
; CHECK-LE-DAG: addc [[TMPREG2:[0-9]+]], 4, 10
; CHECK-LE-DAG: adde [[TMPREG3:[0-9]+]], 5, [[TMPREG]]

; CHECK-LE-DAG: ld [[TMPREG4:[0-9]+]], [[OFFSET2:[0-9]+]](1)
; CHECK-LE-DAG: ld [[TMPREG5:[0-9]+]], [[OFFSET3:[0-9]+]](1)
; CHECK-LE-DAG: addc 3, [[TMPREG2]], [[TMPREG4]]
; CHECK-LE-DAG: adde 4, [[TMPREG3]], [[TMPREG5]]

; CHECK-BE-LABEL: @callee_i128_split
; CHECK-BE-DAG: ld [[TMPREG:[0-9]+]], [[OFFSET:[0-9]+]](1)
; CHECK-BE-DAG: addc [[TMPREG3:[0-9]+]], 5, [[TMPREG]]
; CHECK-BE-DAG: adde [[TMPREG2:[0-9]+]], 4, 10

; CHECK-BE-DAG: ld [[TMPREG4:[0-9]+]], [[OFFSET2:[0-9]+]](1)
; CHECK-BE-DAG: ld [[TMPREG5:[0-9]+]], [[OFFSET3:[0-9]+]](1)
; CHECK-BE-DAG: addc 4, [[TMPREG3]], [[TMPREG4]]
; CHECK-BE-DAG: adde 3, [[TMPREG2]], [[TMPREG5]]

define i128 @i128_split() {
entry:
  %0 = load i128, i128* @a, align 16
  %1 = load i128, i128* @b, align 16
  %call = tail call i128 @callee_i128_split(i32 1, i128 %0, i32 4, i32 5,
                                           i32 6, i32 7, i128 %1, i32 8, i128 9)
  ret i128 %call
}

; CHECK-LE-LABEL: @i128_split
; CHECK-LE-DAG: li 3, 1
; CHECK-LE-DAG: ld 4, 0([[P2BASEREG:[0-9]+]])
; CHECK-LE-DAG: ld 5, 8([[P2BASEREG]])
; CHECK-LE-DAG: li 6, 4
; CHECK-LE-DAG: li 7, 5
; CHECK-LE-DAG: li 8, 6
; CHECK-LE-DAG: li 9, 7
; CHECK-LE-DAG: ld 10, 0([[P7BASEREG:[0-9]+]])
; CHECK-LE-DAG: ld [[TMPREG:[0-9]+]], 8([[P7BASEREG]])
; CHECK-LE-DAG: std [[TMPREG]], [[OFFSET:[0-9]+]](1)
; CHECK-LE: bl callee_i128_split


; CHECK-BE-LABEL: @i128_split
; CHECK-BE-DAG: li 3, 1
; CHECK-BE-DAG: ld 4, 0([[P2BASEREG:[0-9]+]])
; CHECK-BE-DAG: ld 5, 8([[P2BASEREG]])
; CHECK-BE-DAG: li 6, 4
; CHECK-BE-DAG: li 7, 5
; CHECK-BE-DAG: li 8, 6
; CHECK-BE-DAG: li 9, 7
; CHECK-BE-DAG: ld 10, 0([[P7BASEREG:[0-9]+]])
; CHECK-BE-DAG: ld [[TMPREG:[0-9]+]], 8([[P7BASEREG]])
; CHECK-BE-DAG: std [[TMPREG]], [[OFFSET:[0-9]+]](1)
; CHECK-BE: bl {{.?}}callee_i128_split

; CHECK-NOVSX-LABEL: @i128_split
; CHECK-NOVSX-DAG: li 3, 1
; CHECK-NOVSX-DAG: ld 4, 0([[P2BASEREG:[0-9]+]])
; CHECK-NOVSX-DAG: ld 5, 8([[P2BASEREG]])
; CHECK-NOVSX-DAG: li 6, 4
; CHECK-NOVSX-DAG: li 7, 5
; CHECK-NOVSX-DAG: li 8, 6
; CHECK-NOVSX-DAG: li 9, 7
; CHECK-NOVSX-DAG: ld 10, 0([[P7BASEREG:[0-9]+]])
; CHECK-NOVSX-DAG: ld [[TMPREG:[0-9]+]], 8([[P7BASEREG]])
; CHECK-NOVSX-DAG: std [[TMPREG]], [[OFFSET:[0-9]+]](1)
; CHECK-NOVSX: bl {{.?}}callee_i128_split

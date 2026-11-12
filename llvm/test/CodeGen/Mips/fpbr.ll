; RUN: llc < %s -mtriple=mipsel-elf -mcpu=mips32   -relocation-model=pic  | FileCheck %s -check-prefixes=ALL,32-FCC
; RUN: llc < %s -mtriple=mipsel-elf -mcpu=mips32r2 -relocation-model=pic  | FileCheck %s -check-prefixes=ALL,32-FCC
; RUN: llc < %s -mtriple=mipsel-elf -mcpu=mips32r6 -relocation-model=pic  | FileCheck %s -check-prefixes=ALL,R6,32-R6
; RUN: llc < %s -mtriple=mips64el-elf -mcpu=mips64   | FileCheck %s -check-prefixes=ALL,64-FCC
; RUN: llc < %s -mtriple=mips64el-elf -mcpu=mips64r2 | FileCheck %s -check-prefixes=ALL,64-FCC
; RUN: llc < %s -mtriple=mips64el-elf -mcpu=mips64r6 | FileCheck %s -check-prefixes=ALL,R6,64-R6

define void @func0(float %f2, float %f3) nounwind {
entry:
; ALL-LABEL: func0:

; 32-FCC:        c.eq.s $f12, $f14
; 32-FCC:        bc1f   $BB0_2
; 64-FCC:        c.eq.s $f12, $f13
; 64-FCC:        bc1f   .LBB0_2

; 32-R6:        cmp.eq.s $[[FGRCC:f[0-9]+]], $f12, $f14
; 64-R6:        cmp.eq.s $[[FGRCC:f[0-9]+]], $f12, $f13
; R6-NOT:       mfc1
; R6-NOT:       not
; R6-NOT:       andi
; 32-R6:        bc1eqz   $[[FGRCC]], $BB0_2
; 64-R6:        bc1eqz   $[[FGRCC]], .LBB0_2

  %cmp = fcmp oeq float %f2, %f3
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  tail call void (...) @g0() nounwind
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void (...) @g1() nounwind
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

declare void @g0(...)

declare void @g1(...)

define void @func1(float %f2, float %f3) nounwind {
entry:
; ALL-LABEL: func1:

; 32-FCC:        c.olt.s $f12, $f14
; 32-FCC:        bc1f    $BB1_2
; 64-FCC:        c.olt.s $f12, $f13
; 64-FCC:        bc1f    .LBB1_2

; 32-R6:        cmp.ule.s $[[FGRCC:f[0-9]+]], $f14, $f12
; 64-R6:        cmp.ule.s $[[FGRCC:f[0-9]+]], $f13, $f12
; R6-NOT:       mfc1
; R6-NOT:       not
; R6-NOT:       andi
; 32-R6:        bc1nez   $[[FGRCC]], $BB1_2
; 64-R6:        bc1nez   $[[FGRCC]], .LBB1_2

  %cmp = fcmp olt float %f2, %f3
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  tail call void (...) @g0() nounwind
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void (...) @g1() nounwind
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define void @func2(float %f2, float %f3) nounwind {
entry:
; ALL-LABEL: func2:

; 32-FCC:        c.ole.s $f12, $f14
; 32-FCC:        bc1t    $BB2_2
; 64-FCC:        c.ole.s $f12, $f13
; 64-FCC:        bc1t    .LBB2_2

; 32-R6:        cmp.ult.s $[[FGRCC:f[0-9]+]], $f14, $f12
; 64-R6:        cmp.ult.s $[[FGRCC:f[0-9]+]], $f13, $f12
; R6-NOT:       mfc1
; R6-NOT:       not
; R6-NOT:       andi
; 32-R6:        bc1eqz   $[[FGRCC]], $BB2_2
; 64-R6:        bc1eqz   $[[FGRCC]], .LBB2_2

  %cmp = fcmp ugt float %f2, %f3
  br i1 %cmp, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void (...) @g0() nounwind
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void (...) @g1() nounwind
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define void @func3(double %f2, double %f3) nounwind {
entry:
; ALL-LABEL: func3:

; 32-FCC:        c.eq.d $f12, $f14
; 32-FCC:        bc1f $BB3_2
; 64-FCC:        c.eq.d $f12, $f13
; 64-FCC:        bc1f .LBB3_2

; 32-R6:        cmp.eq.d $[[FGRCC:f[0-9]+]], $f12, $f14
; 64-R6:        cmp.eq.d $[[FGRCC:f[0-9]+]], $f12, $f13
; R6-NOT:       mfc1
; R6-NOT:       not
; R6-NOT:       andi
; 32-R6:        bc1eqz   $[[FGRCC]], $BB3_2
; 32-R6-NEXT:   addu     $gp, $2, $25
; 64-R6:        bc1eqz   $[[FGRCC]], .LBB3_2

  %cmp = fcmp oeq double %f2, %f3
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  tail call void (...) @g0() nounwind
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void (...) @g1() nounwind
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define void @func4(double %f2, double %f3) nounwind {
entry:
; ALL-LABEL: func4:

; 32-FCC:        c.olt.d $f12, $f14
; 32-FCC:        bc1f $BB4_2
; 64-FCC:        c.olt.d $f12, $f13
; 64-FCC:        bc1f .LBB4_2

; 32-R6:        cmp.ule.d $[[FGRCC:f[0-9]+]], $f14, $f12
; 64-R6:        cmp.ule.d $[[FGRCC:f[0-9]+]], $f13, $f12
; R6-NOT:       mfc1
; R6-NOT:       not
; R6-NOT:       andi
; 32-R6:        bc1nez   $[[FGRCC]], $BB4_2
; 32-R6-NEXT:   addu     $gp, $2, $25
; 64-R6:        bc1nez   $[[FGRCC]], .LBB4_2

  %cmp = fcmp olt double %f2, %f3
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  tail call void (...) @g0() nounwind
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void (...) @g1() nounwind
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define void @func5(double %f2, double %f3) nounwind {
entry:
; ALL-LABEL: func5:

; 32-FCC:        c.ole.d $f12, $f14
; 32-FCC:        bc1t $BB5_2
; 64-FCC:        c.ole.d $f12, $f13
; 64-FCC:        bc1t .LBB5_2

; 32-R6:        cmp.ult.d $[[FGRCC:f[0-9]+]], $f14, $f12
; 64-R6:        cmp.ult.d $[[FGRCC:f[0-9]+]], $f13, $f12
; R6-NOT:       mfc1
; R6-NOT:       not
; R6-NOT:       andi
; 32-R6:        bc1eqz   $[[FGRCC]], $BB5_2
; 32-R6-NEXT:   addu     $gp, $2, $25
; 64-R6:        bc1eqz   $[[FGRCC]], .LBB5_2

  %cmp = fcmp ugt double %f2, %f3
  br i1 %cmp, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void (...) @g0() nounwind
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void (...) @g1() nounwind
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

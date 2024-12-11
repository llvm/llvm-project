; RUN: llc < %s -march=mipsel -mcpu=mips32   -relocation-model=pic  | FileCheck %s -check-prefixes=ALL,32-FCC
; RUN: llc < %s -march=mipsel -mcpu=mips32r2 -relocation-model=pic  | FileCheck %s -check-prefixes=ALL,32-FCC
; RUN: llc < %s -march=mipsel -mcpu=mips32r6 -relocation-model=pic  | FileCheck %s -check-prefixes=ALL,GPR,32-GPR
; RUN: llc < %s -march=mips64el -mcpu=mips64   | FileCheck %s -check-prefixes=ALL,64-FCC
; RUN: llc < %s -march=mips64el -mcpu=mips64r2 | FileCheck %s -check-prefixes=ALL,64-FCC
; RUN: llc < %s -march=mips64el -mcpu=mips64r6 | FileCheck %s -check-prefixes=ALL,GPR,64-GPR

define void @func0(float %f2, float %f3) nounwind {
entry:
; ALL-LABEL: func0:

; 32-FCC:        c.eq.s $f12, $f14
; 32-FCC:        bc1f   $BB0_2
; 64-FCC:        c.eq.s $f12, $f13
; 64-FCC:        bc1f   .LBB0_2

; 32-GPR:        cmp.eq.s $[[FGRCC:f[0-9]+]], $f12, $f14
; 64-GPR:        cmp.eq.s $[[FGRCC:f[0-9]+]], $f12, $f13
; GPR:           mfc1     $[[GPRCC:[0-9]+]], $[[FGRCC:f[0-9]+]]
; FIXME: We ought to be able to transform not+bnez -> beqz
; GPR:           not      $[[GPRCC]], $[[GPRCC]]
; 32-GPR:        bnez     $[[GPRCC]], $BB0_2
; 64-GPR:        bnezc    $[[GPRCC]], .LBB0_2

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

; 32-GPR:        cmp.ule.s $[[FGRCC:f[0-9]+]], $f14, $f12
; 64-GPR:        cmp.ule.s $[[FGRCC:f[0-9]+]], $f13, $f12
; GPR:           mfc1     $[[GPRCC:[0-9]+]], $[[FGRCC:f[0-9]+]]
; GPR-NOT:       not      $[[GPRCC]], $[[GPRCC]]
; 32-GPR:        bnez     $[[GPRCC]], $BB1_2
; 64-GPR:        bnezc    $[[GPRCC]], .LBB1_2

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

; 32-GPR:        cmp.ult.s $[[FGRCC:f[0-9]+]], $f14, $f12
; 64-GPR:        cmp.ult.s $[[FGRCC:f[0-9]+]], $f13, $f12
; GPR:           mfc1     $[[GPRCC:[0-9]+]], $[[FGRCC:f[0-9]+]]
; GPR-NOT:       not      $[[GPRCC]], $[[GPRCC]]
; 32-GPR:        beqz     $[[GPRCC]], $BB2_2
; 64-GPR:        beqzc    $[[GPRCC]], .LBB2_2

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

; 32-GPR:        cmp.eq.d $[[FGRCC:f[0-9]+]], $f12, $f14
; 64-GPR:        cmp.eq.d $[[FGRCC:f[0-9]+]], $f12, $f13
; GPR:           mfc1     $[[GPRCC:[0-9]+]], $[[FGRCC:f[0-9]+]]
; FIXME: We ought to be able to transform not+bnez -> beqz
; GPR:           not      $[[GPRCC]], $[[GPRCC]]
; 32-GPR:        bnez     $[[GPRCC]], $BB3_2
; 64-GPR:        bnezc    $[[GPRCC]], .LBB3_2

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

; 32-GPR:        cmp.ule.d $[[FGRCC:f[0-9]+]], $f14, $f12
; 64-GPR:        cmp.ule.d $[[FGRCC:f[0-9]+]], $f13, $f12
; GPR:           mfc1     $[[GPRCC:[0-9]+]], $[[FGRCC:f[0-9]+]]
; GPR-NOT:       not      $[[GPRCC]], $[[GPRCC]]
; 32-GPR:        bnez     $[[GPRCC]], $BB4_2
; 64-GPR:        bnezc    $[[GPRCC]], .LBB4_2

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

; 32-GPR:        cmp.ult.d $[[FGRCC:f[0-9]+]], $f14, $f12
; 64-GPR:        cmp.ult.d $[[FGRCC:f[0-9]+]], $f13, $f12
; GPR:           mfc1     $[[GPRCC:[0-9]+]], $[[FGRCC:f[0-9]+]]
; GPR-NOT:       not      $[[GPRCC]], $[[GPRCC]]
; 32-GPR:        beqz     $[[GPRCC]], $BB5_2
; 64-GPR:        beqzc    $[[GPRCC]], .LBB5_2

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

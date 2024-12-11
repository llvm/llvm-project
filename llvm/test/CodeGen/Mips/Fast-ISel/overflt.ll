; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s
; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32 \
; RUN:     < %s | FileCheck %s

@x = common global [128000 x float] zeroinitializer, align 4
@y = global ptr @x, align 4
@result = common global float 0.000000e+00, align 4
@.str = private unnamed_addr constant [5 x i8] c"%f \0A\00", align 1

; Function Attrs: nounwind
define void @foo() {
entry:
; CHECK-LABEL:   .ent  foo
  %0 = load ptr, ptr @y, align 4
  %arrayidx = getelementptr inbounds float, ptr %0, i32 64000
  store float 5.500000e+00, ptr %arrayidx, align 4
; CHECK:        lw      $[[REG_Y_GOT:[0-9]+]], %got(y)(${{[0-9]+}})
; CHECK:        lw      $[[REG_Y:[0-9]+]], 0($[[REG_Y_GOT]])
; CHECK:        lui     $[[REG_FPCONST_INT:[0-9]+]], 16560
; CHECK:        mtc1    $[[REG_FPCONST_INT]], $f[[REG_FPCONST:[0-9]+]]
; CHECK:        lui     $[[REG_IDX_UPPER:[0-9]+]], 3
; CHECK:        ori     $[[REG_IDX:[0-9]+]], $[[REG_IDX_UPPER]], 59392
; CHECK:        addu    $[[REG_Y_IDX:[0-9]+]], $[[REG_IDX]], $[[REG_Y]]
; CHECK:        swc1    $f[[REG_FPCONST]], 0($[[REG_Y_IDX]])
  ret void
; CHECK-LABEL:   .end  foo
}

; Function Attrs: nounwind
define void @goo() {
entry:
; CHECK-LABEL:   .ent  goo
  %0 = load ptr, ptr @y, align 4
  %arrayidx = getelementptr inbounds float, ptr %0, i32 64000
  %1 = load float, ptr %arrayidx, align 4
  store float %1, ptr @result, align 4
; CHECK-DAG:    lw      $[[REG_RESULT:[0-9]+]], %got(result)(${{[0-9]+}})
; CHECK-DAG:    lw      $[[REG_Y_GOT:[0-9]+]], %got(y)(${{[0-9]+}})
; CHECK-DAG:    lw      $[[REG_Y:[0-9]+]], 0($[[REG_Y_GOT]])
; CHECK-DAG:    lui     $[[REG_IDX_UPPER:[0-9]+]], 3
; CHECK-DAG:    ori     $[[REG_IDX:[0-9]+]], $[[REG_IDX_UPPER]], 59392
; CHECK-DAG:    addu    $[[REG_Y_IDX:[0-9]+]], $[[REG_IDX]], $[[REG_Y]]
; CHECK-DAG:    lwc1    $f[[Y_IDX:[0-9]+]], 0($[[REG_Y_IDX]])
; CHECK-DAG:    swc1    $f[[Y_IDX]], 0($[[REG_RESULT]])
; CHECK-LABEL:   .end  goo
  ret void
}

; 
; Original C code for test.
;
;float x[128000];
;ptr y = x;
;float result;


;void foo() {
;  y[64000] = 5.5;
;}

;void goo() {
;  result = y[64000];
;}

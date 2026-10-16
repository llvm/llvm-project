; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvfh -filetype=null %s 2>&1 | FileCheck %s

; CHECK: error: inline assembly requires more registers than available


define void @foo() {
entry:
  br label %for.body3.us312

for.body3.us312:                                  ; preds = %for.body3.us312, %entry
  %acc0.1303.us = phi <vscale x 16 x float> [ zeroinitializer, %entry ], [ %asmresult152.us, %for.body3.us312 ]
  %acc1.1302.us = phi <vscale x 16 x float> [ zeroinitializer, %entry ], [ %asmresult153.us, %for.body3.us312 ]
  %0 = tail call { half, half, <vscale x 16 x half>, <vscale x 16 x float>, <vscale x 16 x float>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half> } asm sideeffect "\0A\09flh $0, 0($26) \0A\09flh $1, 0($27) \0A\09vsetvli zero, $28, e16, m4, ta, ma \0A\09vle16.v $2, ($29) \0A\09vfwadd.vf $3, $2, $0 \0A\09vfwsub.vf $4, $2, $1 \0A\09flh $5, 0($30) \0A\09flh $6, 0($31) \0A\09vle16.v $7, ($32) \0A\09vfwmul.vf $3, $7, $5 \0A\09vfwadd.vf $4, $7, $6 \0A\09flh $8, 0($33) \0A\09flh $9, 0($34) \0A\09vle16.v $10, ($35) \0A\09vfwsub.vf $3, $10, $8 \0A\09vfwmul.vf $4, $10, $9 \0A\09flh $11, 0($36) \0A\09flh $12, 0($37) \0A\09vle16.v $13, ($38) \0A\09vfwadd.vf $3, $13, $11 \0A\09vfwsub.vf $4, $13, $12 \0A\09flh $14, 0($39) \0A\09flh $15, 0($40) \0A\09vle16.v $16, ($41) \0A\09vfwmul.vf $3, $16, $14 \0A\09vfwadd.vf $4, $16, $15 \0A\09flh $17, 0($42) \0A\09flh $18, 0($43) \0A\09vle16.v $19, ($44) \0A\09vfwsub.vf $3, $19, $17 \0A\09vfwmul.vf $4, $19, $18 \0A\09flh $20, 0($45) \0A\09flh $21, 0($46) \0A\09vle16.v $22, ($47) \0A\09vfwadd.vf $3, $22, $20 \0A\09vfwsub.vf $4, $22, $21 \0A\09flh $23, 0($48) \0A\09flh $24, 0($49) \0A\09vle16.v $25, ($50) \0A\09vfwmul.vf $3, $25, $23 \0A\09vfwadd.vf $4, $25, $24 \0A\09", "=&f,=&f,=&^vr,=&^vr,=&^vr,=&f,=&f,=&^vr,=&f,=&f,=&^vr,=&f,=&f,=&^vr,=&f,=&f,=&^vr,=&f,=&f,=&^vr,=&f,=&f,=&^vr,=&f,=&f,=&^vr,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,3,4,~{vtype},~{vl},~{memory},~{vl},~{vtype}"(ptr null, ptr null, i64 0, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, <vscale x 16 x float> %acc0.1303.us, <vscale x 16 x float> %acc1.1302.us)
  %asmresult152.us = extractvalue { half, half, <vscale x 16 x half>, <vscale x 16 x float>, <vscale x 16 x float>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half> } %0, 3
  %asmresult153.us = extractvalue { half, half, <vscale x 16 x half>, <vscale x 16 x float>, <vscale x 16 x float>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half>, half, half, <vscale x 16 x half> } %0, 4
  br label %for.body3.us312
}

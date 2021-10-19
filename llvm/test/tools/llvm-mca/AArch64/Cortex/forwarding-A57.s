# RUN: llvm-mca -mtriple=aarch64-none-eabi -mcpu=cortex-a57 -iterations=1 -timeline < %s | FileCheck %s

# CHECK: [0] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      12
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER  ..   fmul   v0.2s, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     DeeeeeeeeeER   fmla   v0.2s, v1.2s, v2.2s

# CHECK: [1] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      13
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER  . .   fmul  v0.4s, v1.4s, v2.4s
# CHECK-NEXT: [0,1]     DeeeeeeeeeeER   fmla  v0.4s, v1.4s, v2.4s

# CHECK: [2] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      12
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER  ..   fmulx  v0.2s, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     DeeeeeeeeeER   fmls   v0.2s, v1.2s, v2.2s

# CHECK: [3] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      13
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER  . .   fmulx v0.4s, v1.4s, v2.4s
# CHECK-NEXT: [0,1]     DeeeeeeeeeeER   fmls  v0.4s, v1.4s, v2.4s

# CHECK: [4] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      16
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeeeeeER   .   fmla       v0.2s, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D====eeeeeeeeeER   fmla       v0.2s, v3.2s, v4.2s

# CHECK: [5] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      16
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeeeeeER   .   fmls       v0.2s, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D====eeeeeeeeeER   fmls       v0.2s, v3.2s, v4.2s

# CHECK: [6] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      12
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER  ..   fmul   d4, d5, d6
# CHECK-NEXT: [0,1]     DeeeeeeeeeER   fmadd  d1, d2, d3, d4

# CHECK: [7] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      12
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER  ..   fmul   d4, d5, d6
# CHECK-NEXT: [0,1]     DeeeeeeeeeER   fmadd  d1, d2, d3, d4

# CHECK: [8] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      16
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeeeeeER   .   fmadd      d4, d5, d6, d7
# CHECK-NEXT: [0,1]     D====eeeeeeeeeER   fmadd      d1, d2, d3, d4

# CHECK: [9] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      16
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeeeeeER   .   fmsub      d4, d5, d6, d7
# CHECK-NEXT: [0,1]     D====eeeeeeeeeER   fmsub      d1, d2, d3, d4

# CHECK: [10] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      16
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeeeeeER   .   fnmadd     d4, d5, d6, d7
# CHECK-NEXT: [0,1]     D====eeeeeeeeeER   fnmadd     d1, d2, d3, d4

# CHECK: [11] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      16
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeeeeeER   .   fnmsub     d4, d5, d6, d7
# CHECK-NEXT: [0,1]     D====eeeeeeeeeER   fnmsub     d1, d2, d3, d4

# CHECK: [12] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      8
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeER.   saba       v0.2s, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D=eeeeER   saba       v0.2s, v3.2s, v4.2s

# CHECK: [13] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      8
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeER.   sabal      v0.2d, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D=eeeeER   sabal      v0.2d, v3.2s, v4.2s

# CHECK: [14] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      8
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeER.   uaba       v0.2s, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D=eeeeER   uaba       v0.2s, v3.2s, v4.2s

# CHECK: [15] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      8
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeER.   uabal      v0.2d, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D=eeeeER   uabal      v0.2d, v3.2s, v4.2s

# CHECK: [16] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      8
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeER.   sadalp     v0.1d, v1.2s
# CHECK-NEXT: [0,1]     D=eeeeER   sadalp     v0.1d, v2.2s

# CHECK: [17] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      8
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeER.   uadalp     v0.1d, v1.2s
# CHECK-NEXT: [0,1]     D=eeeeER   uadalp     v0.1d, v2.2s

# CHECK: [18] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      8
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeER.   srsra      v0.8b, v1.8b, #3
# CHECK-NEXT: [0,1]     D=eeeeER   srsra      v0.8b, v2.8b, #3

# CHECK: [19] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      8
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeER.   ursra      v0.8b, v1.8b, #3
# CHECK-NEXT: [0,1]     D=eeeeER   ursra      v0.8b, v2.8b, #3

# CHECK: [20] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      8
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeER.   usra       v0.4s, v1.4s, #3
# CHECK-NEXT: [0,1]     D=eeeeER   usra       v0.4s, v2.4s, #3

# CHECK: [21] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      9
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER.   mul       v0.2s, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D=eeeeeER   mla       v0.2s, v1.2s, v2.2s

# CHECK: [22] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      13
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER  . .   pmul  v0.8b, v1.8b, v2.8b
# CHECK-NEXT: [0,1]     D=====eeeeeER   mla   v0.8b, v1.8b, v2.8b

# CHECK: [23] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      13
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER  . .   sqdmulh       v0.2s, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D=====eeeeeER   mla   v0.2s, v1.2s, v2.2s

# CHECK: [24] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      13
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER  . .   sqrdmulh      v0.2s, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D=====eeeeeER   mla   v0.2s, v1.2s, v2.2s

# CHECK: [25] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      9
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER.   smull     v0.2d, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D=eeeeeER   smlal     v0.2d, v1.2s, v2.2s

# CHECK: [26] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      9
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER.   umull     v0.2d, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D=eeeeeER   umlal     v0.2d, v1.2s, v2.2s

# CHECK: [27] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      13
# CHECK: Timeline view:
# CHECK: [0,0] DeeeeeER . . sqdmull v0.2d, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D=====eeeeeER   smlal v0.2d, v1.2s, v2.2s

# CHECK: [28] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      13
# CHECK: Timeline view:
# CHECK: [0,0] DeeeeeER . . pmull v0.8h, v1.8b, v2.8b
# CHECK-NEXT: [0,1]     D=====eeeeeER   smlal v0.8h, v1.8b, v2.8b

# CHECK: [29] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      13
# CHECK: Timeline view:
# CHECK: [0,0] DeeeeeER . . pmull2 v0.8h, v1.16b, v2.16b
# CHECK-NEXT: [0,1]     D=====eeeeeER   smlal v0.8h, v1.8b, v2.8b

# CHECK: [30] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      9
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER.   mla       v0.2s, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D=eeeeeER   mla       v0.2s, v1.2s, v2.2s

# CHECK: [31] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      11
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeeER .   mla     v0.4s, v1.4s, v2.4s
# CHECK-NEXT: [0,1]     .D=eeeeeeER   mla     v0.4s, v1.4s, v2.4s

# CHECK: [32] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      9
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER.   mls       v0.2s, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D=eeeeeER   mls       v0.2s, v1.2s, v2.2s

# CHECK: [33] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      11
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeeER .   mls     v0.4s, v1.4s, v2.4s
# CHECK-NEXT: [0,1]     .D=eeeeeeER   mls     v0.4s, v1.4s, v2.4s

# CHECK: [34] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      9
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER.   smlal     v0.2d, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D=eeeeeER   smlal     v0.2d, v1.2s, v2.2s

# CHECK: [35] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      9
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER.   smlsl     v0.2d, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D=eeeeeER   smlsl     v0.2d, v1.2s, v2.2s

# CHECK: [36] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      9
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER.   umlal     v0.2d, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D=eeeeeER   umlal     v0.2d, v1.2s, v2.2s

# CHECK: [37] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      9
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER.   umlsl     v0.2d, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D=eeeeeER   umlsl     v0.2d, v1.2s, v2.2s

# CHECK: [38] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      10
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER .   sqdmlal  v0.2d, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D==eeeeeER   sqdmlal  v0.2d, v1.2s, v2.2s

# CHECK: [39] Code Region
# CHECK: Instructions:      2
# CHECK-NEXT: Total Cycles:      10
# CHECK: Timeline view:
# CHECK: [0,0]     DeeeeeER .   sqdmlsl  v0.2d, v1.2s, v2.2s
# CHECK-NEXT: [0,1]     D==eeeeeER   sqdmlsl  v0.2d, v1.2s, v2.2s

# ASIMD FP Instructions
# FMUL, FMULX, FMLA, FMLS are impacted
# testing only a subset of combinations
# LLVM-MCA-BEGIN
fmul v0.2s, v1.2s, v2.2s
fmla v0.2s, v1.2s, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
fmul v0.4s, v1.4s, v2.4s
fmla v0.4s, v1.4s, v2.4s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
fmulx v0.2s, v1.2s, v2.2s
fmls v0.2s, v1.2s, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
fmulx v0.4s, v1.4s, v2.4s
fmls v0.4s, v1.4s, v2.4s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
fmla v0.2s, v1.2s, v2.2s
fmla v0.2s, v3.2s, v4.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
fmls v0.2s, v1.2s, v2.2s
fmls v0.2s, v3.2s, v4.2s
# LLVM-MCA-END


# FP Multiply Instructions
# FMUL, FMUL, FNMUL, FMADD, FMSUB, FNMADD, FNMSUB are impacted
# testing only a subset of combinations
# LLVM-MCA-BEGIN
fmul   d4, d5, d6
fmadd d1, d2, d3, d4
# LLVM-MCA-END

# LLVM-MCA-BEGIN
fmul   d4, d5, d6
fmadd d1, d2, d3, d4
# LLVM-MCA-END

# LLVM-MCA-BEGIN
fmadd d4, d5, d6, d7
fmadd d1, d2, d3, d4
# LLVM-MCA-END

# LLVM-MCA-BEGIN
fmsub d4, d5, d6, d7
fmsub d1, d2, d3, d4
# LLVM-MCA-END

# LLVM-MCA-BEGIN
fnmadd d4, d5, d6, d7
fnmadd d1, d2, d3, d4
# LLVM-MCA-END

# LLVM-MCA-BEGIN
fnmsub d4, d5, d6, d7
fnmsub d1, d2, d3, d4
# LLVM-MCA-END



# ASIMD Integer Instructions X-Unit
# SABA, UABA, SABAL, UABAL, SADALP, UADALP, SRSRA, USRA, URSRA are impacted
# testing only a subset of combinations

# LLVM-MCA-BEGIN
saba v0.2s, v1.2s, v2.2s
saba v0.2s, v3.2s, v4.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
sabal v0.2d, v1.2s, v2.2s
sabal v0.2d, v3.2s, v4.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
uaba v0.2s, v1.2s, v2.2s
uaba v0.2s, v3.2s, v4.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
uabal v0.2d, v1.2s, v2.2s
uabal v0.2d, v3.2s, v4.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
sadalp v0.1d, v1.2s
sadalp v0.1d, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
uadalp v0.1d, v1.2s
uadalp v0.1d, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
srsra v0.8b, v1.8b, #3
srsra v0.8b, v2.8b, #3
# LLVM-MCA-END

# LLVM-MCA-BEGIN
ursra v0.8b, v1.8b, #3
ursra v0.8b, v2.8b, #3
# LLVM-MCA-END

# LLVM-MCA-BEGIN
usra v0.4s, v1.4s, #3
usra v0.4s, v2.4s, #3
# LLVM-MCA-END


# ASIMD Multiply Instructions X-Unit
# pmuls and sqd/sqrdmuls dont forward

# MULs
# LLVM-MCA-BEGIN
mul v0.2s, v1.2s, v2.2s
mla v0.2s, v1.2s, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
pmul v0.8b, v1.8b, v2.8b
mla v0.8b, v1.8b, v2.8b
# LLVM-MCA-END

# LLVM-MCA-BEGIN
sqdmulh v0.2s, v1.2s, v2.2s
mla v0.2s, v1.2s, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
sqrdmulh v0.2s, v1.2s, v2.2s
mla v0.2s, v1.2s, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
smull v0.2d, v1.2s, v2.2s
smlal v0.2d, v1.2s, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
umull v0.2d, v1.2s, v2.2s
umlal v0.2d, v1.2s, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
sqdmull v0.2d, v1.2s, v2.2s
smlal v0.2d, v1.2s, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
pmull.8h v0, v1, v2
smlal.8h v0, v1, v2
# LLVM-MCA-END

# LLVM-MCA-BEGIN
pmull2.8h v0, v1, v2
smlal.8h v0, v1, v2
# LLVM-MCA-END


# MLAs
# LLVM-MCA-BEGIN
mla v0.2s, v1.2s, v2.2s
mla v0.2s, v1.2s, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
mla v0.4s, v1.4s, v2.4s
mla v0.4s, v1.4s, v2.4s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
mls v0.2s, v1.2s, v2.2s
mls v0.2s, v1.2s, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
mls v0.4s, v1.4s, v2.4s
mls v0.4s, v1.4s, v2.4s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
smlal v0.2d, v1.2s, v2.2s
smlal v0.2d, v1.2s, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
smlsl v0.2d, v1.2s, v2.2s
smlsl v0.2d, v1.2s, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
umlal v0.2d, v1.2s, v2.2s
umlal v0.2d, v1.2s, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
umlsl v0.2d, v1.2s, v2.2s
umlsl v0.2d, v1.2s, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
sqdmlal v0.2d, v1.2s, v2.2s
sqdmlal v0.2d, v1.2s, v2.2s
# LLVM-MCA-END

# LLVM-MCA-BEGIN
sqdmlsl v0.2d, v1.2s, v2.2s
sqdmlsl v0.2d, v1.2s, v2.2s
# LLVM-MCA-END

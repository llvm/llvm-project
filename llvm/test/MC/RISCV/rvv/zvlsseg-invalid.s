# RUN: not llvm-mc -triple=riscv64 --mattr=+v %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
 
vluxseg2ei8.v v8, (a0), v8, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vluxseg2ei8.v v8, (a0), v8, v0.t

vluxseg2ei8.v v8, (a0), v9, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vluxseg2ei8.v v8, (a0), v9, v0.t
 
vluxseg3ei8.v v8, (a0), v10, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vluxseg3ei8.v v8, (a0), v10, v0.t
 
vluxseg4ei8.v v8, (a0), v11, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vluxseg4ei8.v v8, (a0), v11, v0.t
 
vluxseg5ei8.v v8, (a0), v12, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vluxseg5ei8.v v8, (a0), v12, v0.t
 
vluxseg6ei8.v v8, (a0), v13, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vluxseg6ei8.v v8, (a0), v13, v0.t
 
vluxseg7ei8.v v8, (a0), v14, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vluxseg7ei8.v v8, (a0), v14, v0.t
 
vluxseg8ei8.v v8, (a0), v15, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vluxseg8ei8.v v8, (a0), v15, v0.t
 
vluxseg2ei8.v v8, (a0), v8
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vluxseg2ei8.v v8, (a0), v8
 
vluxseg2ei16.v v8, (a0), v8, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vluxseg2ei16.v v8, (a0), v8, v0.t
 
vluxseg2ei16.v v8, (a0), v8
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vluxseg2ei16.v v8, (a0), v8
 
vluxseg2ei32.v v8, (a0), v8, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vluxseg2ei32.v v8, (a0), v8, v0.t
 
vluxseg2ei32.v v8, (a0), v8
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vluxseg2ei32.v v8, (a0), v8
 
vluxseg2ei64.v v8, (a0), v8, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vluxseg2ei64.v v8, (a0), v8, v0.t
 
vluxseg2ei64.v v8, (a0), v8
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vluxseg2ei64.v v8, (a0), v8
 
vloxseg2ei8.v v8, (a0), v8, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vloxseg2ei8.v v8, (a0), v8, v0.t
 
vloxseg2ei8.v v8, (a0), v8
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vloxseg2ei8.v v8, (a0), v8
 
vloxseg2ei16.v v8, (a0), v8, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vloxseg2ei16.v v8, (a0), v8, v0.t
 
vloxseg2ei16.v v8, (a0), v8
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vloxseg2ei16.v v8, (a0), v8
 
vloxseg2ei32.v v8, (a0), v8, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vloxseg2ei32.v v8, (a0), v8, v0.t
 
vloxseg2ei32.v v8, (a0), v8
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vloxseg2ei32.v v8, (a0), v8
 
vloxseg2ei64.v v8, (a0), v8, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vloxseg2ei64.v v8, (a0), v8, v0.t
 
vloxseg2ei64.v v8, (a0), v8
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vloxseg2ei64.v v8, (a0), v8

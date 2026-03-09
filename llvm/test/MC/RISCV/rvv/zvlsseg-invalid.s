# RUN: not llvm-mc -triple=riscv64 --mattr=+v %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
 
vluxseg2ei8.v v8, (a0), v8, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vluxseg2ei8.v v8, (a0), v8, v0.t
 
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

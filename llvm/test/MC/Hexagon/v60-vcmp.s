#RUN: llvm-mc -triple=hexagon -mcpu=hexagonv60 -filetype=obj %s | \
#RUN: llvm-objdump -triple=hexagon -mcpu=hexagonv60 -d - | \
#RUN: FileCheck %s

#CHECK: 1c81f142 { q2 |= vcmp.eq(v17.b{{ *}},{{ *}}v1.b) }
q2|=vcmp.eq(v17.b,v1.b)

#CHECK: 1c84fb2a { q2 &= vcmp.gt(v27.uw{{ *}},{{ *}}v4.uw) }
q2&=vcmp.gt(v27.uw,v4.uw)

#CHECK: 1c8cf826 { q2 &= vcmp.gt(v24.uh{{ *}},{{ *}}v12.uh) }
q2&=vcmp.gt(v24.uh,v12.uh)

#CHECK: 1c80e720 { q0 &= vcmp.gt(v7.ub{{ *}},{{ *}}v0.ub) }
q0&=vcmp.gt(v7.ub,v0.ub)

#CHECK: 1c9aed1a { q2 &= vcmp.gt(v13.w{{ *}},{{ *}}v26.w) }
q2&=vcmp.gt(v13.w,v26.w)

#CHECK: 1c8de516 { q2 &= vcmp.gt(v5.h{{ *}},{{ *}}v13.h) }
q2&=vcmp.gt(v5.h,v13.h)

#CHECK: 1c8dfc11 { q1 &= vcmp.gt(v28.b{{ *}},{{ *}}v13.b) }
q1&=vcmp.gt(v28.b,v13.b)

#CHECK: 1c94fa0b { q3 &= vcmp.eq(v26.w{{ *}},{{ *}}v20.w) }
q3&=vcmp.eq(v26.w,v20.w)

#CHECK: 1c83e206 { q2 &= vcmp.eq(v2.h{{ *}},{{ *}}v3.h) }
q2&=vcmp.eq(v2.h,v3.h)

#CHECK: 1c85e900 { q0 &= vcmp.eq(v9.b{{ *}},{{ *}}v5.b) }
q0&=vcmp.eq(v9.b,v5.b)

#CHECK: 1c9cfca8 { q0 ^= vcmp.gt(v28.uw{{ *}},{{ *}}v28.uw) }
q0^=vcmp.gt(v28.uw,v28.uw)

#CHECK: 1c81faa0 { q0 ^= vcmp.gt(v26.ub{{ *}},{{ *}}v1.ub) }
q0^=vcmp.gt(v26.ub,v1.ub)

#CHECK: 1c96f0a4 { q0 ^= vcmp.gt(v16.uh{{ *}},{{ *}}v22.uh) }
q0^=vcmp.gt(v16.uh,v22.uh)

#CHECK: 1c9bf795 { q1 ^= vcmp.gt(v23.h{{ *}},{{ *}}v27.h) }
q1^=vcmp.gt(v23.h,v27.h)

#CHECK: 1c9de698 { q0 ^= vcmp.gt(v6.w{{ *}},{{ *}}v29.w) }
q0^=vcmp.gt(v6.w,v29.w)

#CHECK: 1c82ef8a { q2 ^= vcmp.eq(v15.w{{ *}},{{ *}}v2.w) }
q2^=vcmp.eq(v15.w,v2.w)

#CHECK: 1c99e891 { q1 ^= vcmp.gt(v8.b{{ *}},{{ *}}v25.b) }
q1^=vcmp.gt(v8.b,v25.b)

#CHECK: 1c8afe55 { q1 |= vcmp.gt(v30.h{{ *}},{{ *}}v10.h) }
q1|=vcmp.gt(v30.h,v10.h)

#CHECK: 1c92ef50 { q0 |= vcmp.gt(v15.b{{ *}},{{ *}}v18.b) }
q0|=vcmp.gt(v15.b,v18.b)

#CHECK: 1c9ffb4b { q3 |= vcmp.eq(v27.w{{ *}},{{ *}}v31.w) }
q3|=vcmp.eq(v27.w,v31.w)

#CHECK: 1c87e944 { q0 |= vcmp.eq(v9.h{{ *}},{{ *}}v7.h) }
q0|=vcmp.eq(v9.h,v7.h)

#CHECK: 1c8ee768 { q0 |= vcmp.gt(v7.uw{{ *}},{{ *}}v14.uw) }
q0|=vcmp.gt(v7.uw,v14.uw)

#CHECK: 1c92e265 { q1 |= vcmp.gt(v2.uh{{ *}},{{ *}}v18.uh) }
q1|=vcmp.gt(v2.uh,v18.uh)

#CHECK: 1c80f062 { q2 |= vcmp.gt(v16.ub{{ *}},{{ *}}v0.ub) }
q2|=vcmp.gt(v16.ub,v0.ub)

#CHECK: 1c91f75a { q2 |= vcmp.gt(v23.w{{ *}},{{ *}}v17.w) }
q2|=vcmp.gt(v23.w,v17.w)

#CHECK: 1c86fe84 { q0 ^= vcmp.eq(v30.h{{ *}},{{ *}}v6.h) }
q0^=vcmp.eq(v30.h,v6.h)

#CHECK: 1c86ec82 { q2 ^= vcmp.eq(v12.b{{ *}},{{ *}}v6.b) }
q2^=vcmp.eq(v12.b,v6.b)

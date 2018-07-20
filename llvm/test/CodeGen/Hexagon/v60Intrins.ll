; RUN: llc -march=hexagon -mcpu=hexagonv60 -O2 -disable-post-ra  < %s | FileCheck %s

; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} |= vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vcmp.eq(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vcmp.eq(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vcmp.eq(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} &= vcmp.eq(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} &= vcmp.eq(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} &= vcmp.eq(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} |= vcmp.eq(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} |= vcmp.eq(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} |= vcmp.eq(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} ^= vcmp.eq(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} ^= vcmp.eq(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} ^= vcmp.eq(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vcmp.gt(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vcmp.gt(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vcmp.gt(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vcmp.gt(v{{[0-9]*}}.uh,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vcmp.gt(v{{[0-9]*}}.uw,v{{[0-9]*}}.uw)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vcmp.gt(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} &= vcmp.gt(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} &= vcmp.gt(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} &= vcmp.gt(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} &= vcmp.gt(v{{[0-9]*}}.uh,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} &= vcmp.gt(v{{[0-9]*}}.uw,v{{[0-9]*}}.uw)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} &= vcmp.gt(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} |= vcmp.gt(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} |= vcmp.gt(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} |= vcmp.gt(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} |= vcmp.gt(v{{[0-9]*}}.uh,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} |= vcmp.gt(v{{[0-9]*}}.uw,v{{[0-9]*}}.uw)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} |= vcmp.gt(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} ^= vcmp.gt(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} ^= vcmp.gt(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} ^= vcmp.gt(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} ^= vcmp.gt(v{{[0-9]*}}.uh,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} ^= vcmp.gt(v{{[0-9]*}}.uw,v{{[0-9]*}}.uw)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} ^= vcmp.gt(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = xor{{[0-9]*}}(q{{[0-3]}},q{{[0-3]}})
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: v{{[0-9]*}} = v
; CHECK: v{{[0-9]*}} = valign(v{{[0-9]*}},v{{[0-9]*}},#1)
; CHECK: v{{[0-9]*}} = valign(v{{[0-9]*}},v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: v{{[0-9]*}} = vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: v{{[0-9]*}} = vand(v{{[0-9]*}},v{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: v{{[0-9]*}} |= vand(q{{[0-3]}},r{{[0-9]*}})
; CHECK: v{{[0-9]*}} = vdelta(v{{[0-9]*}},v{{[0-9]*}})
; CHECK: v{{[0-9]*}} = vlalign(v{{[0-9]*}},v{{[0-9]*}},#1)
; CHECK: v{{[0-9]*}} = vlalign(v{{[0-9]*}},v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: v{{[0-9]*}} = vmux(q{{[0-3]}},v{{[0-9]*}},v{{[0-9]*}})
; CHECK: v{{[0-9]*}} = vnot(v{{[0-9]*}})
; CHECK: v{{[0-9]*}} = vor{{[0-9]*}}(v{{[0-9]*}},v{{[0-9]*}})
; CHECK: v{{[0-9]*}} = vr{{[0-9]*}}delta(v{{[0-9]*}},v{{[0-9]*}})
; CHECK: v{{[0-9]*}} = vr{{[0-9]*}}or{{[0-9]*}}(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: v{{[0-9]*}} = vxor{{[0-9]*}}(v{{[0-9]*}},v{{[0-9]*}})
; CHECK: v{{[0-9]*}}.b = vadd(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.b = vasr{{[0-9]*}}(v{{[0-9]*}}.h,v{{[0-9]*}}.h,r{{[0-9]*}}):{{[0-9]*}}r{{[0-9]*}}nd:{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.b = vdeal(v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.b = vdeale(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.b = vlut32(v{{[0-9]*}}.b,v{{[0-9]*}}.b,r{{[0-9]*}})
; CHECK: v{{[0-9]*}}.b |= vlut32(v{{[0-9]*}}.b,v{{[0-9]*}}.b,r{{[0-9]*}})
; CHECK: v{{[0-9]*}}.b = vnav{{[0-9]*}}g(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}.b = vpack(v{{[0-9]*}}.h,v{{[0-9]*}}.h):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.b = vpacke(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.b = vpacko(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.b = vr{{[0-9]*}}ound(v{{[0-9]*}}.h,v{{[0-9]*}}.h):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.b = vshuff(v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.b = vshuffe(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.b = vshuffo(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.b = vsub(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.h = vabs(v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vabs(v{{[0-9]*}}.h):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.h = vadd(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vadd(v{{[0-9]*}}.h,v{{[0-9]*}}.h):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.h = vasl(v{{[0-9]*}}.h,r{{[0-9]*}})
; CHECK: v{{[0-9]*}}.h = vasl(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vasr{{[0-9]*}}(v{{[0-9]*}}.h,r{{[0-9]*}})
; CHECK: v{{[0-9]*}}.h = vasr{{[0-9]*}}(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vasr{{[0-9]*}}(v{{[0-9]*}}.w,v{{[0-9]*}}.w,r{{[0-9]*}})
; CHECK: v{{[0-9]*}}.h = vasr{{[0-9]*}}(v{{[0-9]*}}.w,v{{[0-9]*}}.w,r{{[0-9]*}}):{{[0-9]*}}r{{[0-9]*}}nd:{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.h = vasr{{[0-9]*}}(v{{[0-9]*}}.w,v{{[0-9]*}}.w,r{{[0-9]*}}):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.h = vav{{[0-9]*}}g(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vav{{[0-9]*}}g(v{{[0-9]*}}.h,v{{[0-9]*}}.h):{{[0-9]*}}r{{[0-9]*}}nd
; CHECK: v{{[0-9]*}}.h = vdeal(v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vdmpy(v{{[0-9]*}}.ub,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.h += vdmpy(v{{[0-9]*}}.ub,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.h = vlsr{{[0-9]*}}(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vmax(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vmin(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vmpy(v{{[0-9]*}}.h,r{{[0-9]*}}.h):{{[0-9]*}}<<1:{{[0-9]*}}r{{[0-9]*}}nd:{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.h = vmpy(v{{[0-9]*}}.h,r{{[0-9]*}}.h):{{[0-9]*}}<<1:{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.h = vmpy(v{{[0-9]*}}.h,v{{[0-9]*}}.h):{{[0-9]*}}<<1:{{[0-9]*}}r{{[0-9]*}}nd:{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.h = vmpyi(v{{[0-9]*}}.h,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.h = vmpyi(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h += vmpyi(v{{[0-9]*}}.h,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.h += vmpyi(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vnav{{[0-9]*}}g(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vnor{{[0-9]*}}mamt(v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vpack(v{{[0-9]*}}.w,v{{[0-9]*}}.w):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.h = vpacke(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}.h = vpacko(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}.h = vpopcount(v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vr{{[0-9]*}}ound(v{{[0-9]*}}.w,v{{[0-9]*}}.w):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.h = vsat(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}.h = vshuff(v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vshuffe(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vshuffo(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vsub(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.h = vsub(v{{[0-9]*}}.h,v{{[0-9]*}}.h):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.ub = vabsdiff(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}.ub = vadd(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.ub = vasr{{[0-9]*}}(v{{[0-9]*}}.h,v{{[0-9]*}}.h,r{{[0-9]*}}):{{[0-9]*}}r{{[0-9]*}}nd:{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.ub = vasr{{[0-9]*}}(v{{[0-9]*}}.h,v{{[0-9]*}}.h,r{{[0-9]*}}):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.ub = vav{{[0-9]*}}g(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}.ub = vav{{[0-9]*}}g(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub):{{[0-9]*}}r{{[0-9]*}}nd
; CHECK: v{{[0-9]*}}.ub = vmax(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}.ub = vmin(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}.ub = vpack(v{{[0-9]*}}.h,v{{[0-9]*}}.h):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.ub = vr{{[0-9]*}}ound(v{{[0-9]*}}.h,v{{[0-9]*}}.h):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.ub = vsat(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.ub = vsub(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.uh = vabsdiff(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.uh = vabsdiff(v{{[0-9]*}}.uh,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}.uh = vadd(v{{[0-9]*}}.uh,v{{[0-9]*}}.uh):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.uh = vasr{{[0-9]*}}(v{{[0-9]*}}.w,v{{[0-9]*}}.w,r{{[0-9]*}}):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.uh = vav{{[0-9]*}}g(v{{[0-9]*}}.uh,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}.uh = vav{{[0-9]*}}g(v{{[0-9]*}}.uh,v{{[0-9]*}}.uh):{{[0-9]*}}r{{[0-9]*}}nd
; CHECK: v{{[0-9]*}}.uh = vcl0(v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}.uh = vlsr{{[0-9]*}}(v{{[0-9]*}}.uh,r{{[0-9]*}})
; CHECK: v{{[0-9]*}}.uh = vmax(v{{[0-9]*}}.uh,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}.uh = vmin(v{{[0-9]*}}.uh,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}.uh = vpack(v{{[0-9]*}}.w,v{{[0-9]*}}.w):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.uh = vr{{[0-9]*}}ound(v{{[0-9]*}}.w,v{{[0-9]*}}.w):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.uh = vsub(v{{[0-9]*}}.uh,v{{[0-9]*}}.uh):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.uw = vabsdiff(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}.uw = vcl0(v{{[0-9]*}}.uw)
; CHECK: v{{[0-9]*}}.uw = vlsr{{[0-9]*}}(v{{[0-9]*}}.uw,r{{[0-9]*}})
; CHECK: v{{[0-9]*}}.uw = vr{{[0-9]*}}mpy(v{{[0-9]*}}.ub,r{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}.uw = vr{{[0-9]*}}mpy(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}.uw += vr{{[0-9]*}}mpy(v{{[0-9]*}}.ub,r{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}.uw += vr{{[0-9]*}}mpy(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}.w = vabs(v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}.w = vabs(v{{[0-9]*}}.w):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.w = vadd(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}.w = vadd(v{{[0-9]*}}.w,v{{[0-9]*}}.w):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.w = vasl(v{{[0-9]*}}.w,r{{[0-9]*}})
; CHECK: v{{[0-9]*}}.w = vasl(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}.w += vasl(v{{[0-9]*}}.w,r{{[0-9]*}})
; CHECK: v{{[0-9]*}}.w = vasr{{[0-9]*}}(v{{[0-9]*}}.w,r{{[0-9]*}})
; CHECK: v{{[0-9]*}}.w = vasr{{[0-9]*}}(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}.w += vasr{{[0-9]*}}(v{{[0-9]*}}.w,r{{[0-9]*}})
; CHECK: v{{[0-9]*}}.w = vav{{[0-9]*}}g(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}.w = vav{{[0-9]*}}g(v{{[0-9]*}}.w,v{{[0-9]*}}.w):{{[0-9]*}}r{{[0-9]*}}nd
; CHECK: v{{[0-9]*}}.w = vdmpy(v{{[0-9]*}}.h,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.w = vdmpy(v{{[0-9]*}}.h,r{{[0-9]*}}.h):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.w = vdmpy(v{{[0-9]*}}.h,r{{[0-9]*}}.uh):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.w = vdmpy(v{{[0-9]*}}.h,v{{[0-9]*}}.h):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.w = vdmpy(v{{[0-9]*}}:{{[0-9]*}}.h,r{{[0-9]*}}.h):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.w = vdmpy(v{{[0-9]*}}:{{[0-9]*}}.h,r{{[0-9]*}}.uh,#1):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.w += vdmpy(v{{[0-9]*}}.h,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.w += vdmpy(v{{[0-9]*}}.h,r{{[0-9]*}}.h):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.w += vdmpy(v{{[0-9]*}}.h,r{{[0-9]*}}.uh):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.w += vdmpy(v{{[0-9]*}}.h,v{{[0-9]*}}.h):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.w += vdmpy(v{{[0-9]*}}:{{[0-9]*}}.h,r{{[0-9]*}}.h):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.w += vdmpy(v{{[0-9]*}}:{{[0-9]*}}.h,r{{[0-9]*}}.uh,#1):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.w = vinser{{[0-9]*}}t(r{{[0-9]*}})
; CHECK: v{{[0-9]*}}.w = vinser{{[0-9]*}}t(r{{[0-9]*}})
; CHECK: v{{[0-9]*}}.w = vinser{{[0-9]*}}t(r{{[0-9]*}})
; CHECK: v{{[0-9]*}}.w = vlsr{{[0-9]*}}(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}.w = vmax(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}.w = vmin(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}.w = vmpye(v{{[0-9]*}}.w,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}.w = vmpyi(v{{[0-9]*}}.w,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.w = vmpyi(v{{[0-9]*}}.w,r{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.w += vmpyi(v{{[0-9]*}}.w,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.w += vmpyi(v{{[0-9]*}}.w,r{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.w = vmpyie(v{{[0-9]*}}.w,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}.w += vmpyie(v{{[0-9]*}}.w,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.w += vmpyie(v{{[0-9]*}}.w,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}.w = vmpyieo(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.w = vmpyio(v{{[0-9]*}}.w,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}.w = vmpyo(v{{[0-9]*}}.w,v{{[0-9]*}}.h):{{[0-9]*}}<<1:{{[0-9]*}}r{{[0-9]*}}nd:{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.w = vmpyo(v{{[0-9]*}}.w,v{{[0-9]*}}.h):{{[0-9]*}}<<1:{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}.w += vmpyo(v{{[0-9]*}}.w,v{{[0-9]*}}.h):{{[0-9]*}}<<1:{{[0-9]*}}r{{[0-9]*}}nd:{{[0-9]*}}sat:{{[0-9]*}}shift
; CHECK: v{{[0-9]*}}.w += vmpyo(v{{[0-9]*}}.w,v{{[0-9]*}}.h):{{[0-9]*}}<<1:{{[0-9]*}}sat:{{[0-9]*}}shift
; CHECK: v{{[0-9]*}}.w = vnav{{[0-9]*}}g(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}.w = vnor{{[0-9]*}}mamt(v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}.w = vr{{[0-9]*}}mpy(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.w = vr{{[0-9]*}}mpy(v{{[0-9]*}}.ub,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.w = vr{{[0-9]*}}mpy(v{{[0-9]*}}.ub,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.w += vr{{[0-9]*}}mpy(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.w += vr{{[0-9]*}}mpy(v{{[0-9]*}}.ub,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.w += vr{{[0-9]*}}mpy(v{{[0-9]*}}.ub,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}.w = vsub(v{{[0-9]*}}.w,v{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}.w = vsub(v{{[0-9]*}}.w,v{{[0-9]*}}.w):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}:{{[0-9]*}} = vcombine(v{{[0-9]*}},v{{[0-9]*}})
; CHECK: v{{[0-9]*}}:{{[0-9]*}} = vdeal(v{{[0-9]*}},v{{[0-9]*}},r{{[0-9]*}})
; CHECK: v{{[0-9]*}}:{{[0-9]*}} = vshuff(v{{[0-9]*}},v{{[0-9]*}},r{{[0-9]*}})
; CHECK: v{{[0-9]*}}:{{[0-9]*}} = vshuff(v{{[0-9]*}},v{{[0-9]*}},r{{[0-9]*}})
; CHECK: v{{[0-9]*}}:{{[0-9]*}} = vshuff(v{{[0-9]*}},v{{[0-9]*}},r{{[0-9]*}})
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})
; CHECK: v{{[0-9]*}}:{{[0-9]*}} = vswap(q{{[0-3]}},v{{[0-9]*}},v{{[0-9]*}})
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.b = vadd(v{{[0-9]*}}:{{[0-9]*}}.b,v{{[0-9]*}}:{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.b = vshuffoe(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.b = vsub(v{{[0-9]*}}:{{[0-9]*}}.b,v{{[0-9]*}}:{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vadd(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vadd(v{{[0-9]*}}:{{[0-9]*}}.h,v{{[0-9]*}}:{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vadd(v{{[0-9]*}}:{{[0-9]*}}.h,v{{[0-9]*}}:{{[0-9]*}}.h):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vdmpy(v{{[0-9]*}}:{{[0-9]*}}.ub,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h += vdmpy(v{{[0-9]*}}:{{[0-9]*}}.ub,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vlut16(v{{[0-9]*}}.b,v{{[0-9]*}}.h,r{{[0-9]*}})
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h |= vlut16(v{{[0-9]*}}.b,v{{[0-9]*}}.h,r{{[0-9]*}})
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vmpa(v{{[0-9]*}}:{{[0-9]*}}.ub,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vmpa(v{{[0-9]*}}:{{[0-9]*}}.ub,v{{[0-9]*}}:{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vmpa(v{{[0-9]*}}:{{[0-9]*}}.ub,v{{[0-9]*}}:{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h += vmpa(v{{[0-9]*}}:{{[0-9]*}}.ub,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vmpy(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vmpy(v{{[0-9]*}}.ub,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vmpy(v{{[0-9]*}}.ub,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h += vmpy(v{{[0-9]*}}.b,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h += vmpy(v{{[0-9]*}}.ub,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h += vmpy(v{{[0-9]*}}.ub,v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vshuffoe(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vsub(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vsub(v{{[0-9]*}}:{{[0-9]*}}.h,v{{[0-9]*}}:{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vsub(v{{[0-9]*}}:{{[0-9]*}}.h,v{{[0-9]*}}:{{[0-9]*}}.h):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vsxt(v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vtmpy(v{{[0-9]*}}:{{[0-9]*}}.b,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vtmpy(v{{[0-9]*}}:{{[0-9]*}}.ub,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h += vtmpy(v{{[0-9]*}}:{{[0-9]*}}.b,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h += vtmpy(v{{[0-9]*}}:{{[0-9]*}}.ub,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h = vunpack(v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.h |= vunpacko(v{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.ub = vadd(v{{[0-9]*}}:{{[0-9]*}}.ub,v{{[0-9]*}}:{{[0-9]*}}.ub):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.ub = vsub(v{{[0-9]*}}:{{[0-9]*}}.ub,v{{[0-9]*}}:{{[0-9]*}}.ub):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uh = vadd(v{{[0-9]*}}:{{[0-9]*}}.uh,v{{[0-9]*}}:{{[0-9]*}}.uh):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uh = vmpy(v{{[0-9]*}}.ub,r{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uh = vmpy(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uh += vmpy(v{{[0-9]*}}.ub,r{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uh += vmpy(v{{[0-9]*}}.ub,v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uh = vsub(v{{[0-9]*}}:{{[0-9]*}}.uh,v{{[0-9]*}}:{{[0-9]*}}.uh):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uh = vunpack(v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uh = vzxt(v{{[0-9]*}}.ub)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uw = vdsad(v{{[0-9]*}}:{{[0-9]*}}.uh,r{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uw += vdsad(v{{[0-9]*}}:{{[0-9]*}}.uh,r{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uw = vmpy(v{{[0-9]*}}.uh,r{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uw = vmpy(v{{[0-9]*}}.uh,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uw += vmpy(v{{[0-9]*}}.uh,r{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uw += vmpy(v{{[0-9]*}}.uh,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uw = vr{{[0-9]*}}mpy(v{{[0-9]*}}:{{[0-9]*}}.ub,r{{[0-9]*}}.ub,#0)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uw += vr{{[0-9]*}}mpy(v{{[0-9]*}}:{{[0-9]*}}.ub,r{{[0-9]*}}.ub,#0)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uw = vr{{[0-9]*}}sad(v{{[0-9]*}}:{{[0-9]*}}.ub,r{{[0-9]*}}.ub,#0)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uw += vr{{[0-9]*}}sad(v{{[0-9]*}}:{{[0-9]*}}.ub,r{{[0-9]*}}.ub,#0)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uw = vunpack(v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.uw = vzxt(v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vadd(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vadd(v{{[0-9]*}}.uh,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vadd(v{{[0-9]*}}:{{[0-9]*}}.w,v{{[0-9]*}}:{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vadd(v{{[0-9]*}}:{{[0-9]*}}.w,v{{[0-9]*}}:{{[0-9]*}}.w):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vdmpy(v{{[0-9]*}}:{{[0-9]*}}.h,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w += vdmpy(v{{[0-9]*}}:{{[0-9]*}}.h,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vmpa(v{{[0-9]*}}:{{[0-9]*}}.h,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w += vmpa(v{{[0-9]*}}:{{[0-9]*}}.h,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vmpy(v{{[0-9]*}}.h,r{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vmpy(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vmpy(v{{[0-9]*}}.h,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w += vmpy(v{{[0-9]*}}.h,r{{[0-9]*}}.h):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w += vmpy(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w += vmpy(v{{[0-9]*}}.h,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vr{{[0-9]*}}mpy(v{{[0-9]*}}:{{[0-9]*}}.ub,r{{[0-9]*}}.b,#0)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w += vr{{[0-9]*}}mpy(v{{[0-9]*}}:{{[0-9]*}}.ub,r{{[0-9]*}}.b,#0)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vsub(v{{[0-9]*}}.h,v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vsub(v{{[0-9]*}}.uh,v{{[0-9]*}}.uh)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vsub(v{{[0-9]*}}:{{[0-9]*}}.w,v{{[0-9]*}}:{{[0-9]*}}.w)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vsub(v{{[0-9]*}}:{{[0-9]*}}.w,v{{[0-9]*}}:{{[0-9]*}}.w):{{[0-9]*}}sat
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vsxt(v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vtmpy(v{{[0-9]*}}:{{[0-9]*}}.h,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w += vtmpy(v{{[0-9]*}}:{{[0-9]*}}.h,r{{[0-9]*}}.b)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w = vunpack(v{{[0-9]*}}.h)
; CHECK: v{{[0-9]*}}:{{[0-9]*}}.w |= vunpacko(v{{[0-9]*}}.h)
target datalayout = "e-m:e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a:0-n16:32"
target triple = "hexagon"

@K = global i64 0, align 8
@src = global i8 -1, align 1
@vecpreds = common global [15 x <16 x i32>] zeroinitializer, align 64
@Q6VecPredResult = common global <16 x i32> zeroinitializer, align 64
@vectors = common global [15 x <16 x i32>] zeroinitializer, align 64
@VectorResult = common global <16 x i32> zeroinitializer, align 64
@vector_pairs = common global [15 x <32 x i32>] zeroinitializer, align 128
@VectorPairResult = common global <32 x i32> zeroinitializer, align 128
@dst_addresses = common global [15 x i8] zeroinitializer, align 8
@ptr_addresses = common global [15 x i8*] zeroinitializer, align 8
@src_addresses = common global [15 x i8*] zeroinitializer, align 8
@dst = common global i8 0, align 1
@ptr = common global [32768 x i8] zeroinitializer, align 8

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %0 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %1 = bitcast <16 x i32> %0 to <512 x i1>
  %2 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 1), align 64
  %3 = bitcast <16 x i32> %2 to <512 x i1>
  %4 = call <512 x i1> @llvm.hexagon.V6.pred.and(<512 x i1> %1, <512 x i1> %3)
  %5 = bitcast <512 x i1> %4 to <16 x i32>
  store volatile <16 x i32> %5, <16 x i32>* @Q6VecPredResult, align 64
  %6 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %7 = bitcast <16 x i32> %6 to <512 x i1>
  %8 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 1), align 64
  %9 = bitcast <16 x i32> %8 to <512 x i1>
  %10 = call <512 x i1> @llvm.hexagon.V6.pred.and.n(<512 x i1> %7, <512 x i1> %9)
  %11 = bitcast <512 x i1> %10 to <16 x i32>
  store volatile <16 x i32> %11, <16 x i32>* @Q6VecPredResult, align 64
  %12 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %13 = bitcast <16 x i32> %12 to <512 x i1>
  %14 = call <512 x i1> @llvm.hexagon.V6.pred.not(<512 x i1> %13)
  %15 = bitcast <512 x i1> %14 to <16 x i32>
  store volatile <16 x i32> %15, <16 x i32>* @Q6VecPredResult, align 64
  %16 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %17 = bitcast <16 x i32> %16 to <512 x i1>
  %18 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 1), align 64
  %19 = bitcast <16 x i32> %18 to <512 x i1>
  %20 = call <512 x i1> @llvm.hexagon.V6.pred.or(<512 x i1> %17, <512 x i1> %19)
  %21 = bitcast <512 x i1> %20 to <16 x i32>
  store volatile <16 x i32> %21, <16 x i32>* @Q6VecPredResult, align 64
  %22 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %23 = bitcast <16 x i32> %22 to <512 x i1>
  %24 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 1), align 64
  %25 = bitcast <16 x i32> %24 to <512 x i1>
  %26 = call <512 x i1> @llvm.hexagon.V6.pred.or.n(<512 x i1> %23, <512 x i1> %25)
  %27 = bitcast <512 x i1> %26 to <16 x i32>
  store volatile <16 x i32> %27, <16 x i32>* @Q6VecPredResult, align 64
  %28 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %29 = call <512 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %28, i32 -1)
  %30 = bitcast <512 x i1> %29 to <16 x i32>
  store volatile <16 x i32> %30, <16 x i32>* @Q6VecPredResult, align 64
  %31 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %32 = bitcast <16 x i32> %31 to <512 x i1>
  %33 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %34 = call <512 x i1> @llvm.hexagon.V6.vandvrt.acc(<512 x i1> %32, <16 x i32> %33, i32 -1)
  %35 = bitcast <512 x i1> %34 to <16 x i32>
  store volatile <16 x i32> %35, <16 x i32>* @Q6VecPredResult, align 64
  %36 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %37 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %38 = call <512 x i1> @llvm.hexagon.V6.veqb(<16 x i32> %36, <16 x i32> %37)
  %39 = bitcast <512 x i1> %38 to <16 x i32>
  store volatile <16 x i32> %39, <16 x i32>* @Q6VecPredResult, align 64
  %40 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %41 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %42 = call <512 x i1> @llvm.hexagon.V6.veqh(<16 x i32> %40, <16 x i32> %41)
  %43 = bitcast <512 x i1> %42 to <16 x i32>
  store volatile <16 x i32> %43, <16 x i32>* @Q6VecPredResult, align 64
  %44 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %45 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %46 = call <512 x i1> @llvm.hexagon.V6.veqw(<16 x i32> %44, <16 x i32> %45)
  %47 = bitcast <512 x i1> %46 to <16 x i32>
  store volatile <16 x i32> %47, <16 x i32>* @Q6VecPredResult, align 64
  %48 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %49 = bitcast <16 x i32> %48 to <512 x i1>
  %50 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %51 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %52 = call <512 x i1> @llvm.hexagon.V6.veqb.and(<512 x i1> %49, <16 x i32> %50, <16 x i32> %51)
  %53 = bitcast <512 x i1> %52 to <16 x i32>
  store volatile <16 x i32> %53, <16 x i32>* @Q6VecPredResult, align 64
  %54 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %55 = bitcast <16 x i32> %54 to <512 x i1>
  %56 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %57 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %58 = call <512 x i1> @llvm.hexagon.V6.veqh.and(<512 x i1> %55, <16 x i32> %56, <16 x i32> %57)
  %59 = bitcast <512 x i1> %58 to <16 x i32>
  store volatile <16 x i32> %59, <16 x i32>* @Q6VecPredResult, align 64
  %60 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %61 = bitcast <16 x i32> %60 to <512 x i1>
  %62 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %63 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %64 = call <512 x i1> @llvm.hexagon.V6.veqw.and(<512 x i1> %61, <16 x i32> %62, <16 x i32> %63)
  %65 = bitcast <512 x i1> %64 to <16 x i32>
  store volatile <16 x i32> %65, <16 x i32>* @Q6VecPredResult, align 64
  %66 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %67 = bitcast <16 x i32> %66 to <512 x i1>
  %68 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %69 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %70 = call <512 x i1> @llvm.hexagon.V6.veqb.or(<512 x i1> %67, <16 x i32> %68, <16 x i32> %69)
  %71 = bitcast <512 x i1> %70 to <16 x i32>
  store volatile <16 x i32> %71, <16 x i32>* @Q6VecPredResult, align 64
  %72 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %73 = bitcast <16 x i32> %72 to <512 x i1>
  %74 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %75 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %76 = call <512 x i1> @llvm.hexagon.V6.veqh.or(<512 x i1> %73, <16 x i32> %74, <16 x i32> %75)
  %77 = bitcast <512 x i1> %76 to <16 x i32>
  store volatile <16 x i32> %77, <16 x i32>* @Q6VecPredResult, align 64
  %78 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %79 = bitcast <16 x i32> %78 to <512 x i1>
  %80 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %81 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %82 = call <512 x i1> @llvm.hexagon.V6.veqw.or(<512 x i1> %79, <16 x i32> %80, <16 x i32> %81)
  %83 = bitcast <512 x i1> %82 to <16 x i32>
  store volatile <16 x i32> %83, <16 x i32>* @Q6VecPredResult, align 64
  %84 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %85 = bitcast <16 x i32> %84 to <512 x i1>
  %86 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %87 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %88 = call <512 x i1> @llvm.hexagon.V6.veqb.xor(<512 x i1> %85, <16 x i32> %86, <16 x i32> %87)
  %89 = bitcast <512 x i1> %88 to <16 x i32>
  store volatile <16 x i32> %89, <16 x i32>* @Q6VecPredResult, align 64
  %90 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %91 = bitcast <16 x i32> %90 to <512 x i1>
  %92 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %93 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %94 = call <512 x i1> @llvm.hexagon.V6.veqh.xor(<512 x i1> %91, <16 x i32> %92, <16 x i32> %93)
  %95 = bitcast <512 x i1> %94 to <16 x i32>
  store volatile <16 x i32> %95, <16 x i32>* @Q6VecPredResult, align 64
  %96 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %97 = bitcast <16 x i32> %96 to <512 x i1>
  %98 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %99 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %100 = call <512 x i1> @llvm.hexagon.V6.veqw.xor(<512 x i1> %97, <16 x i32> %98, <16 x i32> %99)
  %101 = bitcast <512 x i1> %100 to <16 x i32>
  store volatile <16 x i32> %101, <16 x i32>* @Q6VecPredResult, align 64
  %102 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %103 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %104 = call <512 x i1> @llvm.hexagon.V6.vgtb(<16 x i32> %102, <16 x i32> %103)
  %105 = bitcast <512 x i1> %104 to <16 x i32>
  store volatile <16 x i32> %105, <16 x i32>* @Q6VecPredResult, align 64
  %106 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %107 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %108 = call <512 x i1> @llvm.hexagon.V6.vgth(<16 x i32> %106, <16 x i32> %107)
  %109 = bitcast <512 x i1> %108 to <16 x i32>
  store volatile <16 x i32> %109, <16 x i32>* @Q6VecPredResult, align 64
  %110 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %111 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %112 = call <512 x i1> @llvm.hexagon.V6.vgtub(<16 x i32> %110, <16 x i32> %111)
  %113 = bitcast <512 x i1> %112 to <16 x i32>
  store volatile <16 x i32> %113, <16 x i32>* @Q6VecPredResult, align 64
  %114 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %115 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %116 = call <512 x i1> @llvm.hexagon.V6.vgtuh(<16 x i32> %114, <16 x i32> %115)
  %117 = bitcast <512 x i1> %116 to <16 x i32>
  store volatile <16 x i32> %117, <16 x i32>* @Q6VecPredResult, align 64
  %118 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %119 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %120 = call <512 x i1> @llvm.hexagon.V6.vgtuw(<16 x i32> %118, <16 x i32> %119)
  %121 = bitcast <512 x i1> %120 to <16 x i32>
  store volatile <16 x i32> %121, <16 x i32>* @Q6VecPredResult, align 64
  %122 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %123 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %124 = call <512 x i1> @llvm.hexagon.V6.vgtw(<16 x i32> %122, <16 x i32> %123)
  %125 = bitcast <512 x i1> %124 to <16 x i32>
  store volatile <16 x i32> %125, <16 x i32>* @Q6VecPredResult, align 64
  %126 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %127 = bitcast <16 x i32> %126 to <512 x i1>
  %128 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %129 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %130 = call <512 x i1> @llvm.hexagon.V6.vgtb.and(<512 x i1> %127, <16 x i32> %128, <16 x i32> %129)
  %131 = bitcast <512 x i1> %130 to <16 x i32>
  store volatile <16 x i32> %131, <16 x i32>* @Q6VecPredResult, align 64
  %132 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %133 = bitcast <16 x i32> %132 to <512 x i1>
  %134 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %135 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %136 = call <512 x i1> @llvm.hexagon.V6.vgth.and(<512 x i1> %133, <16 x i32> %134, <16 x i32> %135)
  %137 = bitcast <512 x i1> %136 to <16 x i32>
  store volatile <16 x i32> %137, <16 x i32>* @Q6VecPredResult, align 64
  %138 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %139 = bitcast <16 x i32> %138 to <512 x i1>
  %140 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %141 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %142 = call <512 x i1> @llvm.hexagon.V6.vgtub.and(<512 x i1> %139, <16 x i32> %140, <16 x i32> %141)
  %143 = bitcast <512 x i1> %142 to <16 x i32>
  store volatile <16 x i32> %143, <16 x i32>* @Q6VecPredResult, align 64
  %144 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %145 = bitcast <16 x i32> %144 to <512 x i1>
  %146 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %147 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %148 = call <512 x i1> @llvm.hexagon.V6.vgtuh.and(<512 x i1> %145, <16 x i32> %146, <16 x i32> %147)
  %149 = bitcast <512 x i1> %148 to <16 x i32>
  store volatile <16 x i32> %149, <16 x i32>* @Q6VecPredResult, align 64
  %150 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %151 = bitcast <16 x i32> %150 to <512 x i1>
  %152 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %153 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %154 = call <512 x i1> @llvm.hexagon.V6.vgtuw.and(<512 x i1> %151, <16 x i32> %152, <16 x i32> %153)
  %155 = bitcast <512 x i1> %154 to <16 x i32>
  store volatile <16 x i32> %155, <16 x i32>* @Q6VecPredResult, align 64
  %156 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %157 = bitcast <16 x i32> %156 to <512 x i1>
  %158 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %159 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %160 = call <512 x i1> @llvm.hexagon.V6.vgtw.and(<512 x i1> %157, <16 x i32> %158, <16 x i32> %159)
  %161 = bitcast <512 x i1> %160 to <16 x i32>
  store volatile <16 x i32> %161, <16 x i32>* @Q6VecPredResult, align 64
  %162 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %163 = bitcast <16 x i32> %162 to <512 x i1>
  %164 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %165 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %166 = call <512 x i1> @llvm.hexagon.V6.vgtb.or(<512 x i1> %163, <16 x i32> %164, <16 x i32> %165)
  %167 = bitcast <512 x i1> %166 to <16 x i32>
  store volatile <16 x i32> %167, <16 x i32>* @Q6VecPredResult, align 64
  %168 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %169 = bitcast <16 x i32> %168 to <512 x i1>
  %170 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %171 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %172 = call <512 x i1> @llvm.hexagon.V6.vgth.or(<512 x i1> %169, <16 x i32> %170, <16 x i32> %171)
  %173 = bitcast <512 x i1> %172 to <16 x i32>
  store volatile <16 x i32> %173, <16 x i32>* @Q6VecPredResult, align 64
  %174 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %175 = bitcast <16 x i32> %174 to <512 x i1>
  %176 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %177 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %178 = call <512 x i1> @llvm.hexagon.V6.vgtub.or(<512 x i1> %175, <16 x i32> %176, <16 x i32> %177)
  %179 = bitcast <512 x i1> %178 to <16 x i32>
  store volatile <16 x i32> %179, <16 x i32>* @Q6VecPredResult, align 64
  %180 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %181 = bitcast <16 x i32> %180 to <512 x i1>
  %182 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %183 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %184 = call <512 x i1> @llvm.hexagon.V6.vgtuh.or(<512 x i1> %181, <16 x i32> %182, <16 x i32> %183)
  %185 = bitcast <512 x i1> %184 to <16 x i32>
  store volatile <16 x i32> %185, <16 x i32>* @Q6VecPredResult, align 64
  %186 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %187 = bitcast <16 x i32> %186 to <512 x i1>
  %188 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %189 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %190 = call <512 x i1> @llvm.hexagon.V6.vgtuw.or(<512 x i1> %187, <16 x i32> %188, <16 x i32> %189)
  %191 = bitcast <512 x i1> %190 to <16 x i32>
  store volatile <16 x i32> %191, <16 x i32>* @Q6VecPredResult, align 64
  %192 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %193 = bitcast <16 x i32> %192 to <512 x i1>
  %194 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %195 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %196 = call <512 x i1> @llvm.hexagon.V6.vgtw.or(<512 x i1> %193, <16 x i32> %194, <16 x i32> %195)
  %197 = bitcast <512 x i1> %196 to <16 x i32>
  store volatile <16 x i32> %197, <16 x i32>* @Q6VecPredResult, align 64
  %198 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %199 = bitcast <16 x i32> %198 to <512 x i1>
  %200 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %201 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %202 = call <512 x i1> @llvm.hexagon.V6.vgtb.xor(<512 x i1> %199, <16 x i32> %200, <16 x i32> %201)
  %203 = bitcast <512 x i1> %202 to <16 x i32>
  store volatile <16 x i32> %203, <16 x i32>* @Q6VecPredResult, align 64
  %204 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %205 = bitcast <16 x i32> %204 to <512 x i1>
  %206 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %207 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %208 = call <512 x i1> @llvm.hexagon.V6.vgth.xor(<512 x i1> %205, <16 x i32> %206, <16 x i32> %207)
  %209 = bitcast <512 x i1> %208 to <16 x i32>
  store volatile <16 x i32> %209, <16 x i32>* @Q6VecPredResult, align 64
  %210 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %211 = bitcast <16 x i32> %210 to <512 x i1>
  %212 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %213 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %214 = call <512 x i1> @llvm.hexagon.V6.vgtub.xor(<512 x i1> %211, <16 x i32> %212, <16 x i32> %213)
  %215 = bitcast <512 x i1> %214 to <16 x i32>
  store volatile <16 x i32> %215, <16 x i32>* @Q6VecPredResult, align 64
  %216 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %217 = bitcast <16 x i32> %216 to <512 x i1>
  %218 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %219 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %220 = call <512 x i1> @llvm.hexagon.V6.vgtuh.xor(<512 x i1> %217, <16 x i32> %218, <16 x i32> %219)
  %221 = bitcast <512 x i1> %220 to <16 x i32>
  store volatile <16 x i32> %221, <16 x i32>* @Q6VecPredResult, align 64
  %222 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %223 = bitcast <16 x i32> %222 to <512 x i1>
  %224 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %225 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %226 = call <512 x i1> @llvm.hexagon.V6.vgtuw.xor(<512 x i1> %223, <16 x i32> %224, <16 x i32> %225)
  %227 = bitcast <512 x i1> %226 to <16 x i32>
  store volatile <16 x i32> %227, <16 x i32>* @Q6VecPredResult, align 64
  %228 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %229 = bitcast <16 x i32> %228 to <512 x i1>
  %230 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %231 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %232 = call <512 x i1> @llvm.hexagon.V6.vgtw.xor(<512 x i1> %229, <16 x i32> %230, <16 x i32> %231)
  %233 = bitcast <512 x i1> %232 to <16 x i32>
  store volatile <16 x i32> %233, <16 x i32>* @Q6VecPredResult, align 64
  %234 = call <512 x i1> @llvm.hexagon.V6.pred.scalar2(i32 1)
  %235 = bitcast <512 x i1> %234 to <16 x i32>
  store volatile <16 x i32> %235, <16 x i32>* @Q6VecPredResult, align 64
  %236 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %237 = bitcast <16 x i32> %236 to <512 x i1>
  %238 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 1), align 64
  %239 = bitcast <16 x i32> %238 to <512 x i1>
  %240 = call <512 x i1> @llvm.hexagon.V6.pred.xor(<512 x i1> %237, <512 x i1> %239)
  %241 = bitcast <512 x i1> %240 to <16 x i32>
  store volatile <16 x i32> %241, <16 x i32>* @Q6VecPredResult, align 64
  %242 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %243 = call <16 x i32> @llvm.hexagon.V6.vassign(<16 x i32> %242)
  store volatile <16 x i32> %243, <16 x i32>* @VectorResult, align 64
  %244 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %245 = call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %244)
  store volatile <16 x i32> %245, <16 x i32>* @VectorResult, align 64
  %246 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %247 = call <16 x i32> @llvm.hexagon.V6.lo(<32 x i32> %246)
  store volatile <16 x i32> %247, <16 x i32>* @VectorResult, align 64
  %248 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %249 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %250 = call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %248, <16 x i32> %249, i32 1)
  store volatile <16 x i32> %250, <16 x i32>* @VectorResult, align 64
  %251 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %252 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %253 = call <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32> %251, <16 x i32> %252, i32 -1)
  store volatile <16 x i32> %253, <16 x i32>* @VectorResult, align 64
  %254 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %255 = bitcast <16 x i32> %254 to <512 x i1>
  %256 = call <16 x i32> @llvm.hexagon.V6.vandqrt(<512 x i1> %255, i32 -1)
  store volatile <16 x i32> %256, <16 x i32>* @VectorResult, align 64
  %257 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %258 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %259 = call <16 x i32> @llvm.hexagon.V6.vand(<16 x i32> %257, <16 x i32> %258)
  store volatile <16 x i32> %259, <16 x i32>* @VectorResult, align 64
  %260 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %261 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %262 = bitcast <16 x i32> %261 to <512 x i1>
  %263 = call <16 x i32> @llvm.hexagon.V6.vandqrt.acc(<16 x i32> %260, <512 x i1> %262, i32 -1)
  store volatile <16 x i32> %263, <16 x i32>* @VectorResult, align 64
  %264 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %265 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %266 = call <16 x i32> @llvm.hexagon.V6.vdelta(<16 x i32> %264, <16 x i32> %265)
  store volatile <16 x i32> %266, <16 x i32>* @VectorResult, align 64
  %267 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %268 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %269 = call <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32> %267, <16 x i32> %268, i32 1)
  store volatile <16 x i32> %269, <16 x i32>* @VectorResult, align 64
  %270 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %271 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %272 = call <16 x i32> @llvm.hexagon.V6.vlalignb(<16 x i32> %270, <16 x i32> %271, i32 -1)
  store volatile <16 x i32> %272, <16 x i32>* @VectorResult, align 64
  %273 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %274 = bitcast <16 x i32> %273 to <512 x i1>
  %275 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %276 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %277 = call <16 x i32> @llvm.hexagon.V6.vmux(<512 x i1> %274, <16 x i32> %275, <16 x i32> %276)
  store volatile <16 x i32> %277, <16 x i32>* @VectorResult, align 64
  %278 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %279 = call <16 x i32> @llvm.hexagon.V6.vnot(<16 x i32> %278)
  store volatile <16 x i32> %279, <16 x i32>* @VectorResult, align 64
  %280 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %281 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %282 = call <16 x i32> @llvm.hexagon.V6.vor(<16 x i32> %280, <16 x i32> %281)
  store volatile <16 x i32> %282, <16 x i32>* @VectorResult, align 64
  %283 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %284 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %285 = call <16 x i32> @llvm.hexagon.V6.vrdelta(<16 x i32> %283, <16 x i32> %284)
  store volatile <16 x i32> %285, <16 x i32>* @VectorResult, align 64
  %286 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %287 = call <16 x i32> @llvm.hexagon.V6.vror(<16 x i32> %286, i32 -1)
  store volatile <16 x i32> %287, <16 x i32>* @VectorResult, align 64
  %288 = call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 -1)
  store volatile <16 x i32> %288, <16 x i32>* @VectorResult, align 64
  %289 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %290 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %291 = call <16 x i32> @llvm.hexagon.V6.vxor(<16 x i32> %289, <16 x i32> %290)
  store volatile <16 x i32> %291, <16 x i32>* @VectorResult, align 64
  %292 = call <16 x i32> @llvm.hexagon.V6.vd0()
  store volatile <16 x i32> %292, <16 x i32>* @VectorResult, align 64
  %293 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %294 = bitcast <16 x i32> %293 to <512 x i1>
  %295 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %296 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %297 = call <16 x i32> @llvm.hexagon.V6.vaddbq(<512 x i1> %294, <16 x i32> %295, <16 x i32> %296)
  store volatile <16 x i32> %297, <16 x i32>* @VectorResult, align 64
  %298 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %299 = bitcast <16 x i32> %298 to <512 x i1>
  %300 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %301 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %302 = call <16 x i32> @llvm.hexagon.V6.vaddbnq(<512 x i1> %299, <16 x i32> %300, <16 x i32> %301)
  store volatile <16 x i32> %302, <16 x i32>* @VectorResult, align 64
  %303 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %304 = bitcast <16 x i32> %303 to <512 x i1>
  %305 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %306 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %307 = call <16 x i32> @llvm.hexagon.V6.vsubbq(<512 x i1> %304, <16 x i32> %305, <16 x i32> %306)
  store volatile <16 x i32> %307, <16 x i32>* @VectorResult, align 64
  %308 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %309 = bitcast <16 x i32> %308 to <512 x i1>
  %310 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %311 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %312 = call <16 x i32> @llvm.hexagon.V6.vsubbnq(<512 x i1> %309, <16 x i32> %310, <16 x i32> %311)
  store volatile <16 x i32> %312, <16 x i32>* @VectorResult, align 64
  %313 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %314 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %315 = call <16 x i32> @llvm.hexagon.V6.vaddb(<16 x i32> %313, <16 x i32> %314)
  store volatile <16 x i32> %315, <16 x i32>* @VectorResult, align 64
  %316 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %317 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %318 = call <16 x i32> @llvm.hexagon.V6.vasrhbrndsat(<16 x i32> %316, <16 x i32> %317, i32 -1)
  store volatile <16 x i32> %318, <16 x i32>* @VectorResult, align 64
  %319 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %320 = call <16 x i32> @llvm.hexagon.V6.vdealb(<16 x i32> %319)
  store volatile <16 x i32> %320, <16 x i32>* @VectorResult, align 64
  %321 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %322 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %323 = call <16 x i32> @llvm.hexagon.V6.vdealb4w(<16 x i32> %321, <16 x i32> %322)
  store volatile <16 x i32> %323, <16 x i32>* @VectorResult, align 64
  %324 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %325 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %326 = call <16 x i32> @llvm.hexagon.V6.vlutvvb(<16 x i32> %324, <16 x i32> %325, i32 -1)
  store volatile <16 x i32> %326, <16 x i32>* @VectorResult, align 64
  %327 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %328 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %329 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 2), align 64
  %330 = call <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32> %327, <16 x i32> %328, <16 x i32> %329, i32 -1)
  store volatile <16 x i32> %330, <16 x i32>* @VectorResult, align 64
  %331 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %332 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %333 = call <16 x i32> @llvm.hexagon.V6.vnavgub(<16 x i32> %331, <16 x i32> %332)
  store volatile <16 x i32> %333, <16 x i32>* @VectorResult, align 64
  %334 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %335 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %336 = call <16 x i32> @llvm.hexagon.V6.vpackhb.sat(<16 x i32> %334, <16 x i32> %335)
  store volatile <16 x i32> %336, <16 x i32>* @VectorResult, align 64
  %337 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %338 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %339 = call <16 x i32> @llvm.hexagon.V6.vpackeb(<16 x i32> %337, <16 x i32> %338)
  store volatile <16 x i32> %339, <16 x i32>* @VectorResult, align 64
  %340 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %341 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %342 = call <16 x i32> @llvm.hexagon.V6.vpackob(<16 x i32> %340, <16 x i32> %341)
  store volatile <16 x i32> %342, <16 x i32>* @VectorResult, align 64
  %343 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %344 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %345 = call <16 x i32> @llvm.hexagon.V6.vroundhb(<16 x i32> %343, <16 x i32> %344)
  store volatile <16 x i32> %345, <16 x i32>* @VectorResult, align 64
  %346 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %347 = call <16 x i32> @llvm.hexagon.V6.vshuffb(<16 x i32> %346)
  store volatile <16 x i32> %347, <16 x i32>* @VectorResult, align 64
  %348 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %349 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %350 = call <16 x i32> @llvm.hexagon.V6.vshuffeb(<16 x i32> %348, <16 x i32> %349)
  store volatile <16 x i32> %350, <16 x i32>* @VectorResult, align 64
  %351 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %352 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %353 = call <16 x i32> @llvm.hexagon.V6.vshuffob(<16 x i32> %351, <16 x i32> %352)
  store volatile <16 x i32> %353, <16 x i32>* @VectorResult, align 64
  %354 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %355 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %356 = call <16 x i32> @llvm.hexagon.V6.vsubb(<16 x i32> %354, <16 x i32> %355)
  store volatile <16 x i32> %356, <16 x i32>* @VectorResult, align 64
  %357 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %358 = bitcast <16 x i32> %357 to <512 x i1>
  %359 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %360 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %361 = call <16 x i32> @llvm.hexagon.V6.vaddhq(<512 x i1> %358, <16 x i32> %359, <16 x i32> %360)
  store volatile <16 x i32> %361, <16 x i32>* @VectorResult, align 64
  %362 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %363 = bitcast <16 x i32> %362 to <512 x i1>
  %364 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %365 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %366 = call <16 x i32> @llvm.hexagon.V6.vaddhnq(<512 x i1> %363, <16 x i32> %364, <16 x i32> %365)
  store volatile <16 x i32> %366, <16 x i32>* @VectorResult, align 64
  %367 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %368 = bitcast <16 x i32> %367 to <512 x i1>
  %369 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %370 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %371 = call <16 x i32> @llvm.hexagon.V6.vsubhq(<512 x i1> %368, <16 x i32> %369, <16 x i32> %370)
  store volatile <16 x i32> %371, <16 x i32>* @VectorResult, align 64
  %372 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %373 = bitcast <16 x i32> %372 to <512 x i1>
  %374 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %375 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %376 = call <16 x i32> @llvm.hexagon.V6.vsubhnq(<512 x i1> %373, <16 x i32> %374, <16 x i32> %375)
  store volatile <16 x i32> %376, <16 x i32>* @VectorResult, align 64
  %377 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %378 = call <16 x i32> @llvm.hexagon.V6.vabsh(<16 x i32> %377)
  store volatile <16 x i32> %378, <16 x i32>* @VectorResult, align 64
  %379 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %380 = call <16 x i32> @llvm.hexagon.V6.vabsh.sat(<16 x i32> %379)
  store volatile <16 x i32> %380, <16 x i32>* @VectorResult, align 64
  %381 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %382 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %383 = call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %381, <16 x i32> %382)
  store volatile <16 x i32> %383, <16 x i32>* @VectorResult, align 64
  %384 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %385 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %386 = call <16 x i32> @llvm.hexagon.V6.vaddhsat(<16 x i32> %384, <16 x i32> %385)
  store volatile <16 x i32> %386, <16 x i32>* @VectorResult, align 64
  %387 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %388 = call <16 x i32> @llvm.hexagon.V6.vaslh(<16 x i32> %387, i32 -1)
  store volatile <16 x i32> %388, <16 x i32>* @VectorResult, align 64
  %389 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %390 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %391 = call <16 x i32> @llvm.hexagon.V6.vaslhv(<16 x i32> %389, <16 x i32> %390)
  store volatile <16 x i32> %391, <16 x i32>* @VectorResult, align 64
  %392 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %393 = call <16 x i32> @llvm.hexagon.V6.vasrh(<16 x i32> %392, i32 -1)
  store volatile <16 x i32> %393, <16 x i32>* @VectorResult, align 64
  %394 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %395 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %396 = call <16 x i32> @llvm.hexagon.V6.vasrhv(<16 x i32> %394, <16 x i32> %395)
  store volatile <16 x i32> %396, <16 x i32>* @VectorResult, align 64
  %397 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %398 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %399 = call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %397, <16 x i32> %398, i32 -1)
  store volatile <16 x i32> %399, <16 x i32>* @VectorResult, align 64
  %400 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %401 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %402 = call <16 x i32> @llvm.hexagon.V6.vasrwhrndsat(<16 x i32> %400, <16 x i32> %401, i32 -1)
  store volatile <16 x i32> %402, <16 x i32>* @VectorResult, align 64
  %403 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %404 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %405 = call <16 x i32> @llvm.hexagon.V6.vasrwhsat(<16 x i32> %403, <16 x i32> %404, i32 -1)
  store volatile <16 x i32> %405, <16 x i32>* @VectorResult, align 64
  %406 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %407 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %408 = call <16 x i32> @llvm.hexagon.V6.vavgh(<16 x i32> %406, <16 x i32> %407)
  store volatile <16 x i32> %408, <16 x i32>* @VectorResult, align 64
  %409 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %410 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %411 = call <16 x i32> @llvm.hexagon.V6.vavghrnd(<16 x i32> %409, <16 x i32> %410)
  store volatile <16 x i32> %411, <16 x i32>* @VectorResult, align 64
  %412 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %413 = call <16 x i32> @llvm.hexagon.V6.vdealh(<16 x i32> %412)
  store volatile <16 x i32> %413, <16 x i32>* @VectorResult, align 64
  %414 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %415 = call <16 x i32> @llvm.hexagon.V6.vdmpybus(<16 x i32> %414, i32 -1)
  store volatile <16 x i32> %415, <16 x i32>* @VectorResult, align 64
  %416 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %417 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %418 = call <16 x i32> @llvm.hexagon.V6.vdmpybus.acc(<16 x i32> %416, <16 x i32> %417, i32 -1)
  store volatile <16 x i32> %418, <16 x i32>* @VectorResult, align 64
  %419 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %420 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %421 = call <16 x i32> @llvm.hexagon.V6.vlsrhv(<16 x i32> %419, <16 x i32> %420)
  store volatile <16 x i32> %421, <16 x i32>* @VectorResult, align 64
  %422 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %423 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %424 = call <16 x i32> @llvm.hexagon.V6.vmaxh(<16 x i32> %422, <16 x i32> %423)
  store volatile <16 x i32> %424, <16 x i32>* @VectorResult, align 64
  %425 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %426 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %427 = call <16 x i32> @llvm.hexagon.V6.vminh(<16 x i32> %425, <16 x i32> %426)
  store volatile <16 x i32> %427, <16 x i32>* @VectorResult, align 64
  %428 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %429 = call <16 x i32> @llvm.hexagon.V6.vmpyhsrs(<16 x i32> %428, i32 -1)
  store volatile <16 x i32> %429, <16 x i32>* @VectorResult, align 64
  %430 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %431 = call <16 x i32> @llvm.hexagon.V6.vmpyhss(<16 x i32> %430, i32 -1)
  store volatile <16 x i32> %431, <16 x i32>* @VectorResult, align 64
  %432 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %433 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %434 = call <16 x i32> @llvm.hexagon.V6.vmpyhvsrs(<16 x i32> %432, <16 x i32> %433)
  store volatile <16 x i32> %434, <16 x i32>* @VectorResult, align 64
  %435 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %436 = call <16 x i32> @llvm.hexagon.V6.vmpyihb(<16 x i32> %435, i32 -1)
  store volatile <16 x i32> %436, <16 x i32>* @VectorResult, align 64
  %437 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %438 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %439 = call <16 x i32> @llvm.hexagon.V6.vmpyih(<16 x i32> %437, <16 x i32> %438)
  store volatile <16 x i32> %439, <16 x i32>* @VectorResult, align 64
  %440 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %441 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %442 = call <16 x i32> @llvm.hexagon.V6.vmpyihb.acc(<16 x i32> %440, <16 x i32> %441, i32 -1)
  store volatile <16 x i32> %442, <16 x i32>* @VectorResult, align 64
  %443 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %444 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %445 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 2), align 64
  %446 = call <16 x i32> @llvm.hexagon.V6.vmpyih.acc(<16 x i32> %443, <16 x i32> %444, <16 x i32> %445)
  store volatile <16 x i32> %446, <16 x i32>* @VectorResult, align 64
  %447 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %448 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %449 = call <16 x i32> @llvm.hexagon.V6.vnavgh(<16 x i32> %447, <16 x i32> %448)
  store volatile <16 x i32> %449, <16 x i32>* @VectorResult, align 64
  %450 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %451 = call <16 x i32> @llvm.hexagon.V6.vnormamth(<16 x i32> %450)
  store volatile <16 x i32> %451, <16 x i32>* @VectorResult, align 64
  %452 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %453 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %454 = call <16 x i32> @llvm.hexagon.V6.vpackwh.sat(<16 x i32> %452, <16 x i32> %453)
  store volatile <16 x i32> %454, <16 x i32>* @VectorResult, align 64
  %455 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %456 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %457 = call <16 x i32> @llvm.hexagon.V6.vpackeh(<16 x i32> %455, <16 x i32> %456)
  store volatile <16 x i32> %457, <16 x i32>* @VectorResult, align 64
  %458 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %459 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %460 = call <16 x i32> @llvm.hexagon.V6.vpackoh(<16 x i32> %458, <16 x i32> %459)
  store volatile <16 x i32> %460, <16 x i32>* @VectorResult, align 64
  %461 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %462 = call <16 x i32> @llvm.hexagon.V6.vpopcounth(<16 x i32> %461)
  store volatile <16 x i32> %462, <16 x i32>* @VectorResult, align 64
  %463 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %464 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %465 = call <16 x i32> @llvm.hexagon.V6.vroundwh(<16 x i32> %463, <16 x i32> %464)
  store volatile <16 x i32> %465, <16 x i32>* @VectorResult, align 64
  %466 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %467 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %468 = call <16 x i32> @llvm.hexagon.V6.vsatwh(<16 x i32> %466, <16 x i32> %467)
  store volatile <16 x i32> %468, <16 x i32>* @VectorResult, align 64
  %469 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %470 = call <16 x i32> @llvm.hexagon.V6.vshuffh(<16 x i32> %469)
  store volatile <16 x i32> %470, <16 x i32>* @VectorResult, align 64
  %471 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %472 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %473 = call <16 x i32> @llvm.hexagon.V6.vshufeh(<16 x i32> %471, <16 x i32> %472)
  store volatile <16 x i32> %473, <16 x i32>* @VectorResult, align 64
  %474 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %475 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %476 = call <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32> %474, <16 x i32> %475)
  store volatile <16 x i32> %476, <16 x i32>* @VectorResult, align 64
  %477 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %478 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %479 = call <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32> %477, <16 x i32> %478)
  store volatile <16 x i32> %479, <16 x i32>* @VectorResult, align 64
  %480 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %481 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %482 = call <16 x i32> @llvm.hexagon.V6.vsubhsat(<16 x i32> %480, <16 x i32> %481)
  store volatile <16 x i32> %482, <16 x i32>* @VectorResult, align 64
  %483 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %484 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %485 = call <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32> %483, <16 x i32> %484)
  store volatile <16 x i32> %485, <16 x i32>* @VectorResult, align 64
  %486 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %487 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %488 = call <16 x i32> @llvm.hexagon.V6.vaddubsat(<16 x i32> %486, <16 x i32> %487)
  store volatile <16 x i32> %488, <16 x i32>* @VectorResult, align 64
  %489 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %490 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %491 = call <16 x i32> @llvm.hexagon.V6.vasrhubrndsat(<16 x i32> %489, <16 x i32> %490, i32 -1)
  store volatile <16 x i32> %491, <16 x i32>* @VectorResult, align 64
  %492 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %493 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %494 = call <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32> %492, <16 x i32> %493, i32 -1)
  store volatile <16 x i32> %494, <16 x i32>* @VectorResult, align 64
  %495 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %496 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %497 = call <16 x i32> @llvm.hexagon.V6.vavgub(<16 x i32> %495, <16 x i32> %496)
  store volatile <16 x i32> %497, <16 x i32>* @VectorResult, align 64
  %498 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %499 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %500 = call <16 x i32> @llvm.hexagon.V6.vavgubrnd(<16 x i32> %498, <16 x i32> %499)
  store volatile <16 x i32> %500, <16 x i32>* @VectorResult, align 64
  %501 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %502 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %503 = call <16 x i32> @llvm.hexagon.V6.vmaxub(<16 x i32> %501, <16 x i32> %502)
  store volatile <16 x i32> %503, <16 x i32>* @VectorResult, align 64
  %504 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %505 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %506 = call <16 x i32> @llvm.hexagon.V6.vminub(<16 x i32> %504, <16 x i32> %505)
  store volatile <16 x i32> %506, <16 x i32>* @VectorResult, align 64
  %507 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %508 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %509 = call <16 x i32> @llvm.hexagon.V6.vpackhub.sat(<16 x i32> %507, <16 x i32> %508)
  store volatile <16 x i32> %509, <16 x i32>* @VectorResult, align 64
  %510 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %511 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %512 = call <16 x i32> @llvm.hexagon.V6.vroundhub(<16 x i32> %510, <16 x i32> %511)
  store volatile <16 x i32> %512, <16 x i32>* @VectorResult, align 64
  %513 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %514 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %515 = call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %513, <16 x i32> %514)
  store volatile <16 x i32> %515, <16 x i32>* @VectorResult, align 64
  %516 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %517 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %518 = call <16 x i32> @llvm.hexagon.V6.vsububsat(<16 x i32> %516, <16 x i32> %517)
  store volatile <16 x i32> %518, <16 x i32>* @VectorResult, align 64
  %519 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %520 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %521 = call <16 x i32> @llvm.hexagon.V6.vabsdiffh(<16 x i32> %519, <16 x i32> %520)
  store volatile <16 x i32> %521, <16 x i32>* @VectorResult, align 64
  %522 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %523 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %524 = call <16 x i32> @llvm.hexagon.V6.vabsdiffuh(<16 x i32> %522, <16 x i32> %523)
  store volatile <16 x i32> %524, <16 x i32>* @VectorResult, align 64
  %525 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %526 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %527 = call <16 x i32> @llvm.hexagon.V6.vadduhsat(<16 x i32> %525, <16 x i32> %526)
  store volatile <16 x i32> %527, <16 x i32>* @VectorResult, align 64
  %528 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %529 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %530 = call <16 x i32> @llvm.hexagon.V6.vasrwuhsat(<16 x i32> %528, <16 x i32> %529, i32 -1)
  store volatile <16 x i32> %530, <16 x i32>* @VectorResult, align 64
  %531 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %532 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %533 = call <16 x i32> @llvm.hexagon.V6.vavguh(<16 x i32> %531, <16 x i32> %532)
  store volatile <16 x i32> %533, <16 x i32>* @VectorResult, align 64
  %534 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %535 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %536 = call <16 x i32> @llvm.hexagon.V6.vavguhrnd(<16 x i32> %534, <16 x i32> %535)
  store volatile <16 x i32> %536, <16 x i32>* @VectorResult, align 64
  %537 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %538 = call <16 x i32> @llvm.hexagon.V6.vcl0h(<16 x i32> %537)
  store volatile <16 x i32> %538, <16 x i32>* @VectorResult, align 64
  %539 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %540 = call <16 x i32> @llvm.hexagon.V6.vlsrh(<16 x i32> %539, i32 -1)
  store volatile <16 x i32> %540, <16 x i32>* @VectorResult, align 64
  %541 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %542 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %543 = call <16 x i32> @llvm.hexagon.V6.vmaxuh(<16 x i32> %541, <16 x i32> %542)
  store volatile <16 x i32> %543, <16 x i32>* @VectorResult, align 64
  %544 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %545 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %546 = call <16 x i32> @llvm.hexagon.V6.vminuh(<16 x i32> %544, <16 x i32> %545)
  store volatile <16 x i32> %546, <16 x i32>* @VectorResult, align 64
  %547 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %548 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %549 = call <16 x i32> @llvm.hexagon.V6.vpackwuh.sat(<16 x i32> %547, <16 x i32> %548)
  store volatile <16 x i32> %549, <16 x i32>* @VectorResult, align 64
  %550 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %551 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %552 = call <16 x i32> @llvm.hexagon.V6.vroundwuh(<16 x i32> %550, <16 x i32> %551)
  store volatile <16 x i32> %552, <16 x i32>* @VectorResult, align 64
  %553 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %554 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %555 = call <16 x i32> @llvm.hexagon.V6.vsubuhsat(<16 x i32> %553, <16 x i32> %554)
  store volatile <16 x i32> %555, <16 x i32>* @VectorResult, align 64
  %556 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %557 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %558 = call <16 x i32> @llvm.hexagon.V6.vabsdiffw(<16 x i32> %556, <16 x i32> %557)
  store volatile <16 x i32> %558, <16 x i32>* @VectorResult, align 64
  %559 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %560 = call <16 x i32> @llvm.hexagon.V6.vcl0w(<16 x i32> %559)
  store volatile <16 x i32> %560, <16 x i32>* @VectorResult, align 64
  %561 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %562 = call <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32> %561, i32 -1)
  store volatile <16 x i32> %562, <16 x i32>* @VectorResult, align 64
  %563 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %564 = call <16 x i32> @llvm.hexagon.V6.vrmpyub(<16 x i32> %563, i32 -1)
  store volatile <16 x i32> %564, <16 x i32>* @VectorResult, align 64
  %565 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %566 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %567 = call <16 x i32> @llvm.hexagon.V6.vrmpyubv(<16 x i32> %565, <16 x i32> %566)
  store volatile <16 x i32> %567, <16 x i32>* @VectorResult, align 64
  %568 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %569 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %570 = call <16 x i32> @llvm.hexagon.V6.vrmpyub.acc(<16 x i32> %568, <16 x i32> %569, i32 -1)
  store volatile <16 x i32> %570, <16 x i32>* @VectorResult, align 64
  %571 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %572 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %573 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 2), align 64
  %574 = call <16 x i32> @llvm.hexagon.V6.vrmpyubv.acc(<16 x i32> %571, <16 x i32> %572, <16 x i32> %573)
  store volatile <16 x i32> %574, <16 x i32>* @VectorResult, align 64
  %575 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %576 = bitcast <16 x i32> %575 to <512 x i1>
  %577 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %578 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %579 = call <16 x i32> @llvm.hexagon.V6.vaddwq(<512 x i1> %576, <16 x i32> %577, <16 x i32> %578)
  store volatile <16 x i32> %579, <16 x i32>* @VectorResult, align 64
  %580 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %581 = bitcast <16 x i32> %580 to <512 x i1>
  %582 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %583 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %584 = call <16 x i32> @llvm.hexagon.V6.vaddwnq(<512 x i1> %581, <16 x i32> %582, <16 x i32> %583)
  store volatile <16 x i32> %584, <16 x i32>* @VectorResult, align 64
  %585 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %586 = bitcast <16 x i32> %585 to <512 x i1>
  %587 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %588 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %589 = call <16 x i32> @llvm.hexagon.V6.vsubwq(<512 x i1> %586, <16 x i32> %587, <16 x i32> %588)
  store volatile <16 x i32> %589, <16 x i32>* @VectorResult, align 64
  %590 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %591 = bitcast <16 x i32> %590 to <512 x i1>
  %592 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %593 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %594 = call <16 x i32> @llvm.hexagon.V6.vsubwnq(<512 x i1> %591, <16 x i32> %592, <16 x i32> %593)
  store volatile <16 x i32> %594, <16 x i32>* @VectorResult, align 64
  %595 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %596 = call <16 x i32> @llvm.hexagon.V6.vabsw(<16 x i32> %595)
  store volatile <16 x i32> %596, <16 x i32>* @VectorResult, align 64
  %597 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %598 = call <16 x i32> @llvm.hexagon.V6.vabsw.sat(<16 x i32> %597)
  store volatile <16 x i32> %598, <16 x i32>* @VectorResult, align 64
  %599 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %600 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %601 = call <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32> %599, <16 x i32> %600)
  store volatile <16 x i32> %601, <16 x i32>* @VectorResult, align 64
  %602 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %603 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %604 = call <16 x i32> @llvm.hexagon.V6.vaddwsat(<16 x i32> %602, <16 x i32> %603)
  store volatile <16 x i32> %604, <16 x i32>* @VectorResult, align 64
  %605 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %606 = call <16 x i32> @llvm.hexagon.V6.vaslw(<16 x i32> %605, i32 -1)
  store volatile <16 x i32> %606, <16 x i32>* @VectorResult, align 64
  %607 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %608 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %609 = call <16 x i32> @llvm.hexagon.V6.vaslwv(<16 x i32> %607, <16 x i32> %608)
  store volatile <16 x i32> %609, <16 x i32>* @VectorResult, align 64
  %610 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %611 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %612 = call <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32> %610, <16 x i32> %611, i32 -1)
  store volatile <16 x i32> %612, <16 x i32>* @VectorResult, align 64
  %613 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %614 = call <16 x i32> @llvm.hexagon.V6.vasrw(<16 x i32> %613, i32 -1)
  store volatile <16 x i32> %614, <16 x i32>* @VectorResult, align 64
  %615 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %616 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %617 = call <16 x i32> @llvm.hexagon.V6.vasrwv(<16 x i32> %615, <16 x i32> %616)
  store volatile <16 x i32> %617, <16 x i32>* @VectorResult, align 64
  %618 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %619 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %620 = call <16 x i32> @llvm.hexagon.V6.vasrw.acc(<16 x i32> %618, <16 x i32> %619, i32 -1)
  store volatile <16 x i32> %620, <16 x i32>* @VectorResult, align 64
  %621 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %622 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %623 = call <16 x i32> @llvm.hexagon.V6.vavgw(<16 x i32> %621, <16 x i32> %622)
  store volatile <16 x i32> %623, <16 x i32>* @VectorResult, align 64
  %624 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %625 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %626 = call <16 x i32> @llvm.hexagon.V6.vavgwrnd(<16 x i32> %624, <16 x i32> %625)
  store volatile <16 x i32> %626, <16 x i32>* @VectorResult, align 64
  %627 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %628 = call <16 x i32> @llvm.hexagon.V6.vdmpyhb(<16 x i32> %627, i32 -1)
  store volatile <16 x i32> %628, <16 x i32>* @VectorResult, align 64
  %629 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %630 = call <16 x i32> @llvm.hexagon.V6.vdmpyhsat(<16 x i32> %629, i32 -1)
  store volatile <16 x i32> %630, <16 x i32>* @VectorResult, align 64
  %631 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %632 = call <16 x i32> @llvm.hexagon.V6.vdmpyhsusat(<16 x i32> %631, i32 -1)
  store volatile <16 x i32> %632, <16 x i32>* @VectorResult, align 64
  %633 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %634 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %635 = call <16 x i32> @llvm.hexagon.V6.vdmpyhvsat(<16 x i32> %633, <16 x i32> %634)
  store volatile <16 x i32> %635, <16 x i32>* @VectorResult, align 64
  %636 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %637 = call <16 x i32> @llvm.hexagon.V6.vdmpyhisat(<32 x i32> %636, i32 -1)
  store volatile <16 x i32> %637, <16 x i32>* @VectorResult, align 64
  %638 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %639 = call <16 x i32> @llvm.hexagon.V6.vdmpyhsuisat(<32 x i32> %638, i32 -1)
  store volatile <16 x i32> %639, <16 x i32>* @VectorResult, align 64
  %640 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %641 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %642 = call <16 x i32> @llvm.hexagon.V6.vdmpyhb.acc(<16 x i32> %640, <16 x i32> %641, i32 -1)
  store volatile <16 x i32> %642, <16 x i32>* @VectorResult, align 64
  %643 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %644 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %645 = call <16 x i32> @llvm.hexagon.V6.vdmpyhsat.acc(<16 x i32> %643, <16 x i32> %644, i32 -1)
  store volatile <16 x i32> %645, <16 x i32>* @VectorResult, align 64
  %646 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %647 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %648 = call <16 x i32> @llvm.hexagon.V6.vdmpyhsusat.acc(<16 x i32> %646, <16 x i32> %647, i32 -1)
  store volatile <16 x i32> %648, <16 x i32>* @VectorResult, align 64
  %649 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %650 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %651 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 2), align 64
  %652 = call <16 x i32> @llvm.hexagon.V6.vdmpyhvsat.acc(<16 x i32> %649, <16 x i32> %650, <16 x i32> %651)
  store volatile <16 x i32> %652, <16 x i32>* @VectorResult, align 64
  %653 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %654 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %655 = call <16 x i32> @llvm.hexagon.V6.vdmpyhisat.acc(<16 x i32> %653, <32 x i32> %654, i32 -1)
  store volatile <16 x i32> %655, <16 x i32>* @VectorResult, align 64
  %656 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %657 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %658 = call <16 x i32> @llvm.hexagon.V6.vdmpyhsuisat.acc(<16 x i32> %656, <32 x i32> %657, i32 -1)
  store volatile <16 x i32> %658, <16 x i32>* @VectorResult, align 64
  %659 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %660 = call <16 x i32> @llvm.hexagon.V6.vinsertwr(<16 x i32> %659, i32 -1)
  store volatile <16 x i32> %660, <16 x i32>* @VectorResult, align 64
  %661 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %662 = call <16 x i32> @llvm.hexagon.V6.vinsertwr(<16 x i32> %661, i32 0)
  store volatile <16 x i32> %662, <16 x i32>* @VectorResult, align 64
  %663 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %664 = call <16 x i32> @llvm.hexagon.V6.vinsertwr(<16 x i32> %663, i32 1)
  store volatile <16 x i32> %664, <16 x i32>* @VectorResult, align 64
  %665 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %666 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %667 = call <16 x i32> @llvm.hexagon.V6.vlsrwv(<16 x i32> %665, <16 x i32> %666)
  store volatile <16 x i32> %667, <16 x i32>* @VectorResult, align 64
  %668 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %669 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %670 = call <16 x i32> @llvm.hexagon.V6.vmaxw(<16 x i32> %668, <16 x i32> %669)
  store volatile <16 x i32> %670, <16 x i32>* @VectorResult, align 64
  %671 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %672 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %673 = call <16 x i32> @llvm.hexagon.V6.vminw(<16 x i32> %671, <16 x i32> %672)
  store volatile <16 x i32> %673, <16 x i32>* @VectorResult, align 64
  %674 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %675 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %676 = call <16 x i32> @llvm.hexagon.V6.vmpyewuh(<16 x i32> %674, <16 x i32> %675)
  store volatile <16 x i32> %676, <16 x i32>* @VectorResult, align 64
  %677 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %678 = call <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32> %677, i32 -1)
  store volatile <16 x i32> %678, <16 x i32>* @VectorResult, align 64
  %679 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %680 = call <16 x i32> @llvm.hexagon.V6.vmpyiwh(<16 x i32> %679, i32 -1)
  store volatile <16 x i32> %680, <16 x i32>* @VectorResult, align 64
  %681 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %682 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %683 = call <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32> %681, <16 x i32> %682, i32 -1)
  store volatile <16 x i32> %683, <16 x i32>* @VectorResult, align 64
  %684 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %685 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %686 = call <16 x i32> @llvm.hexagon.V6.vmpyiwh.acc(<16 x i32> %684, <16 x i32> %685, i32 -1)
  store volatile <16 x i32> %686, <16 x i32>* @VectorResult, align 64
  %687 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %688 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %689 = call <16 x i32> @llvm.hexagon.V6.vmpyiewuh(<16 x i32> %687, <16 x i32> %688)
  store volatile <16 x i32> %689, <16 x i32>* @VectorResult, align 64
  %690 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %691 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %692 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 2), align 64
  %693 = call <16 x i32> @llvm.hexagon.V6.vmpyiewh.acc(<16 x i32> %690, <16 x i32> %691, <16 x i32> %692)
  store volatile <16 x i32> %693, <16 x i32>* @VectorResult, align 64
  %694 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %695 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %696 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 2), align 64
  %697 = call <16 x i32> @llvm.hexagon.V6.vmpyiewuh.acc(<16 x i32> %694, <16 x i32> %695, <16 x i32> %696)
  store volatile <16 x i32> %697, <16 x i32>* @VectorResult, align 64
  %698 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %699 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %700 = call <16 x i32> @llvm.hexagon.V6.vmpyieoh(<16 x i32> %698, <16 x i32> %699)
  store volatile <16 x i32> %700, <16 x i32>* @VectorResult, align 64
  %701 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %702 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %703 = call <16 x i32> @llvm.hexagon.V6.vmpyiowh(<16 x i32> %701, <16 x i32> %702)
  store volatile <16 x i32> %703, <16 x i32>* @VectorResult, align 64
  %704 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %705 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %706 = call <16 x i32> @llvm.hexagon.V6.vmpyowh.rnd(<16 x i32> %704, <16 x i32> %705)
  store volatile <16 x i32> %706, <16 x i32>* @VectorResult, align 64
  %707 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %708 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %709 = call <16 x i32> @llvm.hexagon.V6.vmpyowh(<16 x i32> %707, <16 x i32> %708)
  store volatile <16 x i32> %709, <16 x i32>* @VectorResult, align 64
  %710 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %711 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %712 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 2), align 64
  %713 = call <16 x i32> @llvm.hexagon.V6.vmpyowh.rnd.sacc(<16 x i32> %710, <16 x i32> %711, <16 x i32> %712)
  store volatile <16 x i32> %713, <16 x i32>* @VectorResult, align 64
  %714 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %715 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %716 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 2), align 64
  %717 = call <16 x i32> @llvm.hexagon.V6.vmpyowh.sacc(<16 x i32> %714, <16 x i32> %715, <16 x i32> %716)
  store volatile <16 x i32> %717, <16 x i32>* @VectorResult, align 64
  %718 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %719 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %720 = call <16 x i32> @llvm.hexagon.V6.vnavgw(<16 x i32> %718, <16 x i32> %719)
  store volatile <16 x i32> %720, <16 x i32>* @VectorResult, align 64
  %721 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %722 = call <16 x i32> @llvm.hexagon.V6.vnormamtw(<16 x i32> %721)
  store volatile <16 x i32> %722, <16 x i32>* @VectorResult, align 64
  %723 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %724 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %725 = call <16 x i32> @llvm.hexagon.V6.vrmpybv(<16 x i32> %723, <16 x i32> %724)
  store volatile <16 x i32> %725, <16 x i32>* @VectorResult, align 64
  %726 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %727 = call <16 x i32> @llvm.hexagon.V6.vrmpybus(<16 x i32> %726, i32 -1)
  store volatile <16 x i32> %727, <16 x i32>* @VectorResult, align 64
  %728 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %729 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %730 = call <16 x i32> @llvm.hexagon.V6.vrmpybusv(<16 x i32> %728, <16 x i32> %729)
  store volatile <16 x i32> %730, <16 x i32>* @VectorResult, align 64
  %731 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %732 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %733 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 2), align 64
  %734 = call <16 x i32> @llvm.hexagon.V6.vrmpybv.acc(<16 x i32> %731, <16 x i32> %732, <16 x i32> %733)
  store volatile <16 x i32> %734, <16 x i32>* @VectorResult, align 64
  %735 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %736 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %737 = call <16 x i32> @llvm.hexagon.V6.vrmpybus.acc(<16 x i32> %735, <16 x i32> %736, i32 -1)
  store volatile <16 x i32> %737, <16 x i32>* @VectorResult, align 64
  %738 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %739 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %740 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 2), align 64
  %741 = call <16 x i32> @llvm.hexagon.V6.vrmpybusv.acc(<16 x i32> %738, <16 x i32> %739, <16 x i32> %740)
  store volatile <16 x i32> %741, <16 x i32>* @VectorResult, align 64
  %742 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %743 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %744 = call <16 x i32> @llvm.hexagon.V6.vsubw(<16 x i32> %742, <16 x i32> %743)
  store volatile <16 x i32> %744, <16 x i32>* @VectorResult, align 64
  %745 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %746 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %747 = call <16 x i32> @llvm.hexagon.V6.vsubwsat(<16 x i32> %745, <16 x i32> %746)
  store volatile <16 x i32> %747, <16 x i32>* @VectorResult, align 64
  %748 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %749 = call <32 x i32> @llvm.hexagon.V6.vassignp(<32 x i32> %748)
  store volatile <32 x i32> %749, <32 x i32>* @VectorPairResult, align 128
  %750 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %751 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %752 = call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %750, <16 x i32> %751)
  store volatile <32 x i32> %752, <32 x i32>* @VectorPairResult, align 128
  %753 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %754 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %755 = call <32 x i32> @llvm.hexagon.V6.vdealvdd(<16 x i32> %753, <16 x i32> %754, i32 -1)
  store volatile <32 x i32> %755, <32 x i32>* @VectorPairResult, align 128
  %756 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %757 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %758 = call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> %756, <16 x i32> %757, i32 -1)
  store volatile <32 x i32> %758, <32 x i32>* @VectorPairResult, align 128
  %759 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %760 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %761 = call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> %759, <16 x i32> %760, i32 0)
  store volatile <32 x i32> %761, <32 x i32>* @VectorPairResult, align 128
  %762 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %763 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %764 = call <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32> %762, <16 x i32> %763, i32 1)
  store volatile <32 x i32> %764, <32 x i32>* @VectorPairResult, align 128
  %765 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vecpreds, i32 0, i32 0), align 64
  %766 = bitcast <16 x i32> %765 to <512 x i1>
  %767 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %768 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %769 = call <32 x i32> @llvm.hexagon.V6.vswap(<512 x i1> %766, <16 x i32> %767, <16 x i32> %768)
  store volatile <32 x i32> %769, <32 x i32>* @VectorPairResult, align 128
  %770 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %771 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %772 = call <32 x i32> @llvm.hexagon.V6.vaddb.dv(<32 x i32> %770, <32 x i32> %771)
  store volatile <32 x i32> %772, <32 x i32>* @VectorPairResult, align 128
  %773 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %774 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %775 = call <32 x i32> @llvm.hexagon.V6.vshufoeb(<16 x i32> %773, <16 x i32> %774)
  store volatile <32 x i32> %775, <32 x i32>* @VectorPairResult, align 128
  %776 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %777 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %778 = call <32 x i32> @llvm.hexagon.V6.vsubb.dv(<32 x i32> %776, <32 x i32> %777)
  store volatile <32 x i32> %778, <32 x i32>* @VectorPairResult, align 128
  %779 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %780 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %781 = call <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32> %779, <16 x i32> %780)
  store volatile <32 x i32> %781, <32 x i32>* @VectorPairResult, align 128
  %782 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %783 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %784 = call <32 x i32> @llvm.hexagon.V6.vaddh.dv(<32 x i32> %782, <32 x i32> %783)
  store volatile <32 x i32> %784, <32 x i32>* @VectorPairResult, align 128
  %785 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %786 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %787 = call <32 x i32> @llvm.hexagon.V6.vaddhsat.dv(<32 x i32> %785, <32 x i32> %786)
  store volatile <32 x i32> %787, <32 x i32>* @VectorPairResult, align 128
  %788 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %789 = call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv(<32 x i32> %788, i32 -1)
  store volatile <32 x i32> %789, <32 x i32>* @VectorPairResult, align 128
  %790 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %791 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %792 = call <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32> %790, <32 x i32> %791, i32 -1)
  store volatile <32 x i32> %792, <32 x i32>* @VectorPairResult, align 128
  %793 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %794 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %795 = call <32 x i32> @llvm.hexagon.V6.vlutvwh(<16 x i32> %793, <16 x i32> %794, i32 -1)
  store volatile <32 x i32> %795, <32 x i32>* @VectorPairResult, align 128
  %796 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %797 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %798 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %799 = call <32 x i32> @llvm.hexagon.V6.vlutvwh.oracc(<32 x i32> %796, <16 x i32> %797, <16 x i32> %798, i32 -1)
  store volatile <32 x i32> %799, <32 x i32>* @VectorPairResult, align 128
  %800 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %801 = call <32 x i32> @llvm.hexagon.V6.vmpabus(<32 x i32> %800, i32 -1)
  store volatile <32 x i32> %801, <32 x i32>* @VectorPairResult, align 128
  %802 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %803 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %804 = call <32 x i32> @llvm.hexagon.V6.vmpabusv(<32 x i32> %802, <32 x i32> %803)
  store volatile <32 x i32> %804, <32 x i32>* @VectorPairResult, align 128
  %805 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %806 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %807 = call <32 x i32> @llvm.hexagon.V6.vmpabuuv(<32 x i32> %805, <32 x i32> %806)
  store volatile <32 x i32> %807, <32 x i32>* @VectorPairResult, align 128
  %808 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %809 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %810 = call <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32> %808, <32 x i32> %809, i32 -1)
  store volatile <32 x i32> %810, <32 x i32>* @VectorPairResult, align 128
  %811 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %812 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %813 = call <32 x i32> @llvm.hexagon.V6.vmpybv(<16 x i32> %811, <16 x i32> %812)
  store volatile <32 x i32> %813, <32 x i32>* @VectorPairResult, align 128
  %814 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %815 = call <32 x i32> @llvm.hexagon.V6.vmpybus(<16 x i32> %814, i32 -1)
  store volatile <32 x i32> %815, <32 x i32>* @VectorPairResult, align 128
  %816 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %817 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %818 = call <32 x i32> @llvm.hexagon.V6.vmpybusv(<16 x i32> %816, <16 x i32> %817)
  store volatile <32 x i32> %818, <32 x i32>* @VectorPairResult, align 128
  %819 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %820 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %821 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %822 = call <32 x i32> @llvm.hexagon.V6.vmpybv.acc(<32 x i32> %819, <16 x i32> %820, <16 x i32> %821)
  store volatile <32 x i32> %822, <32 x i32>* @VectorPairResult, align 128
  %823 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %824 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %825 = call <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32> %823, <16 x i32> %824, i32 -1)
  store volatile <32 x i32> %825, <32 x i32>* @VectorPairResult, align 128
  %826 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %827 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %828 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %829 = call <32 x i32> @llvm.hexagon.V6.vmpybusv.acc(<32 x i32> %826, <16 x i32> %827, <16 x i32> %828)
  store volatile <32 x i32> %829, <32 x i32>* @VectorPairResult, align 128
  %830 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %831 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %832 = call <32 x i32> @llvm.hexagon.V6.vshufoeh(<16 x i32> %830, <16 x i32> %831)
  store volatile <32 x i32> %832, <32 x i32>* @VectorPairResult, align 128
  %833 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %834 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %835 = call <32 x i32> @llvm.hexagon.V6.vsububh(<16 x i32> %833, <16 x i32> %834)
  store volatile <32 x i32> %835, <32 x i32>* @VectorPairResult, align 128
  %836 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %837 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %838 = call <32 x i32> @llvm.hexagon.V6.vsubh.dv(<32 x i32> %836, <32 x i32> %837)
  store volatile <32 x i32> %838, <32 x i32>* @VectorPairResult, align 128
  %839 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %840 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %841 = call <32 x i32> @llvm.hexagon.V6.vsubhsat.dv(<32 x i32> %839, <32 x i32> %840)
  store volatile <32 x i32> %841, <32 x i32>* @VectorPairResult, align 128
  %842 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %843 = call <32 x i32> @llvm.hexagon.V6.vsb(<16 x i32> %842)
  store volatile <32 x i32> %843, <32 x i32>* @VectorPairResult, align 128
  %844 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %845 = call <32 x i32> @llvm.hexagon.V6.vtmpyb(<32 x i32> %844, i32 -1)
  store volatile <32 x i32> %845, <32 x i32>* @VectorPairResult, align 128
  %846 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %847 = call <32 x i32> @llvm.hexagon.V6.vtmpybus(<32 x i32> %846, i32 -1)
  store volatile <32 x i32> %847, <32 x i32>* @VectorPairResult, align 128
  %848 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %849 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %850 = call <32 x i32> @llvm.hexagon.V6.vtmpyb.acc(<32 x i32> %848, <32 x i32> %849, i32 -1)
  store volatile <32 x i32> %850, <32 x i32>* @VectorPairResult, align 128
  %851 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %852 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %853 = call <32 x i32> @llvm.hexagon.V6.vtmpybus.acc(<32 x i32> %851, <32 x i32> %852, i32 -1)
  store volatile <32 x i32> %853, <32 x i32>* @VectorPairResult, align 128
  %854 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %855 = call <32 x i32> @llvm.hexagon.V6.vunpackb(<16 x i32> %854)
  store volatile <32 x i32> %855, <32 x i32>* @VectorPairResult, align 128
  %856 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %857 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %858 = call <32 x i32> @llvm.hexagon.V6.vunpackob(<32 x i32> %856, <16 x i32> %857)
  store volatile <32 x i32> %858, <32 x i32>* @VectorPairResult, align 128
  %859 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %860 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %861 = call <32 x i32> @llvm.hexagon.V6.vaddubsat.dv(<32 x i32> %859, <32 x i32> %860)
  store volatile <32 x i32> %861, <32 x i32>* @VectorPairResult, align 128
  %862 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %863 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %864 = call <32 x i32> @llvm.hexagon.V6.vsububsat.dv(<32 x i32> %862, <32 x i32> %863)
  store volatile <32 x i32> %864, <32 x i32>* @VectorPairResult, align 128
  %865 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %866 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %867 = call <32 x i32> @llvm.hexagon.V6.vadduhsat.dv(<32 x i32> %865, <32 x i32> %866)
  store volatile <32 x i32> %867, <32 x i32>* @VectorPairResult, align 128
  %868 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %869 = call <32 x i32> @llvm.hexagon.V6.vmpyub(<16 x i32> %868, i32 -1)
  store volatile <32 x i32> %869, <32 x i32>* @VectorPairResult, align 128
  %870 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %871 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %872 = call <32 x i32> @llvm.hexagon.V6.vmpyubv(<16 x i32> %870, <16 x i32> %871)
  store volatile <32 x i32> %872, <32 x i32>* @VectorPairResult, align 128
  %873 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %874 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %875 = call <32 x i32> @llvm.hexagon.V6.vmpyub.acc(<32 x i32> %873, <16 x i32> %874, i32 -1)
  store volatile <32 x i32> %875, <32 x i32>* @VectorPairResult, align 128
  %876 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %877 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %878 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %879 = call <32 x i32> @llvm.hexagon.V6.vmpyubv.acc(<32 x i32> %876, <16 x i32> %877, <16 x i32> %878)
  store volatile <32 x i32> %879, <32 x i32>* @VectorPairResult, align 128
  %880 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %881 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %882 = call <32 x i32> @llvm.hexagon.V6.vsubuhsat.dv(<32 x i32> %880, <32 x i32> %881)
  store volatile <32 x i32> %882, <32 x i32>* @VectorPairResult, align 128
  %883 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %884 = call <32 x i32> @llvm.hexagon.V6.vunpackub(<16 x i32> %883)
  store volatile <32 x i32> %884, <32 x i32>* @VectorPairResult, align 128
  %885 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %886 = call <32 x i32> @llvm.hexagon.V6.vzb(<16 x i32> %885)
  store volatile <32 x i32> %886, <32 x i32>* @VectorPairResult, align 128
  %887 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %888 = call <32 x i32> @llvm.hexagon.V6.vdsaduh(<32 x i32> %887, i32 -1)
  store volatile <32 x i32> %888, <32 x i32>* @VectorPairResult, align 128
  %889 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %890 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %891 = call <32 x i32> @llvm.hexagon.V6.vdsaduh.acc(<32 x i32> %889, <32 x i32> %890, i32 -1)
  store volatile <32 x i32> %891, <32 x i32>* @VectorPairResult, align 128
  %892 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %893 = call <32 x i32> @llvm.hexagon.V6.vmpyuh(<16 x i32> %892, i32 -1)
  store volatile <32 x i32> %893, <32 x i32>* @VectorPairResult, align 128
  %894 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %895 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %896 = call <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32> %894, <16 x i32> %895)
  store volatile <32 x i32> %896, <32 x i32>* @VectorPairResult, align 128
  %897 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %898 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %899 = call <32 x i32> @llvm.hexagon.V6.vmpyuh.acc(<32 x i32> %897, <16 x i32> %898, i32 -1)
  store volatile <32 x i32> %899, <32 x i32>* @VectorPairResult, align 128
  %900 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %901 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %902 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %903 = call <32 x i32> @llvm.hexagon.V6.vmpyuhv.acc(<32 x i32> %900, <16 x i32> %901, <16 x i32> %902)
  store volatile <32 x i32> %903, <32 x i32>* @VectorPairResult, align 128
  %904 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %905 = call <32 x i32> @llvm.hexagon.V6.vrmpyubi(<32 x i32> %904, i32 -1, i32 0)
  store volatile <32 x i32> %905, <32 x i32>* @VectorPairResult, align 128
  %906 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %907 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %908 = call <32 x i32> @llvm.hexagon.V6.vrmpyubi.acc(<32 x i32> %906, <32 x i32> %907, i32 -1, i32 0)
  store volatile <32 x i32> %908, <32 x i32>* @VectorPairResult, align 128
  %909 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %910 = call <32 x i32> @llvm.hexagon.V6.vrsadubi(<32 x i32> %909, i32 -1, i32 0)
  store volatile <32 x i32> %910, <32 x i32>* @VectorPairResult, align 128
  %911 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %912 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %913 = call <32 x i32> @llvm.hexagon.V6.vrsadubi.acc(<32 x i32> %911, <32 x i32> %912, i32 -1, i32 0)
  store volatile <32 x i32> %913, <32 x i32>* @VectorPairResult, align 128
  %914 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %915 = call <32 x i32> @llvm.hexagon.V6.vunpackuh(<16 x i32> %914)
  store volatile <32 x i32> %915, <32 x i32>* @VectorPairResult, align 128
  %916 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %917 = call <32 x i32> @llvm.hexagon.V6.vzh(<16 x i32> %916)
  store volatile <32 x i32> %917, <32 x i32>* @VectorPairResult, align 128
  %918 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %919 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %920 = call <32 x i32> @llvm.hexagon.V6.vaddhw(<16 x i32> %918, <16 x i32> %919)
  store volatile <32 x i32> %920, <32 x i32>* @VectorPairResult, align 128
  %921 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %922 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %923 = call <32 x i32> @llvm.hexagon.V6.vadduhw(<16 x i32> %921, <16 x i32> %922)
  store volatile <32 x i32> %923, <32 x i32>* @VectorPairResult, align 128
  %924 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %925 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %926 = call <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32> %924, <32 x i32> %925)
  store volatile <32 x i32> %926, <32 x i32>* @VectorPairResult, align 128
  %927 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %928 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %929 = call <32 x i32> @llvm.hexagon.V6.vaddwsat.dv(<32 x i32> %927, <32 x i32> %928)
  store volatile <32 x i32> %929, <32 x i32>* @VectorPairResult, align 128
  %930 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %931 = call <32 x i32> @llvm.hexagon.V6.vdmpyhb.dv(<32 x i32> %930, i32 -1)
  store volatile <32 x i32> %931, <32 x i32>* @VectorPairResult, align 128
  %932 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %933 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %934 = call <32 x i32> @llvm.hexagon.V6.vdmpyhb.dv.acc(<32 x i32> %932, <32 x i32> %933, i32 -1)
  store volatile <32 x i32> %934, <32 x i32>* @VectorPairResult, align 128
  %935 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %936 = call <32 x i32> @llvm.hexagon.V6.vmpahb(<32 x i32> %935, i32 -1)
  store volatile <32 x i32> %936, <32 x i32>* @VectorPairResult, align 128
  %937 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %938 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %939 = call <32 x i32> @llvm.hexagon.V6.vmpahb.acc(<32 x i32> %937, <32 x i32> %938, i32 -1)
  store volatile <32 x i32> %939, <32 x i32>* @VectorPairResult, align 128
  %940 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %941 = call <32 x i32> @llvm.hexagon.V6.vmpyh(<16 x i32> %940, i32 -1)
  store volatile <32 x i32> %941, <32 x i32>* @VectorPairResult, align 128
  %942 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %943 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %944 = call <32 x i32> @llvm.hexagon.V6.vmpyhv(<16 x i32> %942, <16 x i32> %943)
  store volatile <32 x i32> %944, <32 x i32>* @VectorPairResult, align 128
  %945 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %946 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %947 = call <32 x i32> @llvm.hexagon.V6.vmpyhus(<16 x i32> %945, <16 x i32> %946)
  store volatile <32 x i32> %947, <32 x i32>* @VectorPairResult, align 128
  %948 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %949 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %950 = call <32 x i32> @llvm.hexagon.V6.vmpyhsat.acc(<32 x i32> %948, <16 x i32> %949, i32 -1)
  store volatile <32 x i32> %950, <32 x i32>* @VectorPairResult, align 128
  %951 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %952 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %953 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %954 = call <32 x i32> @llvm.hexagon.V6.vmpyhv.acc(<32 x i32> %951, <16 x i32> %952, <16 x i32> %953)
  store volatile <32 x i32> %954, <32 x i32>* @VectorPairResult, align 128
  %955 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %956 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %957 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %958 = call <32 x i32> @llvm.hexagon.V6.vmpyhus.acc(<32 x i32> %955, <16 x i32> %956, <16 x i32> %957)
  store volatile <32 x i32> %958, <32 x i32>* @VectorPairResult, align 128
  %959 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %960 = call <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32> %959, i32 -1, i32 0)
  store volatile <32 x i32> %960, <32 x i32>* @VectorPairResult, align 128
  %961 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %962 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %963 = call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %961, <32 x i32> %962, i32 -1, i32 0)
  store volatile <32 x i32> %963, <32 x i32>* @VectorPairResult, align 128
  %964 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %965 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %966 = call <32 x i32> @llvm.hexagon.V6.vsubhw(<16 x i32> %964, <16 x i32> %965)
  store volatile <32 x i32> %966, <32 x i32>* @VectorPairResult, align 128
  %967 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %968 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 1), align 64
  %969 = call <32 x i32> @llvm.hexagon.V6.vsubuhw(<16 x i32> %967, <16 x i32> %968)
  store volatile <32 x i32> %969, <32 x i32>* @VectorPairResult, align 128
  %970 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %971 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %972 = call <32 x i32> @llvm.hexagon.V6.vsubw.dv(<32 x i32> %970, <32 x i32> %971)
  store volatile <32 x i32> %972, <32 x i32>* @VectorPairResult, align 128
  %973 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %974 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %975 = call <32 x i32> @llvm.hexagon.V6.vsubwsat.dv(<32 x i32> %973, <32 x i32> %974)
  store volatile <32 x i32> %975, <32 x i32>* @VectorPairResult, align 128
  %976 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %977 = call <32 x i32> @llvm.hexagon.V6.vsh(<16 x i32> %976)
  store volatile <32 x i32> %977, <32 x i32>* @VectorPairResult, align 128
  %978 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %979 = call <32 x i32> @llvm.hexagon.V6.vtmpyhb(<32 x i32> %978, i32 -1)
  store volatile <32 x i32> %979, <32 x i32>* @VectorPairResult, align 128
  %980 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %981 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 1), align 128
  %982 = call <32 x i32> @llvm.hexagon.V6.vtmpyhb.acc(<32 x i32> %980, <32 x i32> %981, i32 -1)
  store volatile <32 x i32> %982, <32 x i32>* @VectorPairResult, align 128
  %983 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %984 = call <32 x i32> @llvm.hexagon.V6.vunpackh(<16 x i32> %983)
  store volatile <32 x i32> %984, <32 x i32>* @VectorPairResult, align 128
  %985 = load volatile <32 x i32>, <32 x i32>* getelementptr inbounds ([15 x <32 x i32>], [15 x <32 x i32>]* @vector_pairs, i32 0, i32 0), align 128
  %986 = load volatile <16 x i32>, <16 x i32>* getelementptr inbounds ([15 x <16 x i32>], [15 x <16 x i32>]* @vectors, i32 0, i32 0), align 64
  %987 = call <32 x i32> @llvm.hexagon.V6.vunpackoh(<32 x i32> %985, <16 x i32> %986)
  store volatile <32 x i32> %987, <32 x i32>* @VectorPairResult, align 128
  ret i32 0
}

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.pred.and(<512 x i1>, <512 x i1>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.pred.and.n(<512 x i1>, <512 x i1>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.pred.not(<512 x i1>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.pred.or(<512 x i1>, <512 x i1>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.pred.or.n(<512 x i1>, <512 x i1>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vandvrt.acc(<512 x i1>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.veqb(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.veqh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.veqw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.veqb.and(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.veqh.and(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.veqw.and(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.veqb.or(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.veqh.or(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.veqw.or(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.veqb.xor(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.veqh.xor(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.veqw.xor(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtb(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgth(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtub(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtuh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtuw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtb.and(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgth.and(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtub.and(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtuh.and(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtuw.and(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtw.and(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtb.or(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgth.or(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtub.or(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtuh.or(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtuw.or(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtw.or(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtb.xor(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgth.xor(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtub.xor(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtuh.xor(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtuw.xor(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.vgtw.xor(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.pred.scalar2(i32) #1

; Function Attrs: nounwind readnone
declare <512 x i1> @llvm.hexagon.V6.pred.xor(<512 x i1>, <512 x i1>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vassign(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lo(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.valignb(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vandqrt(<512 x i1>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vand(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vandqrt.acc(<16 x i32>, <512 x i1>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdelta(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlalignbi(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlalignb(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmux(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vnot(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vor(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vrdelta(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vror(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vxor(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vd0() #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddbq(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddbnq(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubbq(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubbnq(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddb(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrhbrndsat(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdealb(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdealb4w(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlutvvb(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlutvvb.oracc(<16 x i32>, <16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vnavgub(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vpackhb.sat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vpackeb(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vpackob(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vroundhb(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshuffb(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshuffeb(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshuffob(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubb(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddhq(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddhnq(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubhq(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubhnq(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vabsh(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vabsh.sat(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddhsat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaslh(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaslhv(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrh(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrhv(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrwhrndsat(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrwhsat(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vavgh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vavghrnd(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdealh(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpybus(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpybus.acc(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlsrhv(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmaxh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vminh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyhsrs(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyhss(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyhvsrs(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyihb(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyih(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyihb.acc(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyih.acc(<16 x i32>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vnavgh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vnormamth(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vpackwh.sat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vpackeh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vpackoh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vpopcounth(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vroundwh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsatwh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshuffh(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshufeh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vshufoh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubhsat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vabsdiffub(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddubsat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrhubrndsat(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrhubsat(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vavgub(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vavgubrnd(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmaxub(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vminub(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vpackhub.sat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vroundhub(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsububsat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vabsdiffh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vabsdiffuh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vadduhsat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrwuhsat(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vavguh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vavguhrnd(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vcl0h(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlsrh(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmaxuh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vminuh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vpackwuh.sat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vroundwuh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubuhsat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vabsdiffw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vcl0w(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlsrw(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vrmpyub(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vrmpyubv(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vrmpyub.acc(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vrmpyubv.acc(<16 x i32>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddwq(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddwnq(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubwq(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubwnq(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vabsw(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vabsw.sat(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaddwsat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaslw(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaslwv(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vaslw.acc(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrw(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrwv(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vasrw.acc(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vavgw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vavgwrnd(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpyhb(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpyhsat(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpyhsusat(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpyhvsat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpyhisat(<32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpyhsuisat(<32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpyhb.acc(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpyhsat.acc(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpyhsusat.acc(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpyhvsat.acc(<16 x i32>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpyhisat.acc(<16 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vdmpyhsuisat.acc(<16 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vinsertwr(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vlsrwv(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmaxw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vminw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyewuh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyiwb(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyiwh(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyiwb.acc(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyiwh.acc(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyiewuh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyiewh.acc(<16 x i32>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyiewuh.acc(<16 x i32>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyieoh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyiowh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyowh.rnd(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyowh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyowh.rnd.sacc(<16 x i32>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vmpyowh.sacc(<16 x i32>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vnavgw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vnormamtw(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vrmpybv(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vrmpybus(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vrmpybusv(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vrmpybv.acc(<16 x i32>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vrmpybus.acc(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vrmpybusv.acc(<16 x i32>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.vsubwsat(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vassignp(<32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vdealvdd(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vshuffvdd(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vswap(<512 x i1>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddb.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vshufoeb(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vsubb.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddubh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddh.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddhsat.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vdmpybus.dv(<32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vdmpybus.dv.acc(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vlutvwh(<16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vlutvwh.oracc(<32 x i32>, <16 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpabus(<32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpabusv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpabuuv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpabus.acc(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpybv(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpybus(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpybusv(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpybv.acc(<32 x i32>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpybus.acc(<32 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpybusv.acc(<32 x i32>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vshufoeh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vsububh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vsubh.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vsubhsat.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vsb(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vtmpyb(<32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vtmpybus(<32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vtmpyb.acc(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vtmpybus.acc(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vunpackb(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vunpackob(<32 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddubsat.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vsububsat.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vadduhsat.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyub(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyubv(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyub.acc(<32 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyubv.acc(<32 x i32>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vsubuhsat.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vunpackub(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vzb(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vdsaduh(<32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vdsaduh.acc(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyuh(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyuhv(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyuh.acc(<32 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyuhv.acc(<32 x i32>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vrmpyubi(<32 x i32>, i32, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vrmpyubi.acc(<32 x i32>, <32 x i32>, i32, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vrsadubi(<32 x i32>, i32, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vrsadubi.acc(<32 x i32>, <32 x i32>, i32, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vunpackuh(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vzh(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddhw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vadduhw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddw.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vaddwsat.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vdmpyhb.dv(<32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vdmpyhb.dv.acc(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpahb(<32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpahb.acc(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyh(<16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyhv(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyhus(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyhsat.acc(<32 x i32>, <16 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyhv.acc(<32 x i32>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmpyhus.acc(<32 x i32>, <16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32>, i32, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32>, <32 x i32>, i32, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vsubhw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vsubuhw(<16 x i32>, <16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vsubw.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vsubwsat.dv(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vsh(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vtmpyhb(<32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vtmpyhb.acc(<32 x i32>, <32 x i32>, i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vunpackh(<16 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vunpackoh(<32 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

; RUN: opt %s -passes='print<uniformity>' -disable-output > %t 2>&1
; RUN: FileCheck %s < %t
; RUN: FileCheck %s --check-prefix=UNIFORM < %t
; UNIFORM-NOT: DIVERGENT: %uniform.

target triple = "nvptx64-nvidia-cuda"

; Values named uniform.* must remain uniform; mutable memory and unknown calls
; can return different values to threads with identical operands.
define ptx_kernel void @sources(i32 noundef %arg, ptr addrspace(1) %global, ptr addrspace(4) %constant, ptr addrspace(5) %local) {
; CHECK-LABEL: for function 'sources'
; CHECK-DAG: DIVERGENT: %mutable =
; CHECK-DAG: DIVERGENT: %atomic =
; CHECK-DAG: DIVERGENT: %volatile =
; CHECK-DAG: DIVERGENT: %allocation =
; CHECK-DAG: DIVERGENT: %local.generic =
; CHECK-DAG: DIVERGENT: %frozen =
; CHECK-DAG: DIVERGENT: %thread =
; CHECK-DAG: DIVERGENT: %clock =
; CHECK-DAG: DIVERGENT: %unknown =
  %uniform.block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %uniform.size = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %uniform.min = call i32 @llvm.umin.i32(i32 %arg, i32 %uniform.block)
  %uniform.constant = load i32, ptr addrspace(4) %constant
  %uniform.frozen = freeze i32 %arg
  %mutable = load i32, ptr addrspace(1) %global
  %atomic = load atomic i32, ptr addrspace(1) %global monotonic, align 4
  %volatile = load volatile i32, ptr addrspace(4) %constant
  %allocation = alloca i32
  %local.generic = addrspacecast ptr addrspace(5) %local to ptr
  %frozen = freeze i32 poison
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %clock = call i32 @llvm.nvvm.read.ptx.sreg.clock()
  %unknown = call i32 @unknown(i32 %arg)
  ret void
}

define ptx_kernel void @collectives(i32 noundef %mask, i32 noundef %index, i32 %maybe_undefined) {
; CHECK-LABEL: for function 'collectives'
; CHECK-DAG: DIVERGENT: %thread =
; CHECK-DAG: DIVERGENT: %frozen.broadcast =
; CHECK-DAG: DIVERGENT: %frozen.extracted =
; CHECK-DAG: DIVERGENT: %segmented =
; CHECK-DAG: DIVERGENT: %out.of.range =
; CHECK-DAG: DIVERGENT: %undefined.match.data =
; CHECK-DAG: DIVERGENT: %varying.index =
; CHECK-DAG: DIVERGENT: %varying.mask =
; CHECK-DAG: DIVERGENT: %unknown.mask =
; CHECK-DAG: DIVERGENT: %match.any =
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %uniform.broadcast = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %thread, i32 7, i32 31)
  ; An inactive source can yield undef; freezing it can differ per lane.
  %frozen.broadcast = freeze i32 %uniform.broadcast
  %uniform.pair = call {i32, i1} @llvm.nvvm.shfl.sync.idx.i32p(i32 -1, i32 %thread, i32 7, i32 31)
  %uniform.extracted = extractvalue {i32, i1} %uniform.pair, 0
  %frozen.extracted = freeze i32 %uniform.extracted
  %predicate = icmp ne i32 %thread, 0
  %uniform.vote = call i32 @llvm.nvvm.vote.ballot.sync(i32 %mask, i1 %predicate)
  %uniform.redux = call i32 @llvm.nvvm.redux.sync.add(i32 %thread, i32 %mask)
  %fp = uitofp i32 %thread to float
  %uniform.fp.redux = call float @llvm.nvvm.redux.sync.fmax(float %fp, i32 %mask)
  %uniform.match = call {i32, i1} @llvm.nvvm.match.all.sync.i32p(i32 %mask, i32 %thread)
  %uniform.partial = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 %mask, i32 %thread, i32 %index, i32 31)
  %uniform.barrier = call i32 @llvm.nvvm.barrier.cta.red.popc.count(i32 0, i32 32, i1 %predicate)
  %uniform.clamped = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 %mask, i32 %thread, i32 7, i32 15)
  %uniform.match.any = call i32 @llvm.nvvm.match.any.sync.i32(i32 %mask, i32 %index)
  %segmented = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %thread, i32 0, i32 4111)
  %out.of.range = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %thread, i32 16, i32 15)
  %undefined.match.data = call i32 @llvm.nvvm.match.any.sync.i32(i32 -1, i32 %maybe_undefined)
  %varying.index = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %thread, i32 %thread, i32 31)
  %varying.mask = call i32 @llvm.nvvm.vote.ballot.sync(i32 %thread, i1 true)
  %unknown.mask = call i32 @llvm.nvvm.vote.ballot.sync(i32 %maybe_undefined, i1 true)
  %match.any = call i32 @llvm.nvvm.match.any.sync.i32(i32 %mask, i32 %thread)
  ret void
}

; Data-only shuffles preserve uniform inputs, but their range predicates can vary.
define ptx_kernel void @uniform_shuffle_data(i32 %data) {
; CHECK-LABEL: for function 'uniform_shuffle_data'
; CHECK-DAG: DIVERGENT: %varying.data =
; CHECK-DAG: DIVERGENT: %range.predicate =
; CHECK-DAG: DIVERGENT: %frozen.data =
; CHECK-DAG: DIVERGENT: %legacy.varying =
; CHECK-DAG: DIVERGENT: %legacy.range =
; CHECK-DAG: DIVERGENT: %legacy.vote =
; CHECK-DAG: DIVERGENT: %legacy.ballot =
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %uniform.up = call i32 @llvm.nvvm.shfl.sync.up.i32(i32 -1, i32 %data, i32 1, i32 0)
  %uniform.down = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %data, i32 1, i32 31)
  %uniform.bfly = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %data, i32 1, i32 31)
  %uniform.segmented = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %data, i32 0, i32 4111)
  %varying.data = call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %thread, i32 1, i32 31)
  %pair = call {i32, i1} @llvm.nvvm.shfl.sync.up.i32p(i32 -1, i32 %data, i32 1, i32 0)
  %range.predicate = extractvalue {i32, i1} %pair, 1
  %frozen.data = freeze i32 %uniform.up
  %uniform.legacy = call i32 @llvm.nvvm.shfl.up.i32(i32 %data, i32 1, i32 0)
  %legacy.varying = call i32 @llvm.nvvm.shfl.up.i32(i32 %thread, i32 1, i32 0)
  %legacy.pair = call {i32, i1} @llvm.nvvm.shfl.up.i32p(i32 %data, i32 1, i32 0)
  %legacy.range = extractvalue {i32, i1} %legacy.pair, 1
  %uniform.legacy.vote = call i1 @llvm.nvvm.vote.all(i1 true)
  %predicate = icmp ne i32 %thread, 0
  %legacy.vote = call i1 @llvm.nvvm.vote.all(i1 %predicate)
  %legacy.ballot = call i32 @llvm.nvvm.vote.ballot(i1 true)
  ret void
}

; Separate groups can each return a uniform result that differs at the join.
define ptx_kernel void @partial_groups() {
; CHECK-LABEL: for function 'partial_groups'
; CHECK: DIVERGENT: %joined =
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  %first.half = icmp ult i32 %lane, 16
  br i1 %first.half, label %low, label %high
low:
  %uniform.low = call i32 @llvm.nvvm.vote.ballot.sync(i32 65535, i1 true)
  br label %join
high:
  %uniform.high = call i32 @llvm.nvvm.vote.ballot.sync(i32 -65536, i1 true)
  br label %join
join:
  %joined = phi i32 [%uniform.low, %low], [%uniform.high, %high]
  ret void
}

define ptx_kernel void @warp_index() "nvvm.reqntid"="64,2,1" {
; CHECK-LABEL: for function 'warp_index'
; CHECK: DIVERGENT: %thread =
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %uniform.warp = lshr i32 %thread, 5
  %uniform.masked = and i32 %thread, -32
  ret void
}

; A warp may straddle rows when the x dimension is not a multiple of 32.
define ptx_kernel void @straddling_rows() "nvvm.reqntid"="48,2,1" {
; CHECK-LABEL: for function 'straddling_rows'
; CHECK: DIVERGENT: %warp =
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %warp = lshr i32 %thread, 5
  ret void
}

; A maximum dimension does not establish the actual row boundaries.
define ptx_kernel void @maximum_only() "nvvm.maxntid"="64,2,1" {
; CHECK-LABEL: for function 'maximum_only'
; CHECK: DIVERGENT: %warp =
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %warp = lshr i32 %thread, 5
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.clock()
declare i32 @llvm.umin.i32(i32, i32)
declare i32 @llvm.nvvm.shfl.sync.idx.i32(i32, i32, i32, i32)
declare i32 @llvm.nvvm.shfl.sync.up.i32(i32, i32, i32, i32)
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32)
declare {i32, i1} @llvm.nvvm.shfl.sync.up.i32p(i32, i32, i32, i32)
declare i32 @llvm.nvvm.vote.ballot.sync(i32, i1)
declare i32 @llvm.nvvm.redux.sync.add(i32, i32)
declare i32 @unknown(i32)

; Ordinary arithmetic propagates divergence; warp collectives can remove it.
define ptx_kernel void @arithmetic(float %x, i32 %n, <4 x i32> %vector) {
; CHECK-LABEL: for function 'arithmetic'
; CHECK-DAG: DIVERGENT: %varying.sqrt =
; CHECK-DAG: DIVERGENT: %varying.random =
; CHECK-DAG: DIVERGENT: %rounded =
; CHECK-DAG: DIVERGENT: %exponent =
; CHECK-DAG: DIVERGENT: %dynamic.rounding =
; CHECK-DAG: DIVERGENT: %matrix =
; CHECK-DAG: DIVERGENT: %varying.repeat =
; CHECK-DAG: DIVERGENT: %varying.mask =
; CHECK-DAG: DIVERGENT: %varying.ue5m3 =
; CHECK-DAG: DIVERGENT: %varying.rounding =
; CHECK-DAG: DIVERGENT: %strict.rounding =
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %fp = uitofp i32 %thread to float
  %uniform.sqrt = call float @llvm.sqrt.f32(float %x)
  %varying.sqrt = call float @llvm.sqrt.f32(float %fp)
  %uniform.overflow = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %n, i32 1)
  %uniform.bits = call i32 @llvm.ctpop.i32(i32 %n)
  %uniform.vector = call <4 x i32> @llvm.vp.ctpop.v4i32(<4 x i32> %vector, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 4)
  %uniform.reverse = call <4 x i32> @llvm.experimental.vp.reverse.v4i32(<4 x i32> %vector, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, i32 4)
  %uniform.vector.sum = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %vector)
  %uniform.repeat = call <vscale x 4 x i32> @llvm.vector.repeat.nxv4i32.v4i32(<4 x i32> %vector)
  %varying.vector = insertelement <4 x i32> %vector, i32 %thread, i32 0
  %varying.repeat = call <vscale x 4 x i32> @llvm.vector.repeat.nxv4i32.v4i32(<4 x i32> %varying.vector)
  %uniform.mask.input = icmp ne <4 x i32> %vector, zeroinitializer
  %uniform.mask = call <4 x i1> @llvm.mask.beforefirst.v4i1(<4 x i1> %uniform.mask.input)
  %varying.mask.input = icmp ne <4 x i32> %varying.vector, zeroinitializer
  %varying.mask = call <4 x i1> @llvm.mask.beforefirst.v4i1(<4 x i1> %varying.mask.input)
  %uniform.ex2 = call float @llvm.nvvm.ex2.approx.f32(float %x)
  %uniform.permute = call i32 @llvm.nvvm.prmt(i32 %n, i32 0, i32 0)
  %uniform.random = call <2 x half> @llvm.nvvm.ff2f16x2.rs(float %x, float %x, i32 %n)
  %varying.random = call <2 x half> @llvm.nvvm.ff2f16x2.rs(float %x, float %x, i32 %thread)
  %uniform.ue5m3 = call i16 @llvm.nvvm.ff.to.ue5m3x2.rn(float %x, float %x)
  %uniform.ue5m3.unpack = call <2 x half> @llvm.nvvm.ue5m3x2.to.f16x2.rn(i16 %uniform.ue5m3)
  %varying.ue5m3 = call i16 @llvm.nvvm.ff.to.ue5m3x2.rn(float %x, float %fp)
  %rounded = call i32 @llvm.lround.i32.f32(float %x)
  %exponent = call {float, i32} @llvm.frexp.f32.i32(float %x)
  %uniform.exponent = call {float, i32} @llvm.frexp.f32.i32(float 2.0)
  %uniform.constrained = call float @llvm.experimental.constrained.fadd.f32(float %x, float %x, metadata !"round.tonearest", metadata !"fpexcept.ignore")
  %uniform.rounding.down = call float @llvm.experimental.constrained.fadd.f32(float %x, float %x, metadata !"round.downward", metadata !"fpexcept.ignore")
  %varying.rounding = call float @llvm.experimental.constrained.fadd.f32(float %x, float %fp, metadata !"round.downward", metadata !"fpexcept.ignore")
  %strict.rounding = call float @llvm.experimental.constrained.fadd.f32(float %x, float %x, metadata !"round.downward", metadata !"fpexcept.strict")
  %dynamic.rounding = call float @llvm.experimental.constrained.fadd.f32(float %x, float %x, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  ; A transpose redistributes packed components even with identical inputs.
  %matrix = call i32 @llvm.nvvm.movmatrix.sync.aligned.m8n8.trans.b16(i32 65538)
  ret void
}

define ptx_kernel void @state_queries(ptr %p, ptr addrspace(1) %global, i64 %state) {
; CHECK-LABEL: for function 'state_queries'
; CHECK-DAG: DIVERGENT: %physical.warp =
; CHECK-DAG: DIVERGENT: %physical.sm =
; CHECK-DAG: DIVERGENT: %active =
; CHECK-DAG: DIVERGENT: %barrier.layout =
  %uniform.smem = call i32 @llvm.nvvm.read.ptx.sreg.dynamic_smem_size()
  %uniform.space = call i1 @llvm.nvvm.isspacep.global(ptr %p)
  %uniform.pending = call i32 @llvm.nvvm.mbarrier.pending.count(i64 %state)
  %barrier.layout = call i1 @llvm.nvvm.mbarrier.check_layout.p0(ptr %p, i32 0)
  %uniform.ldu = call i32 @llvm.nvvm.ldu.global.i.i32.p1(ptr addrspace(1) %global, i32 4)
  %physical.warp = call i32 @llvm.nvvm.read.ptx.sreg.warpid()
  %physical.sm = call i32 @llvm.nvvm.read.ptx.sreg.smid()
  %active = call i32 @llvm.nvvm.activemask()
  ret void
}

define ptx_kernel void @aligned_rows() "nvvm.reqntid"="32,2,2" {
; CHECK-LABEL: for function 'aligned_rows'
  %uniform.y = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %uniform.z = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  ret void
}

define ptx_kernel void @straddling_planes() "nvvm.reqntid"="16,3,2" {
; CHECK-LABEL: for function 'straddling_planes'
; CHECK-DAG: DIVERGENT: %y =
; CHECK-DAG: DIVERGENT: %z =
  %y = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %z = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.laneid()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()
declare {i32, i1} @llvm.nvvm.shfl.sync.idx.i32p(i32, i32, i32, i32)
declare float @llvm.nvvm.redux.sync.fmax(float, i32)
declare {i32, i1} @llvm.nvvm.match.all.sync.i32p(i32, i32)
declare i32 @llvm.nvvm.match.any.sync.i32(i32, i32)
declare i32 @llvm.nvvm.barrier.cta.red.popc.count(i32, i32, i1)
declare float @llvm.sqrt.f32(float)
declare {i32, i1} @llvm.uadd.with.overflow.i32(i32, i32)
declare i32 @llvm.ctpop.i32(i32)
declare <4 x i32> @llvm.vp.ctpop.v4i32(<4 x i32>, <4 x i1>, i32)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>)
declare float @llvm.nvvm.ex2.approx.f32(float)
declare i32 @llvm.nvvm.prmt(i32, i32, i32)
declare <2 x half> @llvm.nvvm.ff2f16x2.rs(float, float, i32)
declare i32 @llvm.lround.i32.f32(float)
declare {float, i32} @llvm.frexp.f32.i32(float)
declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare i32 @llvm.nvvm.movmatrix.sync.aligned.m8n8.trans.b16(i32)
declare i32 @llvm.nvvm.read.ptx.sreg.dynamic_smem_size()
declare i1 @llvm.nvvm.isspacep.global(ptr)
declare i32 @llvm.nvvm.mbarrier.pending.count(i64)
declare i32 @llvm.nvvm.ldu.global.i.i32.p1(ptr addrspace(1), i32)
declare i32 @llvm.nvvm.read.ptx.sreg.warpid()
declare i32 @llvm.nvvm.read.ptx.sreg.smid()
declare i32 @llvm.nvvm.activemask()

; Threads terminate at different iterations; later ballots exclude them.
; A uniform result within the loop is divergent when used on different exits.
define ptx_kernel void @collective_loop_exit(ptr %out) {
; CHECK-LABEL: for function 'collective_loop_exit'
; CHECK: DIVERGENT: %escaped =
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  br label %loop
loop:
  %iteration = phi i32 [0, %entry], [%next, %loop]
  %uniform.ballot = call i32 @llvm.nvvm.vote.ballot.sync(i32 -1, i1 true)
  %next = add i32 %iteration, 1
  %done = icmp eq i32 %iteration, %lane
  br i1 %done, label %exit, label %loop
exit:
  %escaped = add i32 %uniform.ballot, 0
  %address = getelementptr i32, ptr %out, i32 %lane
  store i32 %escaped, ptr %address
  ret void
}

declare <4 x i32> @llvm.experimental.vp.reverse.v4i32(<4 x i32>, <4 x i1>, i32)

declare i32 @llvm.nvvm.shfl.up.i32(i32, i32, i32)
declare {i32, i1} @llvm.nvvm.shfl.up.i32p(i32, i32, i32)
declare i1 @llvm.nvvm.vote.all(i1)
declare i32 @llvm.nvvm.vote.ballot(i1)
declare <vscale x 4 x i32> @llvm.vector.repeat.nxv4i32.v4i32(<4 x i32>)
declare <4 x i1> @llvm.mask.beforefirst.v4i1(<4 x i1>)
declare i16 @llvm.nvvm.ff.to.ue5m3x2.rn(float, float)
declare <2 x half> @llvm.nvvm.ue5m3x2.to.f16x2.rn(i16)
declare i1 @llvm.nvvm.mbarrier.check_layout.p0(ptr, i32)

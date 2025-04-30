; RUN: llc -mtriple=next32 -mcpu=next32gen2 < %s | FileCheck %s

define void @store_masked_v2i8(ptr nocapture noundef %p, i32 %mask) {
; CHECK-LABEL: store_masked_v2i8:
; CHECK:       feeder.32	tid
; CHECK-NEXT:  feeder.32	ret_fid
; CHECK-NEXT:  feeder.64	r1
; CHECK-NEXT:  feeder.64	r2
; CHECK-NEXT:  feeder.32	r3
; CHECK-NEXT:  movl	r4, 0x0
; CHECK-NEXT:  vmemwrite.2.8.align[1]	r2, r1, tid, r3
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  dup r1, tid
; CHECK-NEXT:  barrier r1, r4
entry:
  %truncmask = trunc i32 %mask to i2
  %bitmask = bitcast i2 %truncmask to <2 x i1>
  tail call void @llvm.masked.store.v2i8.p0(<2 x i8> zeroinitializer, ptr %p, i32 1, <2 x i1> %bitmask)
  ret void
}

define void @store_masked_v16i8(ptr nocapture noundef %p, i32 %mask) {
; CHECK-LABEL: store_masked_v16i8:
; CHECK:       feeder.32	tid
; CHECK-NEXT:  feeder.32	ret_fid
; CHECK-NEXT:  feeder.64	r1
; CHECK-NEXT:  feeder.64	r2
; CHECK-NEXT:  feeder.32	r3
; CHECK-NEXT:  movl	r4, 0x0
; CHECK-NEXT:  vmemwrite.16.8.align[1]	r2, r1, tid, r3
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  dup r1, tid
; CHECK-NEXT:  barrier r1, r4
entry:
  %truncmask = trunc i32 %mask to i16
  %bitmask = bitcast i16 %truncmask to <16 x i1>
  tail call void @llvm.masked.store.v16i8.p0(<16 x i8> zeroinitializer, ptr %p, i32 1, <16 x i1> %bitmask)
  ret void
}

define void @store_masked_v2i16(ptr nocapture noundef %p, i32 %mask) {
; CHECK-LABEL: store_masked_v2i16:
; CHECK:       feeder.32	tid
; CHECK-NEXT:  feeder.32	ret_fid
; CHECK-NEXT:  feeder.64	r1
; CHECK-NEXT:  feeder.64	r2
; CHECK-NEXT:  feeder.32	r3
; CHECK-NEXT:  movl	r4, 0x0
; CHECK-NEXT:  vmemwrite.2.16.align[1]	r2, r1, tid, r3
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  dup r1, tid
; CHECK-NEXT:  barrier r1, r4
entry:
  %truncmask = trunc i32 %mask to i2
  %bitmask = bitcast i2 %truncmask to <2 x i1>
  tail call void @llvm.masked.store.v2i16.p0(<2 x i16> zeroinitializer, ptr %p, i32 1, <2 x i1> %bitmask)
  ret void
}

define void @store_masked_v16i16(ptr nocapture noundef %p, i32 %mask) {
; CHECK-LABEL: store_masked_v16i16:
; CHECK:       feeder.32	tid
; CHECK-NEXT:  feeder.32	ret_fid
; CHECK-NEXT:  feeder.64	r1
; CHECK-NEXT:  feeder.64	r2
; CHECK-NEXT:  feeder.32	r3
; CHECK-NEXT:  movl	r4, 0x0
; CHECK-NEXT:  vmemwrite.16.16.align[1]	r2, r1, tid, r3
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  dup r1, tid
; CHECK-NEXT:  barrier r1, r4
entry:
  %truncmask = trunc i32 %mask to i16
  %bitmask = bitcast i16 %truncmask to <16 x i1>
  tail call void @llvm.masked.store.v16i16.p0(<16 x i16> zeroinitializer, ptr %p, i32 1, <16 x i1> %bitmask)
  ret void
}

define void @store_masked_v2i32(ptr nocapture noundef %p, i32 %mask) {
; CHECK-LABEL: store_masked_v2i32:
; CHECK:       feeder.32	tid
; CHECK-NEXT:  feeder.32	ret_fid
; CHECK-NEXT:  feeder.64	r1
; CHECK-NEXT:  feeder.64	r2
; CHECK-NEXT:  feeder.32	r3
; CHECK-NEXT:  movl	r4, 0x0
; CHECK-NEXT:  vmemwrite.2.32.align[1]	r2, r1, tid, r3
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  dup r1, tid
; CHECK-NEXT:  barrier r1, r4
entry:
  %truncmask = trunc i32 %mask to i2
  %bitmask = bitcast i2 %truncmask to <2 x i1>
  tail call void @llvm.masked.store.v2i32.p0(<2 x i32> zeroinitializer, ptr %p, i32 1, <2 x i1> %bitmask)
  ret void
}

define void @store_masked_v16i32(ptr nocapture noundef %p, i32 %mask) {
; CHECK-LABEL: store_masked_v16i32
; CHECK:       feeder.32	tid
; CHECK-NEXT:  feeder.32	ret_fid
; CHECK-NEXT:  feeder.64	r1
; CHECK-NEXT:  feeder.64	r2
; CHECK-NEXT:  feeder.32	r3
; CHECK-NEXT:  movl	r4, 0x0
; CHECK-NEXT:  vmemwrite.16.32.align[1]	r2, r1, tid, r3
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  dup r1, tid
; CHECK-NEXT:  barrier r1, r4
entry:
  %truncmask = trunc i32 %mask to i16
  %bitmask = bitcast i16 %truncmask to <16 x i1>
  tail call void @llvm.masked.store.v16i32.p0(<16 x i32> zeroinitializer, ptr %p, i32 1, <16 x i1> %bitmask)
  ret void
}

define void @store_masked_v2i64(ptr nocapture noundef %p, i32 %mask) {
; CHECK-LABEL: store_masked_v2i64:
; CHECK:       feeder.32	tid
; CHECK-NEXT:  feeder.32	ret_fid
; CHECK-NEXT:  feeder.64	r1
; CHECK-NEXT:  feeder.64	r2
; CHECK-NEXT:  feeder.32	r3
; CHECK-NEXT:  movl	r4, 0x0
; CHECK-NEXT:  vmemwrite.2.64.align[1]	r2, r1, tid, r3
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  dup r1, tid
; CHECK-NEXT:  barrier r1, r4
entry:
  %truncmask = trunc i32 %mask to i2
  %bitmask = bitcast i2 %truncmask to <2 x i1>
  tail call void @llvm.masked.store.v2i64.p0(<2 x i64> zeroinitializer, ptr %p, i32 1, <2 x i1> %bitmask)
  ret void
}

define void @store_masked_v8i64(ptr nocapture noundef %p, i32 %mask) {
; CHECK-LABEL: store_masked_v8i64
; CHECK:       feeder.32	tid
; CHECK-NEXT:  feeder.32	ret_fid
; CHECK-NEXT:  feeder.64	r1
; CHECK-NEXT:  feeder.64	r2
; CHECK-NEXT:  feeder.32	r3
; CHECK-NEXT:  movl	r4, 0x0
; CHECK-NEXT:  vmemwrite.8.64.align[1]	r2, r1, tid, r3
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  memdata r4
; CHECK-NEXT:  dup r1, tid
; CHECK-NEXT:  barrier r1, r4
entry:
  %truncmask = trunc i32 %mask to i8
  %bitmask = bitcast i8 %truncmask to <8 x i1>
  tail call void @llvm.masked.store.v8i64.p0(<8 x i64> zeroinitializer, ptr %p, i32 1, <8 x i1> %bitmask)
  ret void
}

define <2 x i8> @load_masked_v2i8_pt_undef(ptr nocapture noundef readonly %p, i32 %mask) {
; CHECK-LABEL: load_masked_v2i8_pt_undef:
; CHECK:      feeder.32       tid
; CHECK-NEXT: feeder.32       ret_fid
; CHECK-NEXT: feeder.64       r1
; CHECK-NEXT: feeder.64       r2
; CHECK-NEXT: feeder.32       r3
; CHECK-NEXT: dup     r4, tid
; CHECK-NEXT: vmemread.2.8.align[8]  r2, r1, tid, r3
; CHECK-NEXT: memdata r1
; CHECK-NEXT: barrier r4, r1
entry:
  %truncmask = trunc i32 %mask to i2
  %bitmask = bitcast i2 %truncmask to <2 x i1>
  %val = tail call <2 x i8> @llvm.masked.load.v2i8.p0(ptr %p, i32 8, <2 x i1> %bitmask, <2 x i8> undef)
  ret <2 x i8> %val
}

define <16 x i8> @load_masked_v16i8_pt_undef(ptr nocapture noundef readonly %p, i32 %mask) {
; CHECK-LABEL: load_masked_v16i8_pt_undef:
; CHECK:      feeder.32       tid
; CHECK-NEXT: feeder.32       ret_fid
; CHECK-NEXT: feeder.64       r1
; CHECK-NEXT: feeder.64       r2
; CHECK-NEXT: feeder.32       r3
; CHECK-NEXT: dup     r4, tid
; CHECK-NEXT: vmemread.16.8.align[8]  r2, r1, tid, r3
; CHECK-NEXT: memdata r1
; CHECK-NEXT: memdata r2
; CHECK-NEXT: memdata r3
; CHECK-NEXT: memdata r5
; CHECK-NEXT: barrier r4, r1
entry:
  %truncmask = trunc i32 %mask to i16
  %bitmask = bitcast i16 %truncmask to <16 x i1>
  %val = tail call <16 x i8> @llvm.masked.load.v16i8.p0(ptr %p, i32 8, <16 x i1> %bitmask, <16 x i8> undef)
  ret <16 x i8> %val
}

define <2 x i16> @load_masked_v2i16_pt_undef(ptr nocapture noundef readonly %p, i32 %mask) {
; CHECK-LABEL: load_masked_v2i16_pt_undef:
; CHECK:      feeder.32       tid
; CHECK-NEXT: feeder.32       ret_fid
; CHECK-NEXT: feeder.64       r1
; CHECK-NEXT: feeder.64       r2
; CHECK-NEXT: feeder.32       r3
; CHECK-NEXT: dup     r4, tid
; CHECK-NEXT: vmemread.2.16.align[8]  r2, r1, tid, r3
; CHECK-NEXT: memdata r1
; CHECK-NEXT: barrier r4, r1
entry:
  %truncmask = trunc i32 %mask to i2
  %bitmask = bitcast i2 %truncmask to <2 x i1>
  %val = tail call <2 x i16> @llvm.masked.load.v2i16.p0(ptr %p, i32 8, <2 x i1> %bitmask, <2 x i16> undef)
  ret <2 x i16> %val
}

define <16 x i16> @load_masked_v16i16_pt_undef(ptr nocapture noundef readonly %p, i32 %mask) {
; CHECK-LABEL: load_masked_v16i16_pt_undef:
; CHECK:      feeder.32       tid
; CHECK-NEXT: feeder.32       ret_fid
; CHECK-NEXT: feeder.64       r1
; CHECK-NEXT: feeder.64       r2
; CHECK-NEXT: feeder.32       r3
; CHECK-NEXT: dup     r4, tid
; CHECK-NEXT: vmemread.16.16.align[8]  r2, r1, tid, r3
; CHECK-NEXT: memdata r1
; CHECK-NEXT: memdata r2
; CHECK-NEXT: memdata r3
; CHECK-NEXT: memdata r5
; CHECK-NEXT: memdata r6
; CHECK-NEXT: memdata r7
; CHECK-NEXT: memdata r8
; CHECK-NEXT: memdata r9
; CHECK-NEXT: barrier r4, r1
entry:
  %truncmask = trunc i32 %mask to i16
  %bitmask = bitcast i16 %truncmask to <16 x i1>
  %val = tail call <16 x i16> @llvm.masked.load.v16i16.p0(ptr %p, i32 8, <16 x i1> %bitmask, <16 x i16> undef)
  ret <16 x i16> %val
}

define <2 x i32> @load_masked_v2i32_pt_undef(ptr nocapture noundef readonly %p, i32 %mask) {
; CHECK-LABEL: load_masked_v2i32_pt_undef:
; CHECK:      feeder.32       tid
; CHECK-NEXT: feeder.32       ret_fid
; CHECK-NEXT: feeder.64       r1
; CHECK-NEXT: feeder.64       r2
; CHECK-NEXT: feeder.32       r3
; CHECK-NEXT: dup     r4, tid
; CHECK-NEXT: vmemread.2.32.align[8]  r2, r1, tid, r3
; CHECK-NEXT: memdata r1
; CHECK-NEXT: memdata r2
; CHECK-NEXT: barrier r4, r1
entry:
  %truncmask = trunc i32 %mask to i2
  %bitmask = bitcast i2 %truncmask to <2 x i1>
  %val = tail call <2 x i32> @llvm.masked.load.v2i32.p0(ptr %p, i32 8, <2 x i1> %bitmask, <2 x i32> undef)
  ret <2 x i32> %val
}

define <16 x i32> @load_masked_v16i32_pt_undef(ptr nocapture noundef readonly %p, i32 %mask) {
; CHECK-LABEL: load_masked_v16i32_pt_undef:
; CHECK:      feeder.32       tid
; CHECK-NEXT: feeder.32       ret_fid
; CHECK-NEXT: feeder.64       r1
; CHECK-NEXT: feeder.64       r2
; CHECK-NEXT: feeder.32       r3
; CHECK-NEXT: dup     r4, tid
; CHECK-NEXT: vmemread.16.32.align[8]  r2, r1, tid, r3
; CHECK-NEXT: memdata r1
; CHECK-NEXT: memdata r2
; CHECK-NEXT: memdata r3
; CHECK-NEXT: memdata r5
; CHECK-NEXT: memdata r6
; CHECK-NEXT: memdata r7
; CHECK-NEXT: memdata r8
; CHECK-NEXT: memdata r9
; CHECK-NEXT: memdata r10
; CHECK-NEXT: memdata r11
; CHECK-NEXT: memdata r12
; CHECK-NEXT: memdata r13
; CHECK-NEXT: memdata r14
; CHECK-NEXT: memdata r15
; CHECK-NEXT: memdata r16
; CHECK-NEXT: memdata r17
; CHECK-NEXT: barrier r4, r1
entry:
  %truncmask = trunc i32 %mask to i16
  %bitmask = bitcast i16 %truncmask to <16 x i1>
  %val = tail call <16 x i32> @llvm.masked.load.v16i32.p0(ptr %p, i32 8, <16 x i1> %bitmask, <16 x i32> undef)
  ret <16 x i32> %val
}

define <2 x i32> @load_masked_v2i32_pt_dynamic(ptr nocapture noundef readonly %p, i32 %mask, i64 %dynpt) {
; CHECK-LABEL: load_masked_v2i32_pt_dynamic:
; CHECK:      feeder.32       tid
; CHECK-NEXT: feeder.32       ret_fid
; CHECK-NEXT: feeder.64       r1
; CHECK-NEXT: feeder.64       r2
; CHECK-NEXT: feeder.32       r3
; CHECK-NEXT: feeder.64       r4
; CHECK-NEXT: feeder.64       r5
; CHECK-NEXT: movl    r6, 0x2
; CHECK-NEXT: dup     r7, r3
; CHECK-NEXT: and     r7, r6
; CHECK-NEXT: movl    r6, 0x1
; CHECK-NEXT: shr     r7, r6
; CHECK-NEXT: movl    r8, 0x0
; CHECK-NEXT: sub     r7, r8
; CHECK-NEXT: dup     r9, r7
; CHECK-NEXT: flags   r9
; CHECK-NEXT: dup     r7, r3
; CHECK-NEXT: and     r7, r6
; CHECK-NEXT: sub     r7, r8
; CHECK-NEXT: dup     r6, r7
; CHECK-NEXT: flags   r6
; CHECK-NEXT: vmemread.2.32.align[8]  r2, r1, tid, r3
; CHECK-NEXT: memdata r1
; CHECK-NEXT: memdata r2
; CHECK-NEXT: select.ne       r4, r2 [r6]
; CHECK-NEXT: select.ne       r5, r1 [r9]
; CHECK-NEXT: dup     r2, tid
; CHECK-NEXT: barrier r2, r1
entry:
  %truncmask = trunc i32 %mask to i2
  %bitmask = bitcast i2 %truncmask to <2 x i1>
  %dynpt_vec = bitcast i64 %dynpt to <2 x i32>
  %val = tail call <2 x i32> @llvm.masked.load.v2i32.p0(ptr %p, i32 8, <2 x i1> %bitmask, <2 x i32> %dynpt_vec)
  ret <2 x i32> %val
}

define <2 x i64> @load_masked_v2i64_pt_undef(ptr nocapture noundef readonly %p, i32 %mask) {
; CHECK-LABEL: load_masked_v2i64_pt_undef:
; CHECK:      feeder.32       tid
; CHECK-NEXT: feeder.32       ret_fid
; CHECK-NEXT: feeder.64       r1
; CHECK-NEXT: feeder.64       r2
; CHECK-NEXT: feeder.32       r3
; CHECK-NEXT: dup     r4, tid
; CHECK-NEXT: vmemread.2.64.align[8]  r2, r1, tid, r3
; CHECK-NEXT: memdata r1
; CHECK-NEXT: memdata r2
; CHECK-NEXT: memdata r3
; CHECK-NEXT: memdata r5
; CHECK-NEXT: barrier r4, r1
entry:
  %truncmask = trunc i32 %mask to i2
  %bitmask = bitcast i2 %truncmask to <2 x i1>
  %val = tail call <2 x i64> @llvm.masked.load.v2i64.p0(ptr %p, i32 8, <2 x i1> %bitmask, <2 x i64> undef)
  ret <2 x i64> %val
}

define <8 x i64> @load_masked_v8i64_pt_undef(ptr nocapture noundef readonly %p, i32 %mask) {
; CHECK-LABEL: load_masked_v8i64_pt_undef:
; CHECK:      feeder.32       tid
; CHECK-NEXT: feeder.32       ret_fid
; CHECK-NEXT: feeder.64       r1
; CHECK-NEXT: feeder.64       r2
; CHECK-NEXT: feeder.32       r3
; CHECK-NEXT: dup     r4, tid
; CHECK-NEXT: vmemread.8.64.align[8]  r2, r1, tid, r3
; CHECK-NEXT: memdata r1
; CHECK-NEXT: memdata r2
; CHECK-NEXT: memdata r3
; CHECK-NEXT: memdata r5
; CHECK-NEXT: memdata r6
; CHECK-NEXT: memdata r7
; CHECK-NEXT: memdata r8
; CHECK-NEXT: memdata r9
; CHECK-NEXT: memdata r10
; CHECK-NEXT: memdata r11
; CHECK-NEXT: memdata r12
; CHECK-NEXT: memdata r13
; CHECK-NEXT: memdata r14
; CHECK-NEXT: memdata r15
; CHECK-NEXT: memdata r16
; CHECK-NEXT: memdata r17
; CHECK-NEXT: barrier r4, r1
entry:
  %truncmask = trunc i32 %mask to i8
  %bitmask = bitcast i8 %truncmask to <8 x i1>
  %val = tail call <8 x i64> @llvm.masked.load.v8i64.p0(ptr %p, i32 8, <8 x i1> %bitmask, <8 x i64> undef)
  ret <8 x i64> %val
}

define <8 x i1> @active_lane_mask(i64 %index, i64 %trip_count) {
; CHECK-LABEL: active_lane_mask:
; CHECK:      sub     r3, r1
; CHECK-NEXT: dup     r1, r3
; CHECK-NEXT: flags   r1
; CHECK-NEXT: umin    r3, r5
; CHECK-NEXT: sbb     r4, r2 [r1]
; CHECK-NEXT: movl    r1, 0x0
; CHECK-NEXT: sub     r4, r1
; CHECK-NEXT: dup     r1, r4
; CHECK-NEXT: flags   r1
; CHECK-NEXT: select.e        r5, r3 [r1]
; CHECK-NEXT: movl    r1, 0xFFFFFFFF
; CHECK-NEXT: shl     r1, r5
; CHECK-NEXT: not     r1
entry:
  %active.lane.mask = call <8 x i1> @llvm.get.active.lane.mask.v8i1.i64(i64 %index, i64 %trip_count)
  ret <8 x i1> %active.lane.mask
}

declare void @llvm.masked.store.v2i8.p0(<2 x i8>, ptr nocapture, i32 immarg, <2 x i1>)
declare void @llvm.masked.store.v16i8.p0(<16 x i8>, ptr nocapture, i32 immarg, <16 x i1>)
declare void @llvm.masked.store.v2i16.p0(<2 x i16>, ptr nocapture, i32 immarg, <2 x i1>)
declare void @llvm.masked.store.v16i16.p0(<16 x i16>, ptr nocapture, i32 immarg, <16 x i1>)
declare void @llvm.masked.store.v2i32.p0(<2 x i32>, ptr nocapture, i32 immarg, <2 x i1>)
declare void @llvm.masked.store.v16i32.p0(<16 x i32>, ptr nocapture, i32 immarg, <16 x i1>)
declare void @llvm.masked.store.v2i64.p0(<2 x i64>, ptr nocapture, i32 immarg, <2 x i1>)
declare void @llvm.masked.store.v8i64.p0(<8 x i64>, ptr nocapture, i32 immarg, <8 x i1>)

declare <2 x i8> @llvm.masked.load.v2i8.p0(ptr, i32, <2 x i1>, <2 x i8>)
declare <16 x i8> @llvm.masked.load.v16i8.p0(ptr, i32, <16 x i1>, <16 x i8>)
declare <2 x i16> @llvm.masked.load.v2i16.p0(ptr, i32, <2 x i1>, <2 x i16>)
declare <16 x i16> @llvm.masked.load.v16i16.p0(ptr, i32, <16 x i1>, <16 x i16>)
declare <2 x i32> @llvm.masked.load.v2i32.p0(ptr, i32, <2 x i1>, <2 x i32>)
declare <16 x i32> @llvm.masked.load.v16i32.p0(ptr, i32, <16 x i1>, <16 x i32>)
declare <2 x i64> @llvm.masked.load.v2i64.p0(ptr, i32, <2 x i1>, <2 x i64>)
declare <8 x i64> @llvm.masked.load.v8i64.p0(ptr, i32, <8 x i1>, <8 x i64>)

declare <8 x i1> @llvm.get.active.lane.mask.v8i1.i64(i64, i64)

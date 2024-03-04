; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z16 -O3 -print-before=peephole-opt \
; RUN:   -print-after=peephole-opt,systemz-finalize-regmem -verify-machineinstrs 2>&1 \
; RUN:   | FileCheck %s

define void @f0(float %a1, ptr %src1, ptr %src2, ptr %src3, ptr %dst) {
; CHECK:       # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f0: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       [[LD1:%[0-9]+:vr32bit]] = VL32 [[ADDR1:%[0-9]+:addr64bit]], 0, $noreg :: (load (s32) from %ir.src1)
; CHECK-NEXT:  vr32bit = nofpexcept WFASB_CCPseudo %0:fp32bit, killed [[LD1]], implicit-def dead $cc, implicit $fpc
; CHECK:       [[LD2:%[0-9]+:vr32bit]] = VL32 %2:addr64bit, 0, $noreg :: (load (s32) from %ir.src2)
; CHECK-NEXT:  vr32bit = nofpexcept WFASB_CCPseudo %0:fp32bit, [[LD2]], implicit-def dead $cc, implicit $fpc
; CHECK-NEXT:  VST32 [[LD2]], %4:addr64bit, 0, $noreg :: (volatile store (s32) into %ir.dst)

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f0: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       fp32bit = nofpexcept AEB %0:fp32bit(tied-def 0), [[ADDR1]], 0, $noreg, implicit-def dead $cc, implicit $fpc :: (load (s32) from %ir.src1)
; CHECK:       %8:vr32bit = nofpexcept WFASB_CCPseudo %0:fp32bit, [[LD2]], implicit-def dead $cc, implicit $fpc

; CHECK:       # *** IR Dump After SystemZ Finalize RegMem (systemz-finalize-regmem) ***:
; CHECK-NEXT:  # Machine code for function f0: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       fp32bit = nofpexcept AEB %0:fp32bit(tied-def 0), [[ADDR1]], 0, $noreg, implicit-def dead $cc, implicit $fpc :: (load (s32) from %ir.src1)
; CHECK:       %8:vr32bit = nofpexcept WFASB %0:fp32bit, [[LD2]], implicit $fpc

  %l1 = load float, ptr %src1
  %res1 = fadd float %a1, %l1
  store volatile float %res1, ptr %dst

  %l2 = load float, ptr %src2
  %res2 = fadd float %a1, %l2
  store volatile float %l2, ptr %dst
  store volatile float %res2, ptr %dst

  ret void
}

define void @f1(double %a1, ptr %src1, ptr %src2, ptr %src3, ptr %dst) {
; CHECK:       # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f1: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       [[LD1:%[0-9]+:vr64bit]] = VL64 [[ADDR1:%[0-9]+:addr64bit]], 0, $noreg :: (load (s64) from %ir.src1)
; CHECK-NEXT:  vr64bit = nofpexcept WFADB_CCPseudo %0:fp64bit, killed [[LD1]], implicit-def dead $cc, implicit $fpc
; CHECK:       [[LD2:%[0-9]+:vr64bit]] = VL64 %2:addr64bit, 0, $noreg :: (load (s64) from %ir.src2)
; CHECK-NEXT:  vr64bit = nofpexcept WFADB_CCPseudo %0:fp64bit, [[LD2]], implicit-def dead $cc, implicit $fpc
; CHECK-NEXT:  VST64 [[LD2]], %4:addr64bit, 0, $noreg :: (volatile store (s64) into %ir.dst)

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f1: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       fp64bit = nofpexcept ADB %0:fp64bit(tied-def 0), [[ADDR1]], 0, $noreg, implicit-def dead $cc, implicit $fpc :: (load (s64) from %ir.src1)
; CHECK:       %8:vr64bit = nofpexcept WFADB_CCPseudo %0:fp64bit, [[LD2]], implicit-def dead $cc, implicit $fpc

; CHECK:       # *** IR Dump After SystemZ Finalize RegMem (systemz-finalize-regmem) ***:
; CHECK-NEXT:  # Machine code for function f1: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       fp64bit = nofpexcept ADB %0:fp64bit(tied-def 0), [[ADDR1]], 0, $noreg, implicit-def dead $cc, implicit $fpc :: (load (s64) from %ir.src1)
; CHECK:       %8:vr64bit = nofpexcept WFADB %0:fp64bit, [[LD2]], implicit $fpc

  %l1 = load double, ptr %src1
  %res1 = fadd double %a1, %l1
  store volatile double %res1, ptr %dst

  %l2 = load double, ptr %src2
  %res2 = fadd double %a1, %l2
  store volatile double %l2, ptr %dst
  store volatile double %res2, ptr %dst

  ret void
}

define void @f2(float %a1, ptr %src1, ptr %src2, ptr %src3, ptr %dst) {
; CHECK:       # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f2: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       [[LD1:%[0-9]+:vr32bit]] = VL32 [[ADDR1:%[0-9]+:addr64bit]], 0, $noreg :: (load (s32) from %ir.src1)
; CHECK-NEXT:  vr32bit = nofpexcept WFSSB_CCPseudo %0:fp32bit, killed [[LD1]], implicit-def dead $cc, implicit $fpc
; CHECK:       [[LD2:%[0-9]+:vr32bit]] = VL32 %2:addr64bit, 0, $noreg :: (load (s32) from %ir.src2)
; CHECK-NEXT:  vr32bit = nofpexcept WFSSB_CCPseudo %0:fp32bit, [[LD2]], implicit-def dead $cc, implicit $fpc
; CHECK-NEXT:  VST32 [[LD2]], %4:addr64bit, 0, $noreg :: (volatile store (s32) into %ir.dst)

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f2: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       fp32bit = nofpexcept SEB %0:fp32bit(tied-def 0), [[ADDR1]], 0, $noreg, implicit-def dead $cc, implicit $fpc :: (load (s32) from %ir.src1)
; CHECK:       %8:vr32bit = nofpexcept WFSSB_CCPseudo %0:fp32bit, [[LD2]], implicit-def dead $cc, implicit $fpc

; CHECK:       # *** IR Dump After SystemZ Finalize RegMem (systemz-finalize-regmem) ***:
; CHECK-NEXT:  # Machine code for function f2: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       fp32bit = nofpexcept SEB %0:fp32bit(tied-def 0), [[ADDR1]], 0, $noreg, implicit-def dead $cc, implicit $fpc :: (load (s32) from %ir.src1)
; CHECK:       %8:vr32bit = nofpexcept WFSSB %0:fp32bit, [[LD2]], implicit $fpc

  %l1 = load float, ptr %src1
  %res1 = fsub float %a1, %l1
  store volatile float %res1, ptr %dst

  %l2 = load float, ptr %src2
  %res2 = fsub float %a1, %l2
  store volatile float %l2, ptr %dst
  store volatile float %res2, ptr %dst

  ret void
}

define void @f3(double %a1, ptr %src1, ptr %src2, ptr %src3, ptr %dst) {
; CHECK:       # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f3: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       [[LD1:%[0-9]+:vr64bit]] = VL64 [[ADDR1:%[0-9]+:addr64bit]], 0, $noreg :: (load (s64) from %ir.src1)
; CHECK-NEXT:  vr64bit = nofpexcept WFSDB_CCPseudo %0:fp64bit, killed [[LD1]], implicit-def dead $cc, implicit $fpc
; CHECK:       [[LD2:%[0-9]+:vr64bit]] = VL64 %2:addr64bit, 0, $noreg :: (load (s64) from %ir.src2)
; CHECK-NEXT:  vr64bit = nofpexcept WFSDB_CCPseudo %0:fp64bit, [[LD2]], implicit-def dead $cc, implicit $fpc
; CHECK-NEXT:  VST64 [[LD2]], %4:addr64bit, 0, $noreg :: (volatile store (s64) into %ir.dst)

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f3: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       fp64bit = nofpexcept SDB %0:fp64bit(tied-def 0), [[ADDR1]], 0, $noreg, implicit-def dead $cc, implicit $fpc :: (load (s64) from %ir.src1)
; CHECK:       %8:vr64bit = nofpexcept WFSDB_CCPseudo %0:fp64bit, [[LD2]], implicit-def dead $cc, implicit $fpc

; CHECK:       # *** IR Dump After SystemZ Finalize RegMem (systemz-finalize-regmem) ***:
; CHECK-NEXT:  # Machine code for function f3: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       fp64bit = nofpexcept SDB %0:fp64bit(tied-def 0), [[ADDR1]], 0, $noreg, implicit-def dead $cc, implicit $fpc :: (load (s64) from %ir.src1)
; CHECK:       %8:vr64bit = nofpexcept WFSDB %0:fp64bit, [[LD2]], implicit $fpc

  %l1 = load double, ptr %src1
  %res1 = fsub double %a1, %l1
  store volatile double %res1, ptr %dst

  %l2 = load double, ptr %src2
  %res2 = fsub double %a1, %l2
  store volatile double %l2, ptr %dst
  store volatile double %res2, ptr %dst

  ret void
}

define void @f4(float %a1, ptr %src1, ptr %src2, ptr %src3, ptr %dst) {
; CHECK-LABEL: # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f4: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       [[LD1:%[0-9]+:vr32bit]] = VL32 [[ADDR1:%[0-9]+:addr64bit]], 0, $noreg :: (load (s32) from %ir.src1)
; CHECK-NEXT:  vr32bit = nofpexcept WFMSB %0:fp32bit, killed [[LD1]], implicit $fpc
; CHECK:       [[LD2:%[0-9]+:vr32bit]] = VL32 %2:addr64bit, 0, $noreg :: (load (s32) from %ir.src2)
; CHECK-NEXT:  vr32bit = nofpexcept WFMSB %0:fp32bit, [[LD2]], implicit $fpc
; CHECK-NEXT:  VST32 [[LD2]], %4:addr64bit, 0, $noreg :: (volatile store (s32) into %ir.dst)

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f4: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       fp32bit = nofpexcept MEEB %0:fp32bit(tied-def 0), [[ADDR1]], 0, $noreg, implicit $fpc :: (load (s32) from %ir.src1)
; CHECK:       %8:vr32bit = nofpexcept WFMSB %0:fp32bit, [[LD2]], implicit $fpc

  %l1 = load float, ptr %src1
  %res1 = fmul float %a1, %l1
  store volatile float %res1, ptr %dst

  %l2 = load float, ptr %src2
  %res2 = fmul float %a1, %l2
  store volatile float %l2, ptr %dst
  store volatile float %res2, ptr %dst

  ret void
}

define void @f5(double %a1, ptr %src1, ptr %src2, ptr %src3, ptr %dst) {
; CHECK-LABEL: # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f5: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       [[LD1:%[0-9]+:vr64bit]] = VL64 [[ADDR1:%[0-9]+:addr64bit]], 0, $noreg :: (load (s64) from %ir.src1)
; CHECK-NEXT:  vr64bit = nofpexcept WFMDB %0:fp64bit, killed [[LD1]], implicit $fpc
; CHECK:       [[LD2:%[0-9]+:vr64bit]] = VL64 %2:addr64bit, 0, $noreg :: (load (s64) from %ir.src2)
; CHECK-NEXT:  vr64bit = nofpexcept WFMDB %0:fp64bit, [[LD2]], implicit $fpc
; CHECK-NEXT:  VST64 [[LD2]], %4:addr64bit, 0, $noreg :: (volatile store (s64) into %ir.dst)

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f5: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       fp64bit = nofpexcept MDB %0:fp64bit(tied-def 0), [[ADDR1]], 0, $noreg, implicit $fpc :: (load (s64) from %ir.src1)
; CHECK:       %8:vr64bit = nofpexcept WFMDB %0:fp64bit, [[LD2]], implicit $fpc

  %l1 = load double, ptr %src1
  %res1 = fmul double %a1, %l1
  store volatile double %res1, ptr %dst

  %l2 = load double, ptr %src2
  %res2 = fmul double %a1, %l2
  store volatile double %l2, ptr %dst
  store volatile double %res2, ptr %dst

  ret void
}

define void @f6(float %a1, ptr %src1, ptr %src2, ptr %src3, ptr %dst) {
; CHECK-LABEL: # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f6: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       [[LD1:%[0-9]+:vr32bit]] = VL32 [[ADDR1:%[0-9]+:addr64bit]], 0, $noreg :: (load (s32) from %ir.src1)
; CHECK-NEXT:  vr32bit = nofpexcept WFDSB %0:fp32bit, killed [[LD1]], implicit $fpc
; CHECK:       [[LD2:%[0-9]+:vr32bit]] = VL32 %2:addr64bit, 0, $noreg :: (load (s32) from %ir.src2)
; CHECK-NEXT:  vr32bit = nofpexcept WFDSB %0:fp32bit, [[LD2]], implicit $fpc
; CHECK-NEXT:  VST32 [[LD2]], %4:addr64bit, 0, $noreg :: (volatile store (s32) into %ir.dst)

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f6: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       fp32bit = nofpexcept DEB %0:fp32bit(tied-def 0), [[ADDR1]], 0, $noreg, implicit $fpc :: (load (s32) from %ir.src1)
; CHECK:       %8:vr32bit = nofpexcept WFDSB %0:fp32bit, [[LD2]], implicit $fpc

  %l1 = load float, ptr %src1
  %res1 = fdiv float %a1, %l1
  store volatile float %res1, ptr %dst

  %l2 = load float, ptr %src2
  %res2 = fdiv float %a1, %l2
  store volatile float %l2, ptr %dst
  store volatile float %res2, ptr %dst

  ret void
}

define void @f7(double %a1, ptr %src1, ptr %src2, ptr %src3, ptr %dst) {
; CHECK-LABEL: # *** IR Dump Before Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f7: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       [[LD1:%[0-9]+:vr64bit]] = VL64 [[ADDR1:%[0-9]+:addr64bit]], 0, $noreg :: (load (s64) from %ir.src1)
; CHECK-NEXT:  vr64bit = nofpexcept WFDDB %0:fp64bit, killed [[LD1]], implicit $fpc
; CHECK:       [[LD2:%[0-9]+:vr64bit]] = VL64 %2:addr64bit, 0, $noreg :: (load (s64) from %ir.src2)
; CHECK-NEXT:  vr64bit = nofpexcept WFDDB %0:fp64bit, [[LD2]], implicit $fpc
; CHECK-NEXT:  VST64 [[LD2]], %4:addr64bit, 0, $noreg :: (volatile store (s64) into %ir.dst)

; CHECK:       # *** IR Dump After Peephole Optimizations (peephole-opt) ***:
; CHECK-NEXT:  # Machine code for function f7: IsSSA, TracksLiveness
; CHECK-LABEL: bb.0 (%ir-block.0):
; CHECK:       fp64bit = nofpexcept DDB %0:fp64bit(tied-def 0), [[ADDR1]], 0, $noreg, implicit $fpc :: (load (s64) from %ir.src1)
; CHECK:       %8:vr64bit = nofpexcept WFDDB %0:fp64bit, [[LD2]], implicit $fpc

  %l1 = load double, ptr %src1
  %res1 = fdiv double %a1, %l1
  store volatile double %res1, ptr %dst

  %l2 = load double, ptr %src2
  %res2 = fdiv double %a1, %l2
  store volatile double %l2, ptr %dst
  store volatile double %res2, ptr %dst

  ret void
}

; Disable machine cse to stress the different path of the algorithm.
; Otherwise, we always fall in the simple case, i.e., only one definition.
; RUN: llc < %s -mtriple=armv7-apple-ios7.0 -disable-machine-cse -arm-stress-promote-const -mattr=+neon -verify-machineinstrs | FileCheck -check-prefix=PROMOTED %s
; The REGULAR run just checks that the inputs passed to promote const expose
; the appropriate patterns.
; RUN: llc < %s -mtriple=armv7-apple-ios7.0 -disable-machine-cse -arm-enable-promote-const=false -mattr=+neon -verify-machineinstrs | FileCheck -check-prefix=REGULAR %s

%struct.uint8x16x4_t = type { [4 x <16 x i8>] }

; Constant is a structure.
define %struct.uint8x16x4_t @test1() {
; PROMOTED-LABEL: _test1:
; Promote constant has created a big constant for the whole structure.
; PROMOTED: __PromotedConst
; PROMOTED: vld1.8
; PROMOTED: vld1.8
; PROMOTED: vld1.8
; PROMOTED: vld1.64
; PROMOTED: bx lr

; REGULAR-LABEL: _test1:
; Regular access is quite bad, it performs 4 loads, one for each chunk of
; the structure.
; REGULAR: LCPI0_0
; REGULAR: LCPI0_1
; REGULAR: LCPI0_2
; REGULAR: LCPI0_3
; REGULAR: bx lr
entry:
  ret %struct.uint8x16x4_t { [4 x <16 x i8>] [<16 x i8> <i8 -40, i8 -93, i8 -118, i8 -99, i8 -75, i8 -105, i8 74, i8 -110, i8 62, i8 -115, i8 -119, i8 -120, i8 34, i8 -124, i8 0, i8 -128>, <16 x i8> <i8 32, i8 124, i8 121, i8 120, i8 8, i8 117, i8 -56, i8 113, i8 -76, i8 110, i8 -53, i8 107, i8 7, i8 105, i8 103, i8 102>, <16 x i8> <i8 -24, i8 99, i8 -121, i8 97, i8 66, i8 95, i8 24, i8 93, i8 6, i8 91, i8 12, i8 89, i8 39, i8 87, i8 86, i8 85>, <16 x i8> <i8 -104, i8 83, i8 -20, i8 81, i8 81, i8 80, i8 -59, i8 78, i8 73, i8 77, i8 -37, i8 75, i8 122, i8 74, i8 37, i8 73>] }
}

; Two different uses of the same constant in the same basic block.
define <16 x i8> @test2(<16 x i8> %arg) {
entry:
; PROMOTED-LABEL: _test2:
; In stress mode, constant vector are promoted.
; PROMOTED: __PromotedConst.1
; PROMOTED: vld1.64
; PROMOTED: vadd.i8
; PROMOTED: vmls.i8
; PROMOTED: bx lr

; REGULAR-LABEL: _test2:
; Regular access is strictly the same as promoted access. The difference is
; that the address, and thus the space in memory, is not shared between
; constants.
; REGULAR: LCPI1_0
; REGULAR: vld1.64
; REGULAR: vadd.i8
; REGULAR: vmls.i8
; REGULAR: bx lr
  %add.i = add <16 x i8> %arg, <i8 -40, i8 -93, i8 -118, i8 -99, i8 -75, i8 -105, i8 74, i8 -110, i8 62, i8 -115, i8 -119, i8 -120, i8 34, i8 -124, i8 0, i8 -128>
  %mul.i = mul <16 x i8> %add.i, <i8 -40, i8 -93, i8 -118, i8 -99, i8 -75, i8 -105, i8 74, i8 -110, i8 62, i8 -115, i8 -119, i8 -120, i8 34, i8 -124, i8 0, i8 -128>
  %sub.i9 = sub <16 x i8> %add.i, %mul.i
  ret <16 x i8> %sub.i9
}

; Two different uses of the same constant in two different basic blocks,
; one dominates the other.
define <16 x i8> @test3(<16 x i8> %arg, i32 %path) {
; PROMOTED-LABEL: _test3:
; In stress mode, constant vector are promoted.
; Since the constant is the same as the previous function, the same address
; must be used.
; PROMOTED: __PromotedConst.1
; PROMOTED: vld1.64
; PROMOTED: __PromotedConst.1
; PROMOTED: __PromotedConst.2
; PROMOTED: vld1.64
; PROMOTED-NOT: vld1.64
; PROMOTED: bx lr

; REGULAR-LABEL: _test3:
; REGULAR: LCPI2_0
; REGULAR: vld1.64
; REGULAR: LCPI2_1
; REGULAR: vld1.64
; REGULAR-NOT: vld1.64
; REGULAR: bx lr
entry:
  %add.i = add <16 x i8> %arg, <i8 -40, i8 -93, i8 -118, i8 -99, i8 -75, i8 -105, i8 74, i8 -110, i8 62, i8 -115, i8 -119, i8 -120, i8 34, i8 -124, i8 0, i8 -128>
  %tobool = icmp eq i32 %path, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:
  %mul.i13 = mul <16 x i8> %add.i, <i8 -40, i8 -93, i8 -118, i8 -99, i8 -75, i8 -105, i8 74, i8 -110, i8 62, i8 -115, i8 -119, i8 -120, i8 34, i8 -124, i8 0, i8 -128>
  br label %if.end

if.else:
  %mul.i = mul <16 x i8> %add.i, <i8 -24, i8 99, i8 -121, i8 97, i8 66, i8 95, i8 24, i8 93, i8 6, i8 91, i8 12, i8 89, i8 39, i8 87, i8 86, i8 85>
  br label %if.end

if.end:
  %ret2.0 = phi <16 x i8> [ %mul.i13, %if.then ], [ %mul.i, %if.else ]
  %add.i12 = add <16 x i8> %add.i, %ret2.0
  ret <16 x i8> %add.i12
}

; Two different uses of the same constant in two different basic blocks,
; none dominates the other.
define <16 x i8> @test4(<16 x i8> %arg, i32 %path) {
; PROMOTED-LABEL: _test4:
; In stress mode, constant vector are promoted.
; Since the constant is the same as the previous function, the same address
; must be used.
; PROMOTED: __PromotedConst.1
; PROMOTED: vld1.64
; PROMOTED-NOT: vld1.64
; PROMOTED: bx lr

; REGULAR-LABEL: _test4:
; REGULAR: LCPI3_0
; REGULAR: vld1.64
; REGULAR-NOT: vld1.64
; REGULAR: bx lr
entry:
  %add.i = add <16 x i8> %arg, <i8 -40, i8 -93, i8 -118, i8 -99, i8 -75, i8 -105, i8 74, i8 -110, i8 62, i8 -115, i8 -119, i8 -120, i8 34, i8 -124, i8 0, i8 -128>
  %tobool = icmp eq i32 %path, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:
  %mul.i = mul <16 x i8> %add.i, <i8 -40, i8 -93, i8 -118, i8 -99, i8 -75, i8 -105, i8 74, i8 -110, i8 62, i8 -115, i8 -119, i8 -120, i8 34, i8 -124, i8 0, i8 -128>
  br label %if.end

if.end:
  %ret.0 = phi <16 x i8> [ %mul.i, %if.then ], [ %add.i, %entry ]
  ret <16 x i8> %ret.0
}

; Two different uses of the same constant in two different basic blocks,
; one is in a phi.
define <16 x i8> @test5(<16 x i8> %arg, i32 %path) {
; PROMOTED-LABEL: _test5:
; In stress mode, constant vector are promoted.
; Since the constant is the same as the previous function, the same address
; must be used.
; PROMOTED: __PromotedConst.1
; PROMOTED: vld1.64
; PROMOTED-NOT: vld1.64
; PROMOTED: bx lr

; REGULAR-LABEL: _test5:
; REGULAR: LCPI4_0
; REGULAR: vld1.64
; REGULAR: LCPI4_1
; REGULAR: vld1.64
; REGULAR: bx lr
entry:
  %tobool = icmp eq i32 %path, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:
  %add.i = add <16 x i8> %arg, <i8 -40, i8 -93, i8 -118, i8 -99, i8 -75, i8 -105, i8 74, i8 -110, i8 62, i8 -115, i8 -119, i8 -120, i8 34, i8 -124, i8 0, i8 -128>
  %mul.i26 = mul <16 x i8> %add.i, <i8 -40, i8 -93, i8 -118, i8 -99, i8 -75, i8 -105, i8 74, i8 -110, i8 62, i8 -115, i8 -119, i8 -120, i8 34, i8 -124, i8 0, i8 -128>
  br label %if.end

if.end:
  %ret.0 = phi <16 x i8> [ %mul.i26, %if.then ], [ <i8 -40, i8 -93, i8 -118, i8 -99, i8 -75, i8 -105, i8 74, i8 -110, i8 62, i8 -115, i8 -119, i8 -120, i8 34, i8 -124, i8 0, i8 -128>, %entry ]
  %mul.i25 = mul <16 x i8> %ret.0, %ret.0
  %mul.i24 = mul <16 x i8> %mul.i25, %mul.i25
  %mul.i23 = mul <16 x i8> %mul.i24, %mul.i24
  %mul.i = mul <16 x i8> %mul.i23, %mul.i23
  ret <16 x i8> %mul.i
}

define void @accessBig(ptr %storage) {
; PROMOTED-LABEL: _accessBig:
; PROMOTED: __PromotedConst.3
; PROMOTED: bx lr
  store <1 x i80> <i80 483673642326615442599424>, ptr %storage
  ret void
}

define void @asmStatement() {
; PROMOTED-LABEL: _asmStatement:
; PROMOTED-NOT: __PromotedConst
; PROMOTED: bx lr
  call void asm sideeffect "bfi r0, r0, $0, $1", "i,i"(i32 28, i32 4)
  ret void
}

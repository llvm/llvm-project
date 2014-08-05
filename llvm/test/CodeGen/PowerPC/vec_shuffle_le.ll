; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mattr=+altivec | FileCheck %s

define void @VPKUHUM_xy(<16 x i8>* %A, <16 x i8>* %B) {
entry:
; CHECK: VPKUHUM_xy:
        %tmp = load <16 x i8>* %A
        %tmp2 = load <16 x i8>* %B
        %tmp3 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp2, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
; CHECK: lvx [[REG1:[0-9]+]]
; CHECK: lvx [[REG2:[0-9]+]]
; CHECK: vpkuhum [[REG3:[0-9]+]], [[REG2]], [[REG1]]
        store <16 x i8> %tmp3, <16 x i8>* %A
        ret void
}

define void @VPKUHUM_xx(<16 x i8>* %A) {
entry:
; CHECK: VPKUHUM_xx:
        %tmp = load <16 x i8>* %A
        %tmp2 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
; CHECK: vpkuhum
        store <16 x i8> %tmp2, <16 x i8>* %A
        ret void
}

define void @VPKUWUM_xy(<16 x i8>* %A, <16 x i8>* %B) {
entry:
; CHECK: VPKUWUM_xy:
        %tmp = load <16 x i8>* %A
        %tmp2 = load <16 x i8>* %B
        %tmp3 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp2, <16 x i32> <i32 0, i32 1, i32 4, i32 5, i32 8, i32 9, i32 12, i32 13, i32 16, i32 17, i32 20, i32 21, i32 24, i32 25, i32 28, i32 29>
; CHECK: lvx [[REG1:[0-9]+]]
; CHECK: lvx [[REG2:[0-9]+]]
; CHECK: vpkuwum [[REG3:[0-9]+]], [[REG2]], [[REG1]]
        store <16 x i8> %tmp3, <16 x i8>* %A
        ret void
}

define void @VPKUWUM_xx(<16 x i8>* %A) {
entry:
; CHECK: VPKUWUM_xx:
        %tmp = load <16 x i8>* %A
        %tmp2 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp, <16 x i32> <i32 0, i32 1, i32 4, i32 5, i32 8, i32 9, i32 12, i32 13, i32 0, i32 1, i32 4, i32 5, i32 8, i32 9, i32 12, i32 13>
; CHECK: vpkuwum
        store <16 x i8> %tmp2, <16 x i8>* %A
        ret void
}

define void @VMRGLB_xy(<16 x i8>* %A, <16 x i8>* %B) {
entry:
; CHECK: VMRGLB_xy:
        %tmp = load <16 x i8>* %A
        %tmp2 = load <16 x i8>* %B
        %tmp3 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp2, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23>
; CHECK: lvx [[REG1:[0-9]+]]
; CHECK: lvx [[REG2:[0-9]+]]
; CHECK: vmrglb [[REG3:[0-9]+]], [[REG2]], [[REG1]]
        store <16 x i8> %tmp3, <16 x i8>* %A
        ret void
}

define void @VMRGLB_xx(<16 x i8>* %A) {
entry:
; CHECK: VMRGLB_xx:
        %tmp = load <16 x i8>* %A
        %tmp2 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp, <16 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3, i32 4, i32 4, i32 5, i32 5, i32 6, i32 6, i32 7, i32 7>
; CHECK: vmrglb
        store <16 x i8> %tmp2, <16 x i8>* %A
        ret void
}

define void @VMRGHB_xy(<16 x i8>* %A, <16 x i8>* %B) {
entry:
; CHECK: VMRGHB_xy:
        %tmp = load <16 x i8>* %A
        %tmp2 = load <16 x i8>* %B
        %tmp3 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp2, <16 x i32> <i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
; CHECK: lvx [[REG1:[0-9]+]]
; CHECK: lvx [[REG2:[0-9]+]]
; CHECK: vmrghb [[REG3:[0-9]+]], [[REG2]], [[REG1]]
        store <16 x i8> %tmp3, <16 x i8>* %A
        ret void
}

define void @VMRGHB_xx(<16 x i8>* %A) {
entry:
; CHECK: VMRGHB_xx:
        %tmp = load <16 x i8>* %A
        %tmp2 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp, <16 x i32> <i32 8, i32 8, i32 9, i32 9, i32 10, i32 10, i32 11, i32 11, i32 12, i32 12, i32 13, i32 13, i32 14, i32 14, i32 15, i32 15>
; CHECK: vmrghb
        store <16 x i8> %tmp2, <16 x i8>* %A
        ret void
}

define void @VMRGLH_xy(<16 x i8>* %A, <16 x i8>* %B) {
entry:
; CHECK: VMRGLH_xy:
        %tmp = load <16 x i8>* %A
        %tmp2 = load <16 x i8>* %B
        %tmp3 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp2, <16 x i32> <i32 0, i32 1, i32 16, i32 17, i32 2, i32 3, i32 18, i32 19, i32 4, i32 5, i32 20, i32 21, i32 6, i32 7, i32 22, i32 23>
; CHECK: lvx [[REG1:[0-9]+]]
; CHECK: lvx [[REG2:[0-9]+]]
; CHECK: vmrglh [[REG3:[0-9]+]], [[REG2]], [[REG1]]
        store <16 x i8> %tmp3, <16 x i8>* %A
        ret void
}

define void @VMRGLH_xx(<16 x i8>* %A) {
entry:
; CHECK: VMRGLH_xx:
        %tmp = load <16 x i8>* %A
        %tmp2 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp, <16 x i32> <i32 0, i32 1, i32 0, i32 1, i32 2, i32 3, i32 2, i32 3, i32 4, i32 5, i32 4, i32 5, i32 6, i32 7, i32 6, i32 7>
; CHECK: vmrglh
        store <16 x i8> %tmp2, <16 x i8>* %A
        ret void
}

define void @VMRGHH_xy(<16 x i8>* %A, <16 x i8>* %B) {
entry:
; CHECK: VMRGHH_xy:
        %tmp = load <16 x i8>* %A
        %tmp2 = load <16 x i8>* %B
        %tmp3 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp2, <16 x i32> <i32 8, i32 9, i32 24, i32 25, i32 10, i32 11, i32 26, i32 27, i32 12, i32 13, i32 28, i32 29, i32 14, i32 15, i32 30, i32 31>
; CHECK: lvx [[REG1:[0-9]+]]
; CHECK: lvx [[REG2:[0-9]+]]
; CHECK: vmrghh [[REG3:[0-9]+]], [[REG2]], [[REG1]]
        store <16 x i8> %tmp3, <16 x i8>* %A
        ret void
}

define void @VMRGHH_xx(<16 x i8>* %A) {
entry:
; CHECK: VMRGHH_xx:
        %tmp = load <16 x i8>* %A
        %tmp2 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp, <16 x i32> <i32 8, i32 9, i32 8, i32 9, i32 10, i32 11, i32 10, i32 11, i32 12, i32 13, i32 12, i32 13, i32 14, i32 15, i32 14, i32 15>
; CHECK: vmrghh
        store <16 x i8> %tmp2, <16 x i8>* %A
        ret void
}

define void @VMRGLW_xy(<16 x i8>* %A, <16 x i8>* %B) {
entry:
; CHECK: VMRGLW_xy:
        %tmp = load <16 x i8>* %A
        %tmp2 = load <16 x i8>* %B
        %tmp3 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 16, i32 17, i32 18, i32 19, i32 4, i32 5, i32 6, i32 7, i32 20, i32 21, i32 22, i32 23>
; CHECK: lvx [[REG1:[0-9]+]]
; CHECK: lvx [[REG2:[0-9]+]]
; CHECK: vmrglw [[REG3:[0-9]+]], [[REG2]], [[REG1]]
        store <16 x i8> %tmp3, <16 x i8>* %A
        ret void
}

define void @VMRGLW_xx(<16 x i8>* %A) {
entry:
; CHECK: VMRGLW_xx:
        %tmp = load <16 x i8>* %A
        %tmp2 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 4, i32 5, i32 6, i32 7>
; CHECK: vmrglw
        store <16 x i8> %tmp2, <16 x i8>* %A
        ret void
}

define void @VMRGHW_xy(<16 x i8>* %A, <16 x i8>* %B) {
entry:
; CHECK: VMRGHW_xy:
        %tmp = load <16 x i8>* %A
        %tmp2 = load <16 x i8>* %B
        %tmp3 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp2, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 24, i32 25, i32 26, i32 27, i32 12, i32 13, i32 14, i32 15, i32 28, i32 29, i32 30, i32 31>
; CHECK: lvx [[REG1:[0-9]+]]
; CHECK: lvx [[REG2:[0-9]+]]
; CHECK: vmrghw [[REG3:[0-9]+]], [[REG2]], [[REG1]]
        store <16 x i8> %tmp3, <16 x i8>* %A
        ret void
}

define void @VMRGHW_xx(<16 x i8>* %A) {
entry:
; CHECK: VMRGHW_xx:
        %tmp = load <16 x i8>* %A
        %tmp2 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 12, i32 13, i32 14, i32 15>
; CHECK: vmrghw
        store <16 x i8> %tmp2, <16 x i8>* %A
        ret void
}

define void @VSLDOI_xy(<16 x i8>* %A, <16 x i8>* %B) {
entry:
; CHECK: VSLDOI_xy:
        %tmp = load <16 x i8>* %A
        %tmp2 = load <16 x i8>* %B
        %tmp3 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp2, <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27>
; CHECK: lvx [[REG1:[0-9]+]]
; CHECK: lvx [[REG2:[0-9]+]]
; CHECK: vsldoi [[REG3:[0-9]+]], [[REG2]], [[REG1]], 4
        store <16 x i8> %tmp3, <16 x i8>* %A
        ret void
}

define void @VSLDOI_xx(<16 x i8>* %A) {
entry:
; CHECK: VSLDOI_xx:
        %tmp = load <16 x i8>* %A
        %tmp2 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp, <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11>
; CHECK: vsldoi
        store <16 x i8> %tmp2, <16 x i8>* %A
        ret void
}


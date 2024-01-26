; RUN: llc -mtriple=aarch64 -verify-machineinstrs --mattr=+cpa -O0 -global-isel=1 -global-isel-abort=1 %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECK-CPA-O0
; RUN: llc -mtriple=aarch64 -verify-machineinstrs --mattr=+cpa -O3 -global-isel=1 -global-isel-abort=1 %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECK-CPA-O3
; RUN: llc -mtriple=aarch64 -verify-machineinstrs --mattr=-cpa -O0 -global-isel=1 -global-isel-abort=1 %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECK-NOCPA-O0
; RUN: llc -mtriple=aarch64 -verify-machineinstrs --mattr=-cpa -O3 -global-isel=1 -global-isel-abort=1 %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECK-NOCPA-O3

%struct.my_type = type { i64, i64 }
%struct.my_type2 = type { i64, i64, i64, i64, i64, i64 }

@array = external dso_local global [10 x %struct.my_type], align 8
@array2 = external dso_local global [10 x %struct.my_type2], align 8

define void @addpt1(i64 %index, i64 %arg) {
; CHECK-CPA-O0-LABEL:    addpt1:
; CHECK-CPA-O0:          addpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #4
; CHECK-CPA-O0:          str	x{{[0-9]+}}, [[[REG1]], #8]
;
; CHECK-CPA-O3-LABEL:    addpt1:
; CHECK-CPA-O3:          addpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #4
; CHECK-CPA-O3:          str	x{{[0-9]+}}, [[[REG1]], #8]
;
; CHECK-NOCPA-O0-LABEL:  addpt1:
; CHECK-NOCPA-O0:        add	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #4
; CHECK-NOCPA-O0:        str	x{{[0-9]+}}, [[[REG1]], #8]
;
; CHECK-NOCPA-O3-LABEL:  addpt1:
; CHECK-NOCPA-O3:        add	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #4
; CHECK-NOCPA-O3:        str	x{{[0-9]+}}, [[[REG1]], #8]
entry:
  %e2 = getelementptr inbounds %struct.my_type, ptr @array, i64 %index, i32 1
  store i64 %arg, ptr %e2, align 8
  ret void
}

define void @maddpt1(i32 %pos, ptr %val) {
; CHECK-CPA-O0-LABEL:    maddpt1:
; CHECK-CPA-O0:          maddpt	x0, x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-CPA-O0:          b	memcpy
;
; CHECK-CPA-O3-LABEL:    maddpt1:
; CHECK-CPA-O3:          maddpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-CPA-O3:          str	q{{[0-9]+}}, [[[REG1]]]
; CHECK-CPA-O3:          str	q{{[0-9]+}}, [[[REG1]], #16]
; CHECK-CPA-O3:          str	q{{[0-9]+}}, [[[REG1]], #32]
;
; CHECK-NOCPA-O0-LABEL:  maddpt1:
; CHECK-NOCPA-O0:        smaddl	x0, w{{[0-9]+}}, w{{[0-9]+}}, x{{[0-9]+}}
; CHECK-NOCPA-O0:        b	memcpy
;
; CHECK-NOCPA-O3-LABEL:  maddpt1:
; CHECK-NOCPA-O3:        smaddl	[[REG1:x[0-9]+]], w{{[0-9]+}}, w{{[0-9]+}}, x{{[0-9]+}}
; CHECK-NOCPA-O3:        str	q{{[0-9]+}}, [[[REG1]]]
; CHECK-NOCPA-O3:        str	q{{[0-9]+}}, [[[REG1]], #16]
; CHECK-NOCPA-O3:        str	q{{[0-9]+}}, [[[REG1]], #32]
entry:
  %idxprom = sext i32 %pos to i64
  %arrayidx = getelementptr inbounds [10 x %struct.my_type2], ptr @array2, i64 0, i64 %idxprom
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 dereferenceable(48) %arrayidx, ptr align 8 dereferenceable(48) %val, i64 48, i1 false)
  ret void
}

define void @msubpt1(i32 %index, i32 %elem) {
; CHECK-CPA-O0-LABEL:    msubpt1:
; CHECK-CPA-O0:          addpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-CPA-O0:          msubpt	x0, x{{[0-9]+}}, x{{[0-9]+}}, [[REG1]]
; CHECK-CPA-O0:          addpt	x1, x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-CPA-O0:          b	memcpy
;
; CHECK-CPA-O3-LABEL:    msubpt1:
; CHECK-CPA-O3:          msubpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-CPA-O3:          str	q{{[0-9]+}}, [[[REG1]], #192]
; CHECK-CPA-O3:          str	q{{[0-9]+}}, [[[REG1]], #208]
; CHECK-CPA-O3:          str	q{{[0-9]+}}, [[[REG1]], #224]
;
; CHECK-NOCPA-O0-LABEL:  msubpt1:
; CHECK-NOCPA-O0:        mneg	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-NOCPA-O0:        add	x0, x{{[0-9]+}}, [[REG1]]
; CHECK-NOCPA-O0:        b	memcpy
;
; CHECK-NOCPA-O3-LABEL:  msubpt1:
; CHECK-NOCPA-O3:        mneg	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-NOCPA-O3:        add	[[REG2:x[0-9]+]], x{{[0-9]+}}, [[REG1]]
; CHECK-NOCPA-O3:        str	q{{[0-9]+}}, [[[REG1]], #192]
; CHECK-NOCPA-O3:        str	q{{[0-9]+}}, [[[REG1]], #208]
; CHECK-NOCPA-O3:        str	q{{[0-9]+}}, [[[REG1]], #224]
entry:
  %idx.ext = sext i32 %index to i64
  %idx.neg = sub nsw i64 0, %idx.ext
  %add.ptr = getelementptr inbounds %struct.my_type2, ptr getelementptr inbounds ([10 x %struct.my_type2], ptr @array2, i64 0, i64 6), i64 %idx.neg
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 dereferenceable(48) %add.ptr, ptr align 8 dereferenceable(48) getelementptr inbounds ([10 x %struct.my_type2], ptr @array2, i64 0, i64 2), i64 48, i1 false), !tbaa.struct !6
  ret void
}

define void @subpt1(i32 %index, i32 %elem) {
; CHECK-CPA-O0-LABEL:    subpt1:
; CHECK-CPA-O0:          addpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-CPA-O0:          str	q{{[0-9]+}}, [[[REG1]], x{{[0-9]+}}, lsl #4]
;
; CHECK-CPA-O3-LABEL:    subpt1:
; CHECK-CPA-O3:          addpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #4
; CHECK-CPA-O3:          str	q{{[0-9]+}}, [[[REG1]], #64]
;
; CHECK-NOCPA-O0-LABEL:  subpt1:
; CHECK-NOCPA-O0:        add	[[REG1:x[0-9]+]], x{{[0-9]+}}, #96
; CHECK-NOCPA-O0:        str	q{{[0-9]+}}, [[[REG1]], x{{[0-9]+}}, lsl #4]
;
; CHECK-NOCPA-O3-LABEL:  subpt1:
; CHECK-NOCPA-O3:        add	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #4
; CHECK-NOCPA-O3:        str	q{{[0-9]+}}, [[[REG1]], #64]
entry:
  %conv = sext i32 %index to i64
  %mul.neg = mul nsw i64 %conv, -16
  %add.ptr = getelementptr inbounds %struct.my_type, ptr getelementptr inbounds ([10 x %struct.my_type], ptr @array, i64 0, i64 6), i64 %mul.neg
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %add.ptr, ptr noundef nonnull align 8 dereferenceable(16) getelementptr inbounds ([10 x %struct.my_type], ptr @array, i64 0, i64 2), i64 16, i1 false), !tbaa.struct !6
  ret void
}

define void @subpt2(i32 %index, i32 %elem) {
; CHECK-CPA-O0-LABEL:    subpt2:
; CHECK-CPA-O0:          addpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-CPA-O0:          str	q{{[0-9]+}}, [[[REG1]], x{{[0-9]+}}, lsl #4]
;
; CHECK-CPA-O3-LABEL:    subpt2:
; CHECK-CPA-O3:          addpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #4
; CHECK-CPA-O3:          str	q{{[0-9]+}}, [[[REG1]], #64]
;
; CHECK-NOCPA-O0-LABEL:  subpt2:
; CHECK-NOCPA-O0:        add	[[REG1:x[0-9]+]], x{{[0-9]+}}, #96
; CHECK-NOCPA-O0:        str	q{{[0-9]+}}, [[[REG1]], x{{[0-9]+}}, lsl #4]
;
; CHECK-NOCPA-O3-LABEL:  subpt2:
; CHECK-NOCPA-O3:        add	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #4
; CHECK-NOCPA-O3:        str	q{{[0-9]+}}, [[[REG1]], #64]
entry:
  %idx.ext = sext i32 %index to i64
  %idx.neg = sub nsw i64 0, %idx.ext
  %add.ptr = getelementptr inbounds %struct.my_type, ptr getelementptr inbounds ([10 x %struct.my_type], ptr @array, i64 0, i64 6), i64 %idx.neg
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %add.ptr, ptr noundef nonnull align 8 dereferenceable(16) getelementptr inbounds ([10 x %struct.my_type], ptr @array, i64 0, i64 2), i64 16, i1 false), !tbaa.struct !11
  ret void
}

define ptr @subpt3(ptr %ptr, i32 %index) {
; CHECK-CPA-O0-LABEL:    subpt3:
; CHECK-CPA-O0:          mov	[[REG1:x[0-9]+]], #-8
; CHECK-CPA-O0:          addpt	x0, x{{[0-9]+}}, [[REG1]]
; CHECK-CPA-O0:          ret
;
; CHECK-CPA-O3-LABEL:    subpt3:
; CHECK-CPA-O3:          mov	[[REG1:x[0-9]+]], #-8
; CHECK-CPA-O3:          addpt	x0, x{{[0-9]+}}, [[REG1]]
; CHECK-CPA-O3:          ret
;
; CHECK-NOCPA-O0-LABEL:  subpt3:
; CHECK-NOCPA-O0:        subs	x0, x{{[0-9]+}}, #8
; CHECK-NOCPA-O0:        ret
;
; CHECK-NOCPA-O3-LABEL:  subpt3:
; CHECK-NOCPA-O3:        sub	x0, x{{[0-9]+}}, #8
; CHECK-NOCPA-O3:        ret
entry:
  %incdec.ptr.i.i.i = getelementptr inbounds i64, ptr %ptr, i64 -1
  ret ptr %incdec.ptr.i.i.i
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

!6 = !{i64 0, i64 8, !7, i64 8, i64 8, !7, i64 16, i64 8, !7, i64 24, i64 8, !7, i64 32, i64 8, !7, i64 40, i64 8, !7}
!7 = !{!8, !8, i64 0}
!8 = !{!"long", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{i64 0, i64 8, !7, i64 8, i64 8, !7}

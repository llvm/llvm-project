; RUN: sed -e "s/ORDER/acquire/" %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/release/" %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/acq_rel/" %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e "s/ORDER/seq_cst/" %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e 's/ORDER/syncscope("singlethread") acquire/' %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e 's/ORDER/syncscope("singlethread") release/' %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e 's/ORDER/syncscope("singlethread") acq_rel/' %s | llc -march=hexagon | FileCheck %s
; RUN: sed -e 's/ORDER/syncscope("singlethread") seq_cst/' %s | llc -march=hexagon | FileCheck %s

define void @fence_func() #0 {
entry:
  fence ORDER
  ret void
}
; CHECK-LABEL: fence_func:
; CHECK: %bb.0
; CHECK-NEXT: {
; CHECK-NEXT:   barrier
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   jumpr r31
; CHECK-NEXT: }

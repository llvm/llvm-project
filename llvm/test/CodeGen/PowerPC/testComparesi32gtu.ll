; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O2 \
; RUN:   -ppc-gpr-icmps=all -ppc-asm-full-reg-names -mcpu=pwr8 < %s | FileCheck %s \
; RUN:  --implicit-check-not cmpw --implicit-check-not cmpd --implicit-check-not cmpl

%struct.tree_common = type { i8, [3 x i8] }
declare signext i32 @fn2(...) local_unnamed_addr #1

; Function Attrs: nounwind
define i32 @testCompare1(%struct.tree_common* nocapture readonly %arg1) {
; CHECK-LABEL: testCompare1:
; CHECK:       # %bb.0: # %entry
; CHECK:         lbz r3, 0(r3)
; CHECK-DAG:     clrlwi r3, r3, 31
; CHECK-DAG:     clrldi r3, r3, 32
; CHECK:         lbz  r4, 0(r4)
; CHECK-DAG:     clrlwi r4, r4, 31
; CHECK-DAG:     clrldi r4, r4, 32
; CHECK:         sub r3, r3, r4
; CHECK-NEXT:    rldicl r3, r3, 1, 63
entry:
  %bf.load = load i8, i8* bitcast (i32 (%struct.tree_common*)* @testCompare1 to i8*), align 4
  %bf.clear = and i8 %bf.load, 1
  %0 = getelementptr inbounds %struct.tree_common, %struct.tree_common* %arg1, i64 0, i32 0
  %bf.load1 = load i8, i8* %0, align 4
  %bf.clear2 = and i8 %bf.load1, 1
  %cmp = icmp ugt i8 %bf.clear, %bf.clear2
  %conv = zext i1 %cmp to i32
  %call = tail call signext i32 bitcast (i32 (...)* @fn2 to i32 (i32)*)(i32 signext %conv) #2
  ret i32 undef
}

; Function Attrs: norecurse nounwind readnone
define signext i32 @testCompare2(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LABEL: testCompare2:
; CHECK:       # %bb.0: # %entry
; CHECK-DAG:     rlwinm r3, r3, 0, 31, 31
; CHECK-DAG:     rlwinm r4, r4, 0, 31, 31
; CHECK-DAG:     clrldi r3, r3, 32
; CHECK-DAG:     clrldi r4, r4, 32
; CHECK:         sub r3, r4, r3
; CHECK-NEXT:    rldicl r3, r3, 1, 63
; CHECK-NEXT:    blr
entry:
  %and = and i32 %a, 1
  %and1 = and i32 %b, 1
  %cmp = icmp ugt i32 %and, %and1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

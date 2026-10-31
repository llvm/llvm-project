; RUN: llc -mtriple=hexagon -stop-after=regallocscoringpass -verify-machineinstrs=true < %s | FileCheck %s --check-prefix=CHECK-MIR
; RUN: llc -mtriple=hexagon -verify-machineinstrs=true < %s | FileCheck %s

;; Basic test with no spilling. Check check both cs0/m0 and cs1/ms1 are allocated.

; CHECK-MIR-LABEL: name: test_spill0
; CHECK-MIR-DAG: $r[[REGM1:[0-9]+]] = A2_tfrsi 100
; CHECK-MIR-DAG: $r[[REGM2:[0-9]+]] = A2_tfrsi 200
; CHECK-MIR-DAG: $m0 = COPY killed renamable $r[[REGM1]]
; CHECK-MIR-DAG: $cs0 = COPY killed renamable $r2
; CHECK-MIR-DAG: $m1 = COPY renamable $r[[REGM2]]
; CHECK-MIR-DAG: $cs1 = COPY renamable $r3
; CHECK-MIR: PS_loadrub_pci killed renamable $r{{[[:digit:]]+}}, 4, {{.*}}renamable $cs_m0
; CHECK-MIR: PS_loadrub_pci killed renamable $r{{[[:digit:]]+}}, 8, {{.*}}renamable $cs_m1

; CHECK-LABEL: test_spill0:
; CHECK-DAG: cs0 = r2
; CHECK-DAG: cs1 = r3
; CHECK-DAG: r[[REGM1:[0-9]+]] = #100
; CHECK-DAG: m0 = r[[REGM1]]
; CHECK-DAG: r[[REGM2:[0-9]+]]:{{[0-9]+}} = combine(##200,r0)
; CHECK-DAG: m1 = r[[REGM2]]
; CHECK-DAG: r{{[0-9]+}} = memub(r{{[0-9]+}}++#4:circ(m0))
; CHECK-DAG: r{{[0-9]+}} = memub(r{{[0-9]+}}++#8:circ(m1))

define i32 @test_spill0(ptr %p1, ptr %p2, ptr %b1, ptr %b2, i32 %n) #0 {
entry:
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%inext, %loop]
  %pp1 = phi ptr [%p1, %entry], [%np1, %loop]
  %pp2 = phi ptr [%p2, %entry], [%np2, %loop]
  %acc = phi i32 [0, %entry], [%accnext, %loop]
  %r1 = call {i32, ptr} @llvm.hexagon.L2.loadrub.pci(ptr %pp1, i32 4, i32 100, ptr %b1)
  %v1 = extractvalue {i32, ptr} %r1, 0
  %np1 = extractvalue {i32, ptr} %r1, 1
  %r2 = call {i32, ptr} @llvm.hexagon.L2.loadrub.pci(ptr %pp2, i32 8, i32 200, ptr %b2)
  %v2 = extractvalue {i32, ptr} %r2, 0
  %np2 = extractvalue {i32, ptr} %r2, 1
  %s1 = add i32 %v1, %v2
  %accnext = add i32 %acc, %s1
  %inext = add i32 %i, 1
  %cmp = icmp slt i32 %inext, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret i32 %accnext
}

declare {i32, ptr} @llvm.hexagon.L2.loadrub.pci(ptr, i32, i32, ptr)

declare void @clobber()

attributes #0 = { nounwind "target-features"="-packets" }

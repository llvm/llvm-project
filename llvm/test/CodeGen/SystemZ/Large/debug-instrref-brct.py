# RUN: %python %s | llc -mtriple=s390x-linux-gnu -x mir --run-pass=systemz-long-branch \
# RUN:    | FileCheck %s

# CHECK: debugValueSubstitutions:
# CHECK:   - { srcinst: 1, srcop: 0, dstinst: 3, dstop: 0, subreg: 0 }
# CHECK:   - { srcinst: 1, srcop: 3, dstinst: 3, dstop: 3, subreg: 0 }
# CHECK-NEXT: constants:       []
# CHECK: $r3l = AHI $r3l, -1
# CHECK-NEXT: BRCL 14, 6, %bb.2
print(" name: main")
print(" alignment: 16")
print(" tracksRegLiveness: true")
print(" liveins: ")
print("   - { reg: '$r1d', virtual-reg: '' }")
print("   - { reg: '$r2d', virtual-reg: '' }")
print("   - { reg: '$r3l', virtual-reg: '' }")
print("   - { reg: '$r4l', virtual-reg: '' }")
print(" debugValueSubstitutions: []")
print(" body:            |")
print("   bb.0:")
print("     liveins: $r3l, $r4l, $r2d, $r3d")
print("     $r3l = BRCT $r3l, %bb.2, implicit-def $cc, debug-instr-number 1")
print("     J %bb.1, debug-instr-number 2")
print("   bb.1:")
print("     liveins: $r1d, $r2d")
for i in range(0, 8192):
    print("     $r1d = LGR $r2d")
    print("     $r2d = LGR $r1d")
print("     Return implicit $r2d")
print("   bb.2:")
print("     liveins: $r4l")
print("     Return implicit $r4l")

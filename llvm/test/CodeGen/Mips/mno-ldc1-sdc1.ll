; Check that [sl]dc1 are normally emitted. MIPS32r2 should have [sl]dxc1 too.
; RUN: llc -march=mipsel -mcpu=mips32 -relocation-model=pic < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R1-LDC1
; RUN: llc -march=mipsel -mcpu=mips32r2 -relocation-model=pic < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R2-LDXC1
; RUN: llc -march=mipsel -mcpu=mips32r6 -relocation-model=pic < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R6-LDC1

; Check that -mno-ldc1-sdc1 disables [sl]dc1
; RUN: llc -march=mipsel -relocation-model=pic -mno-ldc1-sdc1 \
; RUN:   -mcpu=mips32   < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R1,32R1-LE,32R1-LE-PIC
; RUN: llc -march=mipsel -relocation-model=pic -mno-ldc1-sdc1 \
; RUN:   -mcpu=mips32r2 < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R2,32R2-LE,32R2-LE-PIC
; RUN: llc -march=mipsel -relocation-model=pic -mno-ldc1-sdc1 \
; RUN:   -mcpu=mips32r6 < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R6,32R6-LE,32R6-LE-PIC

; Check again for big-endian
; RUN: llc -march=mips -relocation-model=pic -mno-ldc1-sdc1 \
; RUN:   -mcpu=mips32   < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R1,32R1-BE,32R1-BE-PIC
; RUN: llc -march=mips -relocation-model=pic -mno-ldc1-sdc1 \
; RUN:   -mcpu=mips32r2 < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R2,32R2-BE,32R2-BE-PIC
; RUN: llc -march=mips -relocation-model=pic -mno-ldc1-sdc1 \
; RUN:   -mcpu=mips32r6 < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R6,32R6-BE,32R6-BE-PIC

; Check again for the static relocation model
; RUN: llc -march=mipsel -relocation-model=static -mno-ldc1-sdc1 \
; RUN:   -mcpu=mips32   < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R1,32R1-LE,32R1-LE-STATIC
; RUN: llc -march=mipsel -relocation-model=static -mno-ldc1-sdc1 \
; RUN:   -mcpu=mips32r2 < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R2,32R2-LE,32R2-LE-STATIC
; RUN: llc -march=mipsel -relocation-model=static -mno-ldc1-sdc1 \
; RUN:   -mcpu=mips32r6 < %s | \
; RUN:   FileCheck %s -check-prefixes=ALL,32R6,32R6-LE,32R6-LE-STATIC

@g0 = common global double 0.000000e+00, align 8

; ALL-LABEL: test_ldc1:

; 32R1-LE-PIC-DAG:    lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 32R1-LE-PIC-DAG:    lw $[[R1:[0-9]+]], 4(${{[0-9]+}})
; 32R1-LE-PIC-DAG:    mtc1 $[[R0]], $f0
; 32R1-LE-PIC-DAG:    mtc1 $[[R1]], $f1

; 32R2-LE-PIC-DAG:    lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 32R2-LE-PIC-DAG:    lw $[[R1:[0-9]+]], 4(${{[0-9]+}})
; 32R2-LE-PIC-DAG:    mtc1 $[[R0]], $f0
; 32R2-LE-PIC-DAG:    mthc1 $[[R1]], $f0

; 32R6-LE-PIC-DAG:    lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 32R6-LE-PIC-DAG:    lw $[[R1:[0-9]+]], 4(${{[0-9]+}})
; 32R6-LE-PIC-DAG:    mtc1 $[[R0]], $f0
; 32R6-LE-PIC-DAG:    mthc1 $[[R1]], $f0

; 32R1-LE-STATIC-DAG: lui $[[R0:[0-9]+]], %hi(g0)
; 32R1-LE-STATIC-DAG: lw $[[R1:[0-9]+]], %lo(g0)($[[R0]])
; 32R1-LE-STATIC-DAG: addiu $[[R2:[0-9]+]], $[[R0]], %lo(g0)
; 32R1-LE-STATIC-DAG: lw $[[R3:[0-9]+]], 4($[[R2]])
; 32R1-LE-STATIC-DAG: mtc1 $[[R1]], $f0
; 32R1-LE-STATIC-DAG: mtc1 $[[R3]], $f1

; 32R2-LE-STATIC-DAG: lui $[[R0:[0-9]+]], %hi(g0)
; 32R2-LE-STATIC-DAG: lw $[[R1:[0-9]+]], %lo(g0)($[[R0]])
; 32R2-LE-STATIC-DAG: addiu $[[R2:[0-9]+]], $[[R0]], %lo(g0)
; 32R2-LE-STATIC-DAG: lw $[[R3:[0-9]+]], 4($[[R2]])
; 32R2-LE-STATIC-DAG: mtc1 $[[R1]], $f0
; 32R2-LE-STATIC-DAG: mthc1 $[[R3]], $f0

; 32R6-LE-STATIC-DAG: lui $[[R0:[0-9]+]], %hi(g0)
; 32R6-LE-STATIC-DAG: lw $[[R1:[0-9]+]], %lo(g0)($[[R0]])
; 32R6-LE-STATIC-DAG: addiu $[[R2:[0-9]+]], $[[R0]], %lo(g0)
; 32R6-LE-STATIC-DAG: lw $[[R3:[0-9]+]], 4($[[R2]])
; 32R6-LE-STATIC-DAG: mtc1 $[[R1]], $f0
; 32R6-LE-STATIC-DAG: mthc1 $[[R3]], $f0

; 32R1-BE-PIC-DAG:    lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 32R1-BE-PIC-DAG:    lw $[[R1:[0-9]+]], 4(${{[0-9]+}})
; 32R1-BE-PIC-DAG:    mtc1 $[[R1]], $f0
; 32R1-BE-PIC-DAG:    mtc1 $[[R0]], $f1

; 32R2-BE-PIC-DAG:    lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 32R2-BE-PIC-DAG:    lw $[[R1:[0-9]+]], 4(${{[0-9]+}})
; 32R2-BE-PIC-DAG:    mtc1 $[[R1]], $f0
; 32R2-BE-PIC-DAG:    mthc1 $[[R0]], $f0

; 32R6-BE-PIC-DAG:    lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 32R6-BE-PIC-DAG:    lw $[[R1:[0-9]+]], 4(${{[0-9]+}})
; 32R6-BE-PIC-DAG:    mtc1 $[[R1]], $f0
; 32R6-BE-PIC-DAG:    mthc1 $[[R0]], $f0

; 32R1-LDC1:          ldc1 $f0, 0(${{[0-9]+}})

; 32R2-LDXC1:         ldc1 $f0, 0(${{[0-9]+}})

; 32R6-LDC1:          ldc1 $f0, 0(${{[0-9]+}})

define double @test_ldc1() {
entry:
  %0 = load double, double* @g0, align 8
  ret double %0
}

; ALL-LABEL: test_sdc1:

; 32R1-LE-PIC-DAG:    mfc1 $[[R0:[0-9]+]], $f12
; 32R1-LE-PIC-DAG:    mfc1 $[[R1:[0-9]+]], $f13
; 32R1-LE-PIC-DAG:    sw $[[R0]], 0(${{[0-9]+}})
; 32R1-LE-PIC-DAG:    sw $[[R1]], 4(${{[0-9]+}})

; 32R2-LE-PIC-DAG:    mfc1 $[[R0:[0-9]+]], $f12
; 32R2-LE-PIC-DAG:    mfhc1 $[[R1:[0-9]+]], $f12
; 32R2-LE-PIC-DAG:    sw $[[R0]], 0(${{[0-9]+}})
; 32R2-LE-PIC-DAG:    sw $[[R1]], 4(${{[0-9]+}})

; 32R6-LE-PIC-DAG:    mfc1 $[[R0:[0-9]+]], $f12
; 32R6-LE-PIC-DAG:    mfhc1 $[[R1:[0-9]+]], $f12
; 32R6-LE-PIC-DAG:    sw $[[R0]], 0(${{[0-9]+}})
; 32R6-LE-PIC-DAG:    sw $[[R1]], 4(${{[0-9]+}})

; 32R1-LE-STATIC-DAG: mfc1 $[[R0:[0-9]+]], $f12
; 32R1-LE-STATIC-DAG: mfc1 $[[R1:[0-9]+]], $f13
; 32R1-LE-STATIC-DAG: lui $[[R2:[0-9]+]], %hi(g0)
; 32R1-LE-STATIC-DAG: sw $[[R0]], %lo(g0)($[[R2]])
; 32R1-LE-STATIC-DAG: addiu $[[R3:[0-9]+]], $[[R2]], %lo(g0)
; 32R1-LE-STATIC-DAG: sw $[[R1]], 4($[[R3]])

; 32R2-LE-STATIC-DAG: mfc1 $[[R0:[0-9]+]], $f12
; 32R2-LE-STATIC-DAG: mfhc1 $[[R1:[0-9]+]], $f12
; 32R2-LE-STATIC-DAG: lui $[[R2:[0-9]+]], %hi(g0)
; 32R2-LE-STATIC-DAG: sw $[[R0]], %lo(g0)($[[R2]])
; 32R2-LE-STATIC-DAG: addiu $[[R3:[0-9]+]], $[[R2]], %lo(g0)
; 32R2-LE-STATIC-DAG: sw $[[R1]], 4($[[R3]])

; 32R6-LE-STATIC-DAG: mfc1 $[[R0:[0-9]+]], $f12
; 32R6-LE-STATIC-DAG: mfhc1 $[[R1:[0-9]+]], $f12
; 32R6-LE-STATIC-DAG: lui $[[R2:[0-9]+]], %hi(g0)
; 32R6-LE-STATIC-DAG: sw $[[R0]], %lo(g0)($[[R2]])
; 32R6-LE-STATIC-DAG: addiu $[[R3:[0-9]+]], $[[R2]], %lo(g0)
; 32R6-LE-STATIC-DAG: sw $[[R1]], 4($[[R3]])

; 32R1-BE-PIC-DAG:    mfc1 $[[R0:[0-9]+]], $f12
; 32R1-BE-PIC-DAG:    mfc1 $[[R1:[0-9]+]], $f13
; 32R1-BE-PIC-DAG:    sw $[[R1]], 0(${{[0-9]+}})
; 32R1-BE-PIC-DAG:    sw $[[R0]], 4(${{[0-9]+}})

; 32R2-BE-PIC-DAG:    mfc1 $[[R0:[0-9]+]], $f12
; 32R2-BE-PIC-DAG:    mfhc1 $[[R1:[0-9]+]], $f12
; 32R2-BE-PIC-DAG:    sw $[[R1]], 0(${{[0-9]+}})
; 32R2-BE-PIC-DAG:    sw $[[R0]], 4(${{[0-9]+}})

; 32R6-BE-PIC-DAG:    mfc1 $[[R0:[0-9]+]], $f12
; 32R6-BE-PIC-DAG:    mfhc1 $[[R1:[0-9]+]], $f12
; 32R6-BE-PIC-DAG:    sw $[[R1]], 0(${{[0-9]+}})
; 32R6-BE-PIC-DAG:    sw $[[R0]], 4(${{[0-9]+}})

; 32R1-LDC1:          sdc1 $f{{[0-9]+}}, 0(${{[0-9]+}})

; 32R2-LDXC1:         sdc1 $f{{[0-9]+}}, 0(${{[0-9]+}})

; 32R6-LDC1:          sdc1 $f{{[0-9]+}}, 0(${{[0-9]+}})

define void @test_sdc1(double %a) {
entry:
  store double %a, double* @g0, align 8
  ret void
}

; ALL-LABEL: test_ldxc1:

; 32R1-LE-DAG:   lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 32R1-LE-DAG:   lw $[[R1:[0-9]+]], 4(${{[0-9]+}})
; 32R1-BE-DAG:   lw $[[R0:[0-9]+]], 4(${{[0-9]+}})
; 32R1-BE-DAG:   lw $[[R1:[0-9]+]], 0(${{[0-9]+}})
; 32R1-DAG:      mtc1 $[[R0]], $f0
; 32R1-DAG:      mtc1 $[[R1]], $f1

; 32R2-LE-DAG:   lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 32R2-LE-DAG:   lw $[[R1:[0-9]+]], 4(${{[0-9]+}})
; 32R2-BE-DAG:   lw $[[R0:[0-9]+]], 4(${{[0-9]+}})
; 32R2-BE-DAG:   lw $[[R1:[0-9]+]], 0(${{[0-9]+}})
; 32R2-DAG:      mtc1 $[[R0]], $f0
; 32R2-DAG:      mthc1 $[[R1]], $f0

; 32R6-LE-DAG:   lw $[[R0:[0-9]+]], 0(${{[0-9]+}})
; 32R6-LE-DAG:   lw $[[R1:[0-9]+]], 4(${{[0-9]+}})
; 32R6-BE-DAG:   lw $[[R0:[0-9]+]], 4(${{[0-9]+}})
; 32R6-BE-DAG:   lw $[[R1:[0-9]+]], 0(${{[0-9]+}})
; 32R6-DAG:      mtc1 $[[R0]], $f0
; 32R6-DAG:      mthc1 $[[R1]], $f0

; 32R1-LDC1:     ldc1 $f0, 0(${{[0-9]+}})

; 32R2-LDXC1:    sll $[[OFFSET:[0-9]+]], $5, 3
; 32R2-LDXC1:    ldxc1 $f0, $[[OFFSET]]($4)

; 32R6-LDC1:     ldc1 $f0, 0(${{[0-9]+}})

define double @test_ldxc1(double* nocapture readonly %a, i32 %i) {
entry:
  %arrayidx = getelementptr inbounds double, double* %a, i32 %i
  %0 = load double, double* %arrayidx, align 8
  ret double %0
}

; ALL-LABEL: test_sdxc1:

; 32R1-DAG:      mfc1 $[[R0:[0-9]+]], $f12
; 32R1-DAG:      mfc1 $[[R1:[0-9]+]], $f13
; 32R1-DAG:      sw $[[R0]], 0(${{[0-9]+}})
; 32R1-DAG:      sw $[[R1]], 4(${{[0-9]+}})

; 32R2-DAG:      mfc1 $[[R0:[0-9]+]], $f12
; 32R2-DAG:      mfhc1 $[[R1:[0-9]+]], $f12
; 32R2-DAG:      sw $[[R0]], 0(${{[0-9]+}})
; 32R2-DAG:      sw $[[R1]], 4(${{[0-9]+}})

; 32R6-DAG:      mfc1 $[[R0:[0-9]+]], $f12
; 32R6-DAG:      mfhc1 $[[R1:[0-9]+]], $f12
; 32R6-DAG:      sw $[[R0]], 0(${{[0-9]+}})
; 32R6-DAG:      sw $[[R1]], 4(${{[0-9]+}})

; 32R1-LDC1:     sdc1 $f{{[0-9]+}}, 0(${{[0-9]+}})

; 32R2-LDXC1:    sll $[[OFFSET:[0-9]+]], $7, 3
; 32R2-LDXC1:    sdxc1 $f{{[0-9]+}}, $[[OFFSET]]($6)

; 32R6-LDC1:     sdc1 $f{{[0-9]+}}, 0(${{[0-9]+}})

define void @test_sdxc1(double %b, double* nocapture %a, i32 %i) {
entry:
  %arrayidx = getelementptr inbounds double, double* %a, i32 %i
  store double %b, double* %arrayidx, align 8
  ret void
}

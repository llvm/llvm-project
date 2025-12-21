# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s -o %t
# RUN: llvm-readobj -s %t | FileCheck %s

lu12i.w $a1, %gd_hi20(gd_abs)
# CHECK:      Symbol {
# CHECK:        Name: gd_abs
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT:   Size: 0
# CHECK-NEXT:   Binding: Global
# CHECK-NEXT:   Type: TLS
# CHECK-NEXT:   Other: 0
# CHECK-NEXT:   Section: Undefined
# CHECK-NEXT: }

pcalau12i $a1, %gd_pc_hi20(gd_pcrel)
# CHECK-NEXT: Symbol {
# CHECK-NEXT:   Name: gd_pcrel
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT:   Size: 0
# CHECK-NEXT:   Binding: Global
# CHECK-NEXT:   Type: TLS
# CHECK-NEXT:   Other: 0
# CHECK-NEXT:   Section: Undefined
# CHECK-NEXT: }

lu12i.w $a1, %ld_hi20(ld_abs)
# CHECK-NEXT: Symbol {
# CHECK-NEXT:   Name: ld_abs
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT:   Size: 0
# CHECK-NEXT:   Binding: Global
# CHECK-NEXT:   Type: TLS
# CHECK-NEXT:   Other: 0
# CHECK-NEXT:   Section: Undefined
# CHECK-NEXT: }

pcalau12i $a1, %ld_pc_hi20(ld_pcrel)
# CHECK-NEXT: Symbol {
# CHECK-NEXT:   Name: ld_pcrel
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT:   Size: 0
# CHECK-NEXT:   Binding: Global
# CHECK-NEXT:   Type: TLS
# CHECK-NEXT:   Other: 0
# CHECK-NEXT:   Section: Undefined
# CHECK-NEXT: }

lu12i.w $a1, %ie_hi20(ie_abs)
# CHECK-NEXT: Symbol {
# CHECK-NEXT:   Name: ie_abs
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT:   Size: 0
# CHECK-NEXT:   Binding: Global
# CHECK-NEXT:   Type: TLS
# CHECK-NEXT:   Other: 0
# CHECK-NEXT:   Section: Undefined
# CHECK-NEXT: }

pcalau12i $a1, %ie_pc_hi20(ie_pcrel)
# CHECK-NEXT: Symbol {
# CHECK-NEXT:   Name: ie_pcrel
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT:   Size: 0
# CHECK-NEXT:   Binding: Global
# CHECK-NEXT:   Type: TLS
# CHECK-NEXT:   Other: 0
# CHECK-NEXT:   Section: Undefined
# CHECK-NEXT: }

lu12i.w $a1, %le_hi20(le)
# CHECK-NEXT: Symbol {
# CHECK-NEXT:   Name: le
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT:   Size: 0
# CHECK-NEXT:   Binding: Global
# CHECK-NEXT:   Type: TLS
# CHECK-NEXT:   Other: 0
# CHECK-NEXT:   Section: Undefined
# CHECK-NEXT: }

pcalau12i $a1, %desc_pc_hi20(desc_pc)
# CHECK-NEXT: Symbol {
# CHECK-NEXT:   Name: desc_pc
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT:   Size: 0
# CHECK-NEXT:   Binding: Global
# CHECK-NEXT:   Type: TLS
# CHECK-NEXT:   Other: 0
# CHECK-NEXT:   Section: Undefined
# CHECK-NEXT: }

lu12i.w $a1, %desc_hi20(desc_abs)
# CHECK-NEXT: Symbol {
# CHECK-NEXT:   Name: desc_abs
# CHECK-NEXT:   Value: 0x0
# CHECK-NEXT:   Size: 0
# CHECK-NEXT:   Binding: Global
# CHECK-NEXT:   Type: TLS
# CHECK-NEXT:   Other: 0
# CHECK-NEXT:   Section: Undefined
# CHECK-NEXT: }

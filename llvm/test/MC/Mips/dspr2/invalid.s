# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -mattr=+dspr2 -show-encoding 2>%t1
# RUN: FileCheck %s < %t1
  append $2, $3, -1             # CHECK: :[[@LINE]]:18: error: expected 5-bit unsigned immediate
  append $2, $3, 32             # CHECK: :[[@LINE]]:18: error: expected 5-bit unsigned immediate
  balign $2, $3, -1             # CHECK: :[[@LINE]]:18: error: expected 2-bit unsigned immediate
  balign $2, $3, 4              # CHECK: :[[@LINE]]:18: error: expected 2-bit unsigned immediate
  precr_sra.ph.w $24, $25, -1   # CHECK: :[[@LINE]]:28: error: expected 5-bit unsigned immediate
  precr_sra.ph.w $24, $25, 32   # CHECK: :[[@LINE]]:28: error: expected 5-bit unsigned immediate
  precr_sra_r.ph.w $25 ,$26, -1 # CHECK: :[[@LINE]]:30: error: expected 5-bit unsigned immediate
  precr_sra_r.ph.w $25 ,$26, 32 # CHECK: :[[@LINE]]:30: error: expected 5-bit unsigned immediate
  prepend $2, $3, -1            # CHECK: :[[@LINE]]:19: error: expected 5-bit unsigned immediate
  prepend $2, $3, 32            # CHECK: :[[@LINE]]:19: error: expected 5-bit unsigned immediate
  shra.qb $3, $4, 8        # CHECK: :[[@LINE]]:19: error: expected 3-bit unsigned immediate
  shra.qb $3, $4, -1       # CHECK: :[[@LINE]]:19: error: expected 3-bit unsigned immediate
  shra_r.qb $3, $4, 8      # CHECK: :[[@LINE]]:21: error: expected 3-bit unsigned immediate
  shra_r.qb $3, $4, -1     # CHECK: :[[@LINE]]:21: error: expected 3-bit unsigned immediate
  shrl.ph $3, $4, 16       # CHECK: :[[@LINE]]:19: error: expected 4-bit unsigned immediate
  shrl.ph $3, $4, -1       # CHECK: :[[@LINE]]:19: error: expected 4-bit unsigned immediate

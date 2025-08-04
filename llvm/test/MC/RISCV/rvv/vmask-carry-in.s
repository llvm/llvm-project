# RUN: llvm-mc -triple=riscv64 -show-inst --mattr=+v %s \
# RUN:   --M no-aliases | FileCheck %s

# Check if there is a MCOperand for the carry-in mask.

vmerge.vvm v8, v4, v20, v0
# CHECK: <MCInst #{{[0-9]+}} VMERGE_VVM
# CHECK-COUNT-4: MCOperand Reg

vmerge.vxm v8, v4, a0, v0
# CHECK: <MCInst #{{[0-9]+}} VMERGE_VXM
# CHECK-COUNT-4: MCOperand Reg

vmerge.vim v8, v4, 15, v0
# CHECK: <MCInst #{{[0-9]+}} VMERGE_VIM
# CHECK-NEXT: MCOperand Reg
# CHECK-NEXT: MCOperand Reg
# CHECK-NEXT: MCOperand Imm
# CHECK-NEXT: MCOperand Reg

vadc.vvm v8, v4, v20, v0
# CHECK: <MCInst #{{[0-9]+}} VADC_VVM
# CHECK-COUNT-4: MCOperand Reg

vadc.vxm v8, v4, a0, v0
# CHECK: <MCInst #{{[0-9]+}} VADC_VXM
# CHECK-COUNT-4: MCOperand Reg

vadc.vim v8, v4, 15, v0
# CHECK: <MCInst #{{[0-9]+}} VADC_VIM
# CHECK-NEXT: MCOperand Reg
# CHECK-NEXT: MCOperand Reg
# CHECK-NEXT: MCOperand Imm
# CHECK-NEXT: MCOperand Reg

vmadc.vvm v8, v4, v20, v0
# CHECK: <MCInst #{{[0-9]+}} VMADC_VVM
# CHECK-COUNT-4: MCOperand Reg

vmadc.vxm v8, v4, a0, v0
# CHECK: <MCInst #{{[0-9]+}} VMADC_VXM
# CHECK-COUNT-4: MCOperand Reg

vmadc.vim v8, v4, 15, v0
# CHECK: <MCInst #{{[0-9]+}} VMADC_VIM
# CHECK-NEXT: MCOperand Reg
# CHECK-NEXT: MCOperand Reg
# CHECK-NEXT: MCOperand Imm
# CHECK-NEXT: MCOperand Reg

vsbc.vvm v8, v4, v20, v0
# CHECK: <MCInst #{{[0-9]+}} VSBC_VVM
# CHECK-COUNT-4: MCOperand Reg

vsbc.vxm v8, v4, a0, v0
# CHECK: <MCInst #{{[0-9]+}} VSBC_VXM
# CHECK-COUNT-4: MCOperand Reg

vmsbc.vvm v8, v4, v20, v0
# CHECK: <MCInst #{{[0-9]+}} VMSBC_VVM
# CHECK-COUNT-4: MCOperand Reg

vmsbc.vxm v8, v4, a0, v0
# CHECK: <MCInst #{{[0-9]+}} VMSBC_VXM
# CHECK-COUNT-4: MCOperand Reg

vfmerge.vfm v8, v4, fa0, v0
# CHECK: <MCInst #{{[0-9]+}} VFMERGE_VFM
# CHECK-COUNT-4: MCOperand Reg

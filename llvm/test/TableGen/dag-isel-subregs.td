// RUN: llvm-tblgen -gen-dag-isel -I %p/../../include -I %p/Common %s | FileCheck %s

include "reg-with-subregs-common.td"

// CHECK-LABEL: OPC_CheckOpcode, TARGET_VAL(ISD::EXTRACT_SUBVECTOR),
// CHECK: OPC_CheckChild1Integer, 0,
// CHECK: OPC_EmitStringInteger32, sub0_sub1,
def : Pat<(v2i32 (extract_subvector v32i32:$src, (i32 0))),
          (EXTRACT_SUBREG GPR_1024:$src, sub0_sub1)>;

// CHECK: OPC_CheckChild1Integer, 30,
// CHECK: OPC_EmitInteger32, 10|128,2/*266*/,
def : Pat<(v2i32 (extract_subvector v32i32:$src, (i32 15))),
          (EXTRACT_SUBREG GPR_1024:$src, sub30_sub31)>;

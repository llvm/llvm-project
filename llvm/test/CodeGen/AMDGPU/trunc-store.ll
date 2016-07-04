; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}truncstore_arg_v16i32_to_v16i8:
; SI: buffer_store_dwordx4
define void @truncstore_arg_v16i32_to_v16i8(<16 x i8> addrspace(1)* %out, <16 x i32> %in) {
  %trunc = trunc <16 x i32> %in to <16 x i8>
  store <16 x i8> %trunc, <16 x i8> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}truncstore_arg_v16i64_to_v16i8:
; SI: buffer_store_dwordx4
define void @truncstore_arg_v16i64_to_v16i8(<16 x i8> addrspace(1)* %out, <16 x i64> %in) {
  %trunc = trunc <16 x i64> %in to <16 x i8>
  store <16 x i8> %trunc, <16 x i8> addrspace(1)* %out
  ret void
}

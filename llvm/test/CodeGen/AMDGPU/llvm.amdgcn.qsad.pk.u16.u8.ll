; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i64 @llvm.amdgcn.qsad.pk.u16.u8(i64, i32, i64) #0

; GCN-LABEL: {{^}}v_qsad_pk_u16_u8:
; GCN: v_qsad_pk_u16_u8 v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
define void @v_qsad_pk_u16_u8(i64 addrspace(1)* %out, i64 %src) {
  %result= call i64 @llvm.amdgcn.qsad.pk.u16.u8(i64 %src, i32 100, i64 100) #0
  store i64 %result, i64 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v_qsad_pk_u16_u8_non_immediate:
; GCN: v_qsad_pk_u16_u8 v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
define void @v_qsad_pk_u16_u8_non_immediate(i64 addrspace(1)* %out, i64 %src, i32 %a, i64 %b) {
  %result= call i64 @llvm.amdgcn.qsad.pk.u16.u8(i64 %src, i32 %a, i64 %b) #0
  store i64 %result, i64 addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind readnone }

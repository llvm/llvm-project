; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; Test that add/sub with a constant is swapped to sub/add with negated
; constant to minimize code size.

; GCN-LABEL: {{^}}v_test_i32_x_sub_64:
; GCN: {{buffer|flat}}_load_dword [[X:v[0-9]+]]
; GCN: v_subrev_i32_e32 v{{[0-9]+}}, vcc, 64, [[X]]
define void @v_test_i32_x_sub_64(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %tid.ext
  %gep.out = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %tid.ext
  %x = load i32, i32 addrspace(1)* %gep
  %result = sub i32 %x, 64
  store i32 %result, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}v_test_i32_x_sub_64_multi_use:
; GCN: {{buffer|flat}}_load_dword [[X:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[Y:v[0-9]+]]
; GCN-DAG: v_subrev_i32_e32 v{{[0-9]+}}, vcc, 64, [[X]]
; GCN-DAG: v_subrev_i32_e32 v{{[0-9]+}}, vcc, 64, [[Y]]
define void @v_test_i32_x_sub_64_multi_use(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %tid.ext
  %gep.out = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %tid.ext
  %x = load volatile i32, i32 addrspace(1)* %gep
  %y = load volatile i32, i32 addrspace(1)* %gep
  %result0 = sub i32 %x, 64
  %result1 = sub i32 %y, 64
  store volatile i32 %result0, i32 addrspace(1)* %gep.out
  store volatile i32 %result1, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}v_test_i32_64_sub_x:
; GCN: {{buffer|flat}}_load_dword [[X:v[0-9]+]]
; GCN: v_sub_i32_e32 v{{[0-9]+}}, vcc, 64, [[X]]
define void @v_test_i32_64_sub_x(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %tid.ext
  %gep.out = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %tid.ext
  %x = load i32, i32 addrspace(1)* %gep
  %result = sub i32 64, %x
  store i32 %result, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}v_test_i32_x_sub_65:
; GCN: {{buffer|flat}}_load_dword [[X:v[0-9]+]]
; GCN: v_add_i32_e32 v{{[0-9]+}}, vcc, 0xffffffbf, [[X]]
define void @v_test_i32_x_sub_65(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %tid.ext
  %gep.out = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %tid.ext
  %x = load i32, i32 addrspace(1)* %gep
  %result = sub i32 %x, 65
  store i32 %result, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}v_test_i32_65_sub_x:
; GCN: {{buffer|flat}}_load_dword [[X:v[0-9]+]]
; GCN: v_sub_i32_e32 v{{[0-9]+}}, vcc, 0x41, [[X]]
define void @v_test_i32_65_sub_x(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %tid.ext
  %gep.out = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %tid.ext
  %x = load i32, i32 addrspace(1)* %gep
  %result = sub i32 65, %x
  store i32 %result, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}v_test_i32_x_sub_neg16:
; GCN: {{buffer|flat}}_load_dword [[X:v[0-9]+]]
; GCN: v_add_i32_e32 v{{[0-9]+}}, vcc, 16, [[X]]
define void @v_test_i32_x_sub_neg16(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %tid.ext
  %gep.out = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %tid.ext
  %x = load i32, i32 addrspace(1)* %gep
  %result = sub i32 %x, -16
  store i32 %result, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}v_test_i32_neg16_sub_x:
; GCN: {{buffer|flat}}_load_dword [[X:v[0-9]+]]
; GCN: v_sub_i32_e32 v{{[0-9]+}}, vcc, -16, [[X]]
define void @v_test_i32_neg16_sub_x(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %tid.ext
  %gep.out = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %tid.ext
  %x = load i32, i32 addrspace(1)* %gep
  %result = sub i32 -16, %x
  store i32 %result, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}v_test_i32_x_sub_neg17:
; GCN: {{buffer|flat}}_load_dword [[X:v[0-9]+]]
; GCN: v_add_i32_e32 v{{[0-9]+}}, vcc, 17, [[X]]
define void @v_test_i32_x_sub_neg17(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %tid.ext
  %gep.out = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %tid.ext
  %x = load i32, i32 addrspace(1)* %gep
  %result = sub i32 %x, -17
  store i32 %result, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}v_test_i32_neg17_sub_x:
; GCN: {{buffer|flat}}_load_dword [[X:v[0-9]+]]
; GCN: v_sub_i32_e32 v{{[0-9]+}}, vcc, 0xffffffef, [[X]]
define void @v_test_i32_neg17_sub_x(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %tid.ext
  %gep.out = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %tid.ext
  %x = load i32, i32 addrspace(1)* %gep
  %result = sub i32 -17, %x
  store i32 %result, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}s_test_i32_x_sub_64:
; GCN: s_load_dword [[X:s[0-9]+]]
; GCN: s_sub_i32 s{{[0-9]+}}, [[X]], 64
define void @s_test_i32_x_sub_64(i32 %x) #0 {
  %result = sub i32 %x, 64
  call void asm sideeffect "; use $0", "s"(i32 %result)
  ret void
}

; GCN-LABEL: {{^}}v_test_i16_x_sub_64:
; VI: {{buffer|flat}}_load_ushort [[X:v[0-9]+]]
; VI: v_subrev_u16_e32 v{{[0-9]+}}, 64, [[X]]
define void @v_test_i16_x_sub_64(i16 addrspace(1)* %out, i16 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep = getelementptr inbounds i16, i16 addrspace(1)* %in, i64 %tid.ext
  %gep.out = getelementptr inbounds i16, i16 addrspace(1)* %out, i64 %tid.ext
  %x = load i16, i16 addrspace(1)* %gep
  %result = sub i16 %x, 64
  store i16 %result, i16 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}v_test_i16_x_sub_64_multi_use:
; GCN: {{buffer|flat}}_load_ushort [[X:v[0-9]+]]
; GCN: {{buffer|flat}}_load_ushort [[Y:v[0-9]+]]
; VI-DAG: v_subrev_u16_e32 v{{[0-9]+}}, 64, [[X]]
; VI-DAG: v_subrev_u16_e32 v{{[0-9]+}}, 64, [[Y]]

; SI-DAG: v_subrev_i32_e32 v{{[0-9]+}}, vcc, 64, [[X]]
; SI-DAG: v_subrev_i32_e32 v{{[0-9]+}}, vcc, 64, [[Y]]
define void @v_test_i16_x_sub_64_multi_use(i16 addrspace(1)* %out, i16 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep = getelementptr inbounds i16, i16 addrspace(1)* %in, i64 %tid.ext
  %gep.out = getelementptr inbounds i16, i16 addrspace(1)* %out, i64 %tid.ext
  %x = load volatile i16, i16 addrspace(1)* %gep
  %y = load volatile i16, i16 addrspace(1)* %gep
  %result0 = sub i16 %x, 64
  %result1 = sub i16 %y, 64
  store volatile i16 %result0, i16 addrspace(1)* %gep.out
  store volatile i16 %result1, i16 addrspace(1)* %gep.out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

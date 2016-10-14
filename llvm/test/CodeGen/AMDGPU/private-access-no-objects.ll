; RUN: llc -O0 -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=OPTNONE %s

; GCN-LABEL: {{^}}store_to_undef:

; -O0 should assume spilling, so the input scratch resource descriptor
; -should be used directly without any copies.

; OPTNONE-NOT: s_mov_b32
; OPTNONE: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[0:3], s7 offen{{$}}
define void @store_to_undef() #0 {
  store volatile i32 0, i32* undef
  ret void
}

; GCN-LABEL: {{^}}store_to_inttoptr:
define void @store_to_inttoptr() #0 {
 store volatile i32 0, i32* inttoptr (i32 123 to i32*)
 ret void
}

; GCN-LABEL: {{^}}load_from_undef:
define void @load_from_undef() #0 {
  %ld = load volatile i32, i32* undef
  ret void
}

; GCN-LABEL: {{^}}load_from_inttoptr:
define void @load_from_inttoptr() #0 {
  %ld = load volatile i32, i32* inttoptr (i32 123 to i32*)
  ret void
}

attributes #0 = { nounwind }

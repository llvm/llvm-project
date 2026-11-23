; Test that there is no s_code_end padding if .text is otherwise empty.

; RUN: llc -mtriple=amdgpu12.00--amdpal < %s | FileCheck %s --check-prefixes=GCN

@globalVar = global i32 37

declare amdgpu_ps void @funcDecl()

; GCN-NOT: .fill
